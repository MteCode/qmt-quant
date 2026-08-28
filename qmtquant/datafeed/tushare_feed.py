"""Tushare Pro 数据源。

## 它补的是 QMT 补不了的那块

QMT 的行情数据又快又全，但有两样东西它给不了：

1. **历史指数成分股** —— QMT 只有当前快照。而沪深300 每半年调样，
   用今天的名单回测 2021 年，等于让策略提前知道「哪些股票后来会因为
   涨得好而被纳入指数」。实测同一组均值回归参数，加不加纳入日过滤，
   总收益从 **+320.33% 变成 -4.49%** —— 320 个百分点全是偏差。
2. **逐日估值因子**（PE/PB/股息率/换手率）—— 财报数据有公告日滞后，
   而 `daily_basic` 是每日更新的，没有 point-in-time 麻烦。

## 积分门槛

本模块用到的接口都要 **2000 积分（200 元/年）**：
`index_weight` / `daily_basic` / `fina_indicator` / `index_daily`。

## 关于 token

token 是付费凭证，**绝不能写进代码或提交到 git**。
优先读环境变量 ``TUSHARE_TOKEN``，其次读 config.yaml（已在 .gitignore）。

## 代码格式

Tushare 的代码格式（``000001.SZ``）与 xtquant 完全一致，
所以直接复用 ``utils.symbol`` 的转换函数，不需要单独写一套。
"""
import logging
import os
import time
from collections import deque

import pandas as pd

from ..config import TushareConfig
from ..utils.symbol import from_xt_symbol, normalize, to_xt_symbol

logger = logging.getLogger(__name__)

#: 常用指数的 Tushare 代码。注意与 xtquant 一致，都是 `.SH` / `.SZ`
INDEX_CODES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "000016.SH": "上证50",
    "399006.SZ": "创业板指",
}

#: index_weight 单次请求的最大返回行数（官方限制）
_MAX_ROWS = 4000


class TushareError(RuntimeError):
    """Tushare 调用失败（含积分不足、token 无效、限流）"""


def resolve_token(cfg: TushareConfig | None = None) -> str:
    """取 token：环境变量优先，其次配置文件。

    环境变量优先是有意的 —— 这样在 CI 或临时终端里可以覆盖，
    而不必去改一个已被 .gitignore 的文件。
    """
    env = os.environ.get("TUSHARE_TOKEN", "").strip()
    if env:
        return env
    if cfg is not None and cfg.token.strip():
        return cfg.token.strip()
    raise TushareError(
        "未配置 Tushare token。二选一：\n"
        "  1. 设环境变量  TUSHARE_TOKEN=你的token\n"
        "  2. 在 config/config.yaml 写 tushare.token（该文件不会入库）\n"
        "token 在 https://tushare.pro/user/token 获取")


class TushareClient:
    """带限流与重试的 Tushare Pro 客户端。

    为什么要自己做限流：批量拉历史时很容易在几秒内打出几百次请求，
    触发官方风控后会被临时封禁，而错误信息不明显、
    表现为部分接口静默返回空 DataFrame —— 比直接报错更难排查。
    """

    def __init__(self, cfg: TushareConfig | None = None) -> None:
        self.cfg = cfg or TushareConfig()
        self._token = resolve_token(self.cfg)
        self._pro = None
        #: 最近一分钟内的调用时间戳，用于滑动窗口限流
        self._calls: deque[float] = deque()

    @property
    def pro(self):
        """惰性初始化。import tushare 本身有 1s 级开销，
        且未配置 token 时不应在 import 阶段就炸掉"""
        if self._pro is None:
            try:
                import tushare as ts
            except ImportError as e:
                raise TushareError(
                    "未安装 tushare，请先 pip install tushare") from e
            self._pro = ts.pro_api(self._token)
        return self._pro

    def _throttle(self) -> None:
        """滑动窗口限流：保证任意 60 秒内不超过 calls_per_minute 次"""
        limit = self.cfg.calls_per_minute
        if limit <= 0:
            return
        # 用循环而非递归：等待次数没有上界，递归会撞 Python 栈深度限制
        while True:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= 60.0:
                self._calls.popleft()
            if len(self._calls) < limit:
                break
            sleep = 60.0 - (now - self._calls[0]) + 0.05
            if sleep <= 0:
                break
            logger.debug("触发本地限流，等待 %.1fs", sleep)
            time.sleep(sleep)
            # sleep 时长是精确算到 _calls[0] 过期的，睡醒它必然已过期。
            # 直接弹出而不是重新读时钟，避免依赖 sleep 的实际精度
            self._calls.popleft()
        self._calls.append(time.monotonic())

    def query(self, api: str, **params) -> pd.DataFrame:
        """调用任意接口，带限流与指数退避重试。

        :raises TushareError: 重试耗尽后仍失败
        """
        last: Exception | None = None
        for attempt in range(1, self.cfg.max_retry + 1):
            self._throttle()
            try:
                df = getattr(self.pro, api)(**params)
                return df if df is not None else pd.DataFrame()
            except Exception as e:  # noqa: BLE001 —— 三方库异常类型不稳定
                last = e
                msg = str(e)
                # 积分不足是配置问题，重试多少次都没用，直接抛
                if "积分" in msg or "权限" in msg or "没有接口访问权限" in msg:
                    raise TushareError(
                        f"{api} 调用被拒（很可能是积分不足）：{msg}") from e
                delay = self.cfg.retry_base_delay * (2 ** (attempt - 1))
                logger.warning("%s 第 %d/%d 次失败：%s，%.1fs 后重试",
                               api, attempt, self.cfg.max_retry, msg, delay)
                if attempt < self.cfg.max_retry:
                    time.sleep(delay)
        raise TushareError(f"{api} 重试 {self.cfg.max_retry} 次后仍失败：{last}")

    # ------------------------------------------------------------ 连通性

    def check(self) -> dict:
        """探活：确认 token 有效且 2000 积分接口可用。

        故意用 index_weight 而不是 stock_basic —— 后者 120 积分就能调，
        通过了也说明不了 2000 积分档是否生效。
        """
        info: dict = {"token_ok": False, "points_2000_ok": False, "notes": []}
        try:
            basic = self.query("stock_basic", exchange="", list_status="L",
                               fields="ts_code")
            info["token_ok"] = not basic.empty
            info["stock_count"] = len(basic)
        except TushareError as e:
            info["notes"].append(f"基础接口失败：{e}")
            return info

        try:
            w = self.query("index_weight", index_code="000300.SH",
                           start_date="20240101", end_date="20240131")
            info["points_2000_ok"] = not w.empty
            if w.empty:
                info["notes"].append("index_weight 返回空，可能积分未到账")
        except TushareError as e:
            info["notes"].append(f"index_weight 不可用：{e}")
        return info


class TushareFeed:
    """面向本项目语义的封装。

    与 ``TushareClient`` 的分工：Client 只管「可靠地调通接口」，
    Feed 负责「翻译成本项目的概念」—— vt_symbol、月度成分名单、因子表。
    """

    def __init__(self, cfg: TushareConfig | None = None,
                 client: TushareClient | None = None) -> None:
        self.client = client or TushareClient(cfg)

    # ------------------------------------------------------- 历史成分股

    def index_weight(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        """拉取指数历史成分与权重。

        ``index_weight`` 是**月度**数据，且单次最多返回 4000 行 ——
        沪深300 每月 300 行，一次请求最多覆盖 13 个月。
        为了不踩这个上限，这里按**自然月**逐月请求。

        :param index_code: Tushare 指数代码，如 ``000300.SH``
        :param start: ``YYYY-MM-DD`` 或 ``YYYYMMDD``
        :return: 列 ``[date, symbol, weight]``，symbol 为 vt_symbol，
            date 为该期成分名单的生效日（月末调整日）
        """
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        empty = pd.DataFrame(columns=["date", "symbol", "weight"])
        # 必须先判空再对齐月初：start 落在 end 之后时，两者对齐到同一个
        # 月初会凭空产生一次请求（如 03-05 ~ 03-01 都会变成 03-01）
        if start_ts > end_ts:
            return empty

        months = pd.date_range(start_ts.normalize().replace(day=1),
                               end_ts, freq="MS")
        if len(months) == 0:
            return empty

        frames = []
        for i, m in enumerate(months, 1):
            m_end = m + pd.offsets.MonthEnd(0)
            df = self.client.query(
                "index_weight", index_code=index_code,
                start_date=m.strftime("%Y%m%d"),
                end_date=m_end.strftime("%Y%m%d"))
            if df.empty:
                logger.debug("%s %s 无成分数据", index_code, m.strftime("%Y-%m"))
                continue
            if len(df) >= _MAX_ROWS:
                logger.warning("%s %s 返回 %d 行，已达单次上限，数据可能被截断",
                               index_code, m.strftime("%Y-%m"), len(df))
            frames.append(df)
            if i % 12 == 0:
                logger.info("%s 已拉取 %d/%d 个月", index_code, i, len(months))

        if not frames:
            return empty

        raw = pd.concat(frames, ignore_index=True)
        out = pd.DataFrame({
            "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d"),
            "symbol": raw["con_code"].map(from_xt_symbol),
            "weight": pd.to_numeric(raw["weight"], errors="coerce"),
        })
        # 同一天可能因重复请求出现重复行，去重后按日期+代码排序
        out = (out.drop_duplicates(subset=["date", "symbol"])
                  .sort_values(["date", "symbol"])
                  .reset_index(drop=True))
        return out

    # ------------------------------------------------------- 估值因子

    def daily_basic(self, trade_date: str,
                    fields: str | None = None) -> pd.DataFrame:
        """某个交易日的全市场估值因子。

        这是逐日快照，天然 point-in-time —— 不像财报要处理公告日滞后。

        :param trade_date: ``YYYY-MM-DD`` 或 ``YYYYMMDD``
        :return: 含 ``symbol``（vt_symbol）列的 DataFrame
        """
        default = ("ts_code,trade_date,close,turnover_rate,turnover_rate_f,"
                   "volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,"
                   "total_share,float_share,free_share,total_mv,circ_mv")
        df = self.client.query(
            "daily_basic",
            trade_date=pd.Timestamp(trade_date).strftime("%Y%m%d"),
            fields=fields or default)
        if df.empty:
            return df
        df = df.copy()
        df["symbol"] = df["ts_code"].map(from_xt_symbol)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        return df

    def fina_indicator(self, vt_symbol: str) -> pd.DataFrame:
        """单只股票的全历史财务指标。

        含 ``ann_date``（公告日），必须按它做 point-in-time 过滤 ——
        按报告期取数等于提前三个月知道年报，实测茅台年报平均滞后 100 天以上。
        """
        df = self.client.query("fina_indicator",
                               ts_code=to_xt_symbol(normalize(vt_symbol)))
        if df.empty:
            return df
        df = df.copy()
        for col in ("ann_date", "end_date", "f_ann_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="%Y%m%d",
                                         errors="coerce")
        df["symbol"] = normalize(vt_symbol)
        return df

    def stock_basic(self) -> pd.DataFrame:
        """全市场股票基础信息，含**行业分类**。

        行业是因子中性化的必需输入 —— 不做行业中性化的话，
        「低换手因子」实测选出 15/20 是银行，所谓因子显著
        很可能只是行业暴露，不是选股能力。

        Tushare 的 industry 字段是申万一级的简化版，全历史静态
        （不随时间变化）。这有个已知局限：公司转型换行业时，
        历史区间也会被打上现在的标签。对中性化来说影响可接受 ——
        行业标签错几个，好过完全不做中性化。
        """
        df = self.client.query(
            "stock_basic", exchange="", list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date")
        # 退市股也要，否则历史成分里的老票没有行业标签
        for status in ("D", "P"):
            extra = self.client.query(
                "stock_basic", exchange="", list_status=status,
                fields="ts_code,symbol,name,area,industry,market,list_date")
            if not extra.empty:
                df = pd.concat([df, extra], ignore_index=True)

        if df.empty:
            return df
        df = df.drop_duplicates(subset=["ts_code"], keep="first").copy()
        df["vt_symbol"] = df["ts_code"].map(from_xt_symbol)
        df["list_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d",
                                         errors="coerce")
        return df

    def trade_dates(self, start: str, end: str,
                    exchange: str = "SSE") -> list[pd.Timestamp]:
        """交易日历。用于按日循环拉 daily_basic 时跳过非交易日。"""
        df = self.client.query(
            "trade_cal", exchange=exchange,
            start_date=pd.Timestamp(start).strftime("%Y%m%d"),
            end_date=pd.Timestamp(end).strftime("%Y%m%d"),
            is_open="1")
        if df.empty:
            return []
        return sorted(pd.to_datetime(df["cal_date"], format="%Y%m%d").tolist())
