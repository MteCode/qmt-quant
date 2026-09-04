"""数据清洗 —— 把校验发现的问题真正修掉。

`validator.py` 的原则是「只报告，不静默修改」，因为自动修复比脏数据更危险：
你不知道它改了什么。本模块是那条原则的另一半 —— **显式调用、逐条记录**。

每次清洗都返回一份动作清单，说明改了哪些行、依据什么规则。
不留记录的清洗等于把问题从「看得见的脏数据」变成「看不见的假数据」。

## 清洗规则的取舍

只做**有唯一正确答案**的修复：

- 重复时间戳 -> 保留最后一条（后到的数据覆盖先到的，符合增量下载语义）
- 时间未升序 -> 排序（顺序本身不含信息）
- 停牌日成交量为 0 而价格仍在变 -> 保留，仅标记（可能是复权因子调整）

**不做**需要猜测的修复：

- 价格为 0 或负 -> 不填补，直接剔除该行并记录。填补等于凭空造价格
- 涨跌幅超限 -> 不修改，仅标记。可能是真实极端行情，也可能是复权跳变，
  分不清就不该动
- 缺失交易日 -> 不插值。停牌与数据缺失在结果上无法区分，
  插值会把停牌期伪造成正常交易
"""
import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CleanAction:
    """一次清洗动作的记录。"""
    rule: str
    n_rows: int
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.rule}: {self.n_rows} 行" + (f"（{self.detail}）"
                                                   if self.detail else "")


@dataclass
class CleanResult:
    df: pd.DataFrame
    actions: list = field(default_factory=list)
    #: 只标记未修改的问题，供下游决定是否采用
    flags: list = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.actions)

    def summary(self) -> str:
        if not self.actions and not self.flags:
            return "无需清洗"
        parts = [str(a) for a in self.actions]
        if self.flags:
            parts.append("仅标记未改：" + "; ".join(str(f) for f in self.flags))
        return " | ".join(parts)


def clean_bars(df: pd.DataFrame, vt_symbol: str,
               drop_nonpositive: bool = True) -> CleanResult:
    """清洗 K 线。

    :param drop_nonpositive: 价格非正的行是否剔除。默认剔除 ——
        这类行无法用于任何计算，留着只会让下游各自处理、各处理各的
    """
    actions, flags = [], []
    if df is None or df.empty:
        return CleanResult(df=df if df is not None else pd.DataFrame())

    out = df.copy()
    n0 = len(out)

    # --- 重复时间戳：保留最后一条。增量下载时后到的数据更新，应覆盖先到的
    dup = out.index.duplicated(keep="last")
    if dup.any():
        out = out[~dup]
        actions.append(CleanAction("去重", int(dup.sum()), "保留最后一条"))

    # --- 排序。顺序本身不含信息，可以放心修
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
        actions.append(CleanAction("按时间排序", len(out)))

    price_cols = [c for c in ("open", "high", "low", "close")
                  if c in out.columns]

    # --- 价格非正：剔除而不填补。填补等于凭空造价格
    if drop_nonpositive and price_cols:
        bad = pd.Series(False, index=out.index)
        for c in price_cols:
            bad |= out[c].notna() & (out[c] <= 0)
        if bad.any():
            out = out[~bad]
            actions.append(CleanAction("剔除非正价格", int(bad.sum()),
                                       "无法用于计算，不填补"))

    # --- 成交量为负：置零并记录。负成交量一定是错误，0 是安全的下界
    if "volume" in out.columns:
        neg = out["volume"].notna() & (out["volume"] < 0)
        if neg.any():
            out.loc[neg, "volume"] = 0
            actions.append(CleanAction("负成交量置零", int(neg.sum())))

    # --- OHLC 逻辑矛盾：只标记不修。改哪一个都是猜
    if {"high", "low"} <= set(out.columns):
        bad = out["high"].notna() & out["low"].notna() & (out["high"] < out["low"])
        if bad.any():
            flags.append(CleanAction("最高价低于最低价", int(bad.sum()),
                                     "改哪一边都是猜，交由下游判断"))
    if {"high", "open", "close"} <= set(out.columns):
        bad = (out["high"].notna()
               & ((out["high"] < out["open"]) | (out["high"] < out["close"])))
        if bad.any():
            flags.append(CleanAction("最高价低于开盘或收盘", int(bad.sum())))

    # --- 涨跌幅异常：只标记。真实极端行情与复权跳变分不清，不该动
    if "close" in out.columns and len(out) > 1:
        from ..core.constants import get_price_limit
        limit = get_price_limit(vt_symbol)
        chg = out["close"].pct_change()
        bad = chg.abs() > limit * 1.5
        if bad.any():
            flags.append(CleanAction(
                f"涨跌幅超 {limit:.0%} 的 1.5 倍", int(bad.sum()),
                f"最大 {chg.abs().max():.1%}，多为复权跳变"))

    if len(out) != n0:
        logger.debug("%s 清洗后 %d -> %d 行", vt_symbol, n0, len(out))
    return CleanResult(df=out, actions=actions, flags=flags)


def _to_date(s: pd.Series) -> pd.Series:
    """转成日期。已是 datetime 就原样返回。

    各数据源的日期格式不一：QMT 给 yyyymmdd 整数，Tushare 经 pandas
    读回来已是 datetime64。对已是日期的列强行按 "%Y%m%d" 解析字符串，
    会得到 "2016-01-04 00:00:00" 这种串而全部解析失败 ——
    资金流 318 万行曾因此被整体清空。
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")


def clean_financial(df: pd.DataFrame, vt_symbol: str,
                    table: str = "") -> CleanResult:
    """清洗财务报表。

    财报的脏数据形态与 K 线完全不同，核心是**时间语义**：

    - ``m_timetag`` 是报告期（如 20240331 表示 2024 一季报）
    - ``m_anntime`` 是公告日 —— 这才是数据「可获得」的时点

    用报告期当可用时点是典型的前视偏差：2024 一季报要到 4 月底才公告，
    在 3 月 31 日用它选股等于开了天眼。因此公告日缺失的行必须剔除，
    而不是拿报告期顶替。
    """
    actions, flags = [], []
    if df is None or df.empty:
        return CleanResult(df=df if df is not None else pd.DataFrame())

    out = df.copy()

    # --- 报告期与公告日转成日期类型，无法解析的行剔除
    for col in ("m_timetag", "m_anntime"):
        if col in out.columns:
            out[col] = _to_date(out[col])

    if "m_timetag" in out.columns:
        bad = out["m_timetag"].isna()
        if bad.any():
            out = out[~bad]
            actions.append(CleanAction("剔除报告期无法解析", int(bad.sum())))

    # --- 公告日缺失：剔除。用报告期顶替就是前视偏差
    if "m_anntime" in out.columns:
        bad = out["m_anntime"].isna()
        if bad.any():
            out = out[~bad]
            actions.append(CleanAction(
                "剔除公告日缺失", int(bad.sum()),
                "无公告日则无法确定可获得时点，用报告期顶替会造成前视"))

        # --- 公告日早于报告期：不可能，数据错误
        if "m_timetag" in out.columns:
            bad = (out["m_anntime"].notna() & out["m_timetag"].notna()
                   & (out["m_anntime"] < out["m_timetag"]))
            if bad.any():
                out = out[~bad]
                actions.append(CleanAction(
                    "剔除公告日早于报告期", int(bad.sum()), "时间逻辑不可能"))

        # --- 公告日在未来：数据源错误
        future = out["m_anntime"] > pd.Timestamp.today()
        if future.any():
            out = out[~future]
            actions.append(CleanAction("剔除公告日在未来", int(future.sum())))

    # --- 同一报告期多条记录：保留公告日最晚的（财报会修订，以最新为准）
    if "m_timetag" in out.columns and len(out) > 1:
        sort_cols = ["m_timetag"] + (["m_anntime"]
                                     if "m_anntime" in out.columns else [])
        out = out.sort_values(sort_cols)
        dup = out["m_timetag"].duplicated(keep="last")
        if dup.any():
            out = out[~dup]
            actions.append(CleanAction("同一报告期去重", int(dup.sum()),
                                       "保留公告日最晚的（财报会修订）"))

    # --- 全空列只标记。财报字段多，不同行业填报口径不同，空列是常态
    empty = [c for c in out.columns if out[c].isna().all()]
    if empty:
        flags.append(CleanAction("整列为空", len(empty),
                                 f"{', '.join(map(str, empty[:5]))} 等"))

    return CleanResult(df=out, actions=actions, flags=flags)


def clean_flow(df: pd.DataFrame, name: str = "") -> CleanResult:
    """清洗资金流（龙虎榜、两融）。

    这类数据是**事件流**：每行一条记录，同一天同一标的可能多行
    （龙虎榜有多个营业部席位）。因此不能按「日期+标的」去重 ——
    那会丢掉真实记录。只去完全重复的整行。
    """
    actions, flags = [], []
    if df is None or df.empty:
        return CleanResult(df=df if df is not None else pd.DataFrame())

    out = df.copy()

    if "trade_date" in out.columns:
        out["trade_date"] = _to_date(out["trade_date"])
        bad = out["trade_date"].isna()
        if bad.any():
            out = out[~bad]
            actions.append(CleanAction("剔除日期无法解析", int(bad.sum())))

        future = out["trade_date"] > pd.Timestamp.today()
        if future.any():
            out = out[~future]
            actions.append(CleanAction("剔除日期在未来", int(future.sum())))

    # --- 只去完全重复的整行。同一天同一标的多行是正常的（多席位）
    dup = out.duplicated(keep="first")
    if dup.any():
        out = out[~dup]
        actions.append(CleanAction("去除完全重复行", int(dup.sum()),
                                   "同日同标的多行属正常，不按键去重"))

    if "trade_date" in out.columns:
        out = out.sort_values("trade_date")

    return CleanResult(df=out, actions=actions, flags=flags)


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """把 QMT 原始日线整理成统一 schema。

    QMT 的原始表用 yyyymmdd 整数作索引、另有一列毫秒时间戳，
    还带着股票用不到的期货字段（结算价、持仓量）。统一成
    `date` 索引 + 标准列，下游不必各自适配。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # 索引可能是 yyyymmdd 整数或字符串，统一成日期
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        out.index = pd.to_datetime(idx.astype(str), format="%Y%m%d",
                                   errors="coerce")
    out = out[out.index.notna()]
    out.index.name = "date"

    # time 列与索引重复，期货字段对股票无意义
    drop = [c for c in ("time", "settelementPrice", "openInterest")
            if c in out.columns]
    if drop:
        out = out.drop(columns=drop)

    return out
