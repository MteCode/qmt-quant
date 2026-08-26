"""基于 akshare 的数据源。

存在的唯一理由：**它能提供已退市股票的历史行情，而 QMT 不能。**
没有退市股数据，选股回测就必然带幸存者偏差。

⚠ 使用约束（akshare 是爬取公开网页，不是数据库服务）：
- 速度不可控，取决于上游网站负载与反爬策略
- 上游改版会导致接口失效，需等社区修复
- **绝不可用于实盘链路**，只用于盘后一次性拉取历史数据
"""
import logging
import time
from pathlib import Path

import pandas as pd

from ..core.constants import Interval
from ..core.objects import BarData
from ..utils.symbol import normalize, split_vt_symbol
from .base import BaseDataFeed

logger = logging.getLogger(__name__)

#: 内部周期 → akshare period
PERIOD_MAP = {
    Interval.DAILY: "daily",
    Interval.WEEKLY: "weekly",
    Interval.MONTHLY: "monthly",
}

#: akshare 中文列名 → 内部列名
COLUMN_MAP = {
    "日期": "datetime", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
}

#: 复权方式映射
ADJUST_MAP = {"front": "qfq", "back": "hfq", "none": ""}


class AkshareDataFeed(BaseDataFeed):
    """akshare 数据源，主要用于补齐 QMT 缺失的退市股行情"""

    def __init__(self, store_dir: str, dividend_type: str = "front",
                 request_interval: float = 2.0, max_retry: int = 3,
                 circuit_breaker: int = 15) -> None:
        """
        :param request_interval: 请求间隔（秒）。
            实测 0.3s 跑 254 个标的后被东财限流，直连与代理均被拒，
            且封禁持续存在。保守取 2.0s。
        :param max_retry: 单标的重试次数，指数退避
        :param circuit_breaker: 连续失败多少个就中止。
            连续失败通常意味着被限流，继续打只会延长封禁时间。
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.dividend_type = dividend_type
        self.request_interval = request_interval
        self.max_retry = max_retry
        self.circuit_breaker = circuit_breaker

    def _path(self, vt_symbol: str, interval: Interval) -> Path:
        symbol, exchange = split_vt_symbol(vt_symbol)
        d = self.store_dir / interval.value / exchange.value
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol}.parquet"

    def has_data(self, vt_symbol: str, interval: Interval) -> bool:
        return self._path(normalize(vt_symbol), interval).exists()

    # ------------------------------------------------------------ 下载

    def download_history(self, vt_symbols: list[str], start: str, end: str,
                         interval: Interval = Interval.DAILY,
                         skip_existing: bool = False,
                         progress=None, timeout: float = 45.0) -> dict:
        """下载历史行情。

        :param timeout: 单个标的的超时（秒）。akshare 偶发卡死，必须设上限，
            否则一只股票就能把整批任务挂住。
        """
        import concurrent.futures as cf

        import akshare as ak

        period = PERIOD_MAP.get(interval)
        if period is None:
            raise ValueError(f"akshare 数据源不支持周期 {interval.value}")

        adjust = ADJUST_MAP.get(self.dividend_type, "qfq")
        s, e = start.replace("-", ""), end.replace("-", "")
        result = {"ok": [], "failed": [], "skipped": []}
        total = len(vt_symbols)
        consecutive_failures = 0

        for i, raw in enumerate(vt_symbols, 1):
            vt_symbol = normalize(raw)
            if progress:
                progress(i, total, vt_symbol)

            if skip_existing and self.has_data(vt_symbol, interval):
                result["skipped"].append(vt_symbol)
                continue

            symbol, _ = split_vt_symbol(vt_symbol)
            df = self._fetch_with_retry(ak, cf, symbol, period, s, e, adjust,
                                        timeout, vt_symbol)

            if df is None or df.empty:
                result["failed"].append(vt_symbol)
                consecutive_failures += 1
                # 连续失败基本等于被限流，继续打只会延长封禁
                if consecutive_failures >= self.circuit_breaker:
                    logger.error(
                        "连续 %d 个标的失败，判定为上游限流，中止本次下载。"
                        "建议等待 30~60 分钟后加 --resume 重跑",
                        consecutive_failures)
                    result["failed"].extend(
                        normalize(x) for x in vt_symbols[i:])
                    break
                continue

            consecutive_failures = 0
            df = self._normalize(df)
            df.to_parquet(self._path(vt_symbol, interval))
            result["ok"].append(vt_symbol)
            time.sleep(self.request_interval)

        logger.info("akshare 下载完成 %s：成功 %d 跳过 %d 失败 %d", interval.value,
                    len(result["ok"]), len(result["skipped"]), len(result["failed"]))
        return result

    def _fetch_with_retry(self, ak, cf, symbol: str, period: str, s: str, e: str,
                          adjust: str, timeout: float, vt_symbol: str):
        """取单个标的，失败按指数退避重试。

        注意：**退市股只有东财有数据**，新浪接口对退市代码返回空。
        所以这里不做数据源回退，只能重试等限流解除。
        """
        last_err = None
        for attempt in range(self.max_retry):
            if attempt:
                delay = self.request_interval * (2 ** attempt)
                logger.debug("重试 %s（第 %d 次），等待 %.1fs", vt_symbol, attempt, delay)
                time.sleep(delay)
            try:
                # akshare 偶发无限阻塞，用线程池强制超时
                with cf.ThreadPoolExecutor(1) as ex:
                    return ex.submit(
                        ak.stock_zh_a_hist, symbol=symbol, period=period,
                        start_date=s, end_date=e, adjust=adjust,
                    ).result(timeout=timeout)
            except cf.TimeoutError:
                last_err = f"超时>{timeout:.0f}s"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {str(exc)[:80]}"

        logger.warning("下载失败 %s（重试 %d 次）: %s",
                       vt_symbol, self.max_retry, last_err)
        return None

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """中文列名转内部列名，日期设为索引"""
        df = df.rename(columns=COLUMN_MAP)
        df["datetime"] = pd.to_datetime(df["datetime"])
        keep = [c for c in ["open", "high", "low", "close", "volume", "amount"]
                if c in df.columns]
        return df.set_index("datetime")[keep].sort_index()

    # ------------------------------------------------------------ 读取

    def load_bars(self, vt_symbols: list[str], start: str, end: str,
                  interval: Interval = Interval.DAILY) -> list[BarData]:
        bars: list[BarData] = []
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)

        for raw in vt_symbols:
            vt_symbol = normalize(raw)
            path = self._path(vt_symbol, interval)
            if not path.exists():
                continue

            df = pd.read_parquet(path)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            symbol, exchange = split_vt_symbol(vt_symbol)

            for dt, row in df.iterrows():
                volume = float(row.get("volume", 0) or 0)
                bars.append(BarData(
                    symbol=symbol, exchange=exchange, datetime=dt.to_pydatetime(),
                    interval=interval,
                    open_price=float(row["open"]), high_price=float(row["high"]),
                    low_price=float(row["low"]), close_price=float(row["close"]),
                    volume=volume, turnover=float(row.get("amount", 0) or 0),
                    suspended=(volume == 0),
                    gateway_name="AK",
                ))

        bars.sort(key=lambda b: (b.datetime, b.vt_symbol))
        return bars


# ---------------------------------------------------------------- 标的名单


def fetch_delisted_stocks() -> pd.DataFrame:
    """取沪深两市已退市股票名单。

    这是消除幸存者偏差的关键数据 —— QMT 中退市股完全不存在。

    :return: DataFrame[vt_symbol, name, listing_date, delist_date]
    """
    import akshare as ak

    rows = []

    try:
        sz = ak.stock_info_sz_delist(symbol="终止上市公司")
        for r in sz.itertuples():
            rows.append({
                "vt_symbol": normalize(str(r.证券代码).zfill(6) + ".SZ"),
                "name": r.证券简称,
                "listing_date": pd.to_datetime(r.上市日期, errors="coerce"),
                "delist_date": pd.to_datetime(r.终止上市日期, errors="coerce"),
            })
    except Exception:
        logger.exception("获取深市退市名单失败")

    try:
        sh = ak.stock_info_sh_delist()
        for r in sh.itertuples():
            rows.append({
                "vt_symbol": normalize(str(r.公司代码).zfill(6) + ".SH"),
                "name": r.公司简称,
                "listing_date": pd.to_datetime(r.上市日期, errors="coerce"),
                # 沪市该接口给的是「暂停上市日期」，近似作为退市日
                "delist_date": pd.to_datetime(r.暂停上市日期, errors="coerce"),
            })
    except Exception:
        logger.exception("获取沪市退市名单失败")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["vt_symbol"]).reset_index(drop=True)
    logger.info("已获取退市股 %d 只", len(df))
    return df


def fetch_index_constituents(index_code: str = "000300") -> pd.DataFrame:
    """取指数当前成分股。

    ⚠ 只有当前快照，不含历史调整记录。
    """
    import akshare as ak

    df = ak.index_stock_cons_csindex(symbol=index_code)
    out = pd.DataFrame({
        "vt_symbol": [
            normalize(str(c).zfill(6) + ("." + ("SH" if str(e).startswith("上海") else "SZ")))
            for c, e in zip(df["成分券代码"], df["交易所"])
        ],
        "name": df["成分券名称"],
        "date": pd.to_datetime(df["日期"]),
    })
    return out
