"""基于 xtdata 的行情数据源（miniQMT / 大 QMT）。

xtdata 只有在 QMT 客户端运行时才能取数，因此本类做了两件事：
1. 下载 → 落地为 Parquet，脱离客户端也能回测
2. 优先读本地 Parquet，缺失才回源下载
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from ..core.constants import Interval
from ..core.objects import BarData
from ..utils.symbol import from_xt_symbol, normalize, split_vt_symbol, to_xt_symbol
from .base import BaseDataFeed

logger = logging.getLogger(__name__)

#: 内部周期 → xtdata period 字符串
PERIOD_MAP = {
    Interval.MINUTE: "1m",
    Interval.MINUTE_5: "5m",
    Interval.MINUTE_15: "15m",
    Interval.MINUTE_30: "30m",
    Interval.HOUR: "1h",
    Interval.DAILY: "1d",
    Interval.WEEKLY: "1w",
    Interval.MONTHLY: "1mon",
}

#: pandas resample 规则。W-FRI = 以周五为周终点，符合 A 股周线惯例
RESAMPLE_RULE = {
    Interval.WEEKLY: "W-FRI",
    Interval.MONTHLY: "ME",
}

#: 常用板块名称，xtdata 的板块名是中文
SECTOR_HS300 = "沪深300"
SECTOR_ZZ500 = "中证500"
SECTOR_SZ50 = "上证50"
SECTOR_ALL_A = "沪深A股"


class XtDataFeed(BaseDataFeed):
    """xtdata 数据源"""

    def __init__(self, store_dir: str, dividend_type: str = "front") -> None:
        """
        :param dividend_type: front(前复权) / back(后复权) / none(不复权)
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.dividend_type = dividend_type

    def _path(self, vt_symbol: str, interval: Interval) -> Path:
        symbol, exchange = split_vt_symbol(vt_symbol)
        d = self.store_dir / interval.value / exchange.value
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol}.parquet"

    def has_data(self, vt_symbol: str, interval: Interval) -> bool:
        """本地是否已有该标的数据，用于断点续传"""
        return self._path(normalize(vt_symbol), interval).exists()

    # ------------------------------------------------------------ 板块成分股

    def get_sector_stocks(self, sector: str = SECTOR_HS300) -> list[str]:
        """取板块成分股，返回 vt_symbol 列表。

        注意：xtdata 返回的是**当前**成分股，不是历史成分。
        用它回测历史区间会有幸存者偏差 —— 被调出指数的股票不在列表里。
        """
        from xtquant import xtdata

        codes = xtdata.get_stock_list_in_sector(sector)
        if not codes:
            logger.warning("板块 %s 未取到成分股。请确认 QMT 客户端已登录，"
                           "并在客户端「板块」中下载过该板块数据", sector)
            return []

        result = []
        for code in codes:
            try:
                result.append(from_xt_symbol(code))
            except (KeyError, ValueError):
                # 忽略非沪深北的标的（如指数、期货代码）
                logger.debug("跳过无法解析的代码: %s", code)
        logger.info("板块 %s 共 %d 只成分股", sector, len(result))
        return sorted(result)

    def get_stock_names(self, vt_symbols: list[str]) -> dict[str, str]:
        """批量取标的名称，用于报告展示"""
        from xtquant import xtdata

        names = {}
        for vt_symbol in vt_symbols:
            try:
                detail = xtdata.get_instrument_detail(to_xt_symbol(vt_symbol))
                names[vt_symbol] = (detail or {}).get("InstrumentName", "")
            except Exception:
                names[vt_symbol] = ""
        return names

    # ------------------------------------------------------------ 下载

    def download_history(self, vt_symbols: list[str], start: str, end: str,
                         interval: Interval = Interval.DAILY,
                         skip_existing: bool = False,
                         progress: Callable[[int, int, str], None] | None = None) -> dict:
        """下载历史数据并落地 Parquet。

        :param skip_existing: 已有本地文件则跳过，用于断点续传
        :param progress: 进度回调 (已完成数, 总数, 当前标的)
        :return: {"ok": [...], "failed": [...], "skipped": [...]}
        """
        # 周线/月线由本地日线重采样得到，不需要 xtquant，离线也能生成
        if interval in RESAMPLE_RULE:
            return self.resample_from_daily(vt_symbols, interval, progress=progress)

        try:
            from xtquant import xtdata
        except ImportError:
            logger.error("未安装 xtquant，无法下载。参见 docs/IMPLEMENTATION.md 第 1.2 节")
            return {"ok": [], "failed": list(vt_symbols), "skipped": []}

        period = PERIOD_MAP[interval]
        s, e = start.replace("-", ""), end.replace("-", "")
        result = {"ok": [], "failed": [], "skipped": []}
        total = len(vt_symbols)

        for i, raw in enumerate(vt_symbols, 1):
            vt_symbol = normalize(raw)
            if progress:
                progress(i, total, vt_symbol)

            if skip_existing and self.has_data(vt_symbol, interval):
                result["skipped"].append(vt_symbol)
                continue

            xt_symbol = to_xt_symbol(vt_symbol)
            try:
                # 先把数据补进 QMT 本地缓存，再读出来落 Parquet
                xtdata.download_history_data(xt_symbol, period=period,
                                             start_time=s, end_time=e)
                df = self._read_market_data(xtdata, xt_symbol, period, s, e)
                if df is None or df.empty:
                    logger.warning("无数据: %s %s", vt_symbol, period)
                    result["failed"].append(vt_symbol)
                    continue

                df.to_parquet(self._path(vt_symbol, interval))
                result["ok"].append(vt_symbol)
                logger.debug("已下载 %s %s，%d 条", vt_symbol, period, len(df))
            except Exception:
                logger.exception("下载失败: %s %s", vt_symbol, period)
                result["failed"].append(vt_symbol)

        logger.info("下载完成 %s：成功 %d，跳过 %d，失败 %d", period,
                    len(result["ok"]), len(result["skipped"]), len(result["failed"]))
        return result

    def _read_market_data(self, xtdata, xt_symbol: str, period: str,
                          s: str, e: str) -> pd.DataFrame | None:
        """从 xtdata 读取单标的数据"""
        data = xtdata.get_market_data_ex(
            field_list=[], stock_list=[xt_symbol], period=period,
            start_time=s, end_time=e,
            dividend_type=self.dividend_type, fill_data=False,
        )
        return data.get(xt_symbol)

    # ------------------------------------------------------------ 周线/月线合成

    def resample_from_daily(self, vt_symbols: list[str], interval: Interval,
                            progress: Callable[[int, int, str], None] | None = None) -> dict:
        """由本地日线重采样出周线/月线。

        为什么不直接从 xtdata 下周线：
        1. 少一次网络往返，300 只股票能省不少时间
        2. 复权口径与日线严格一致，不会出现日线前复权、周线后复权的错配
        3. 结果与交易所口径一致（周线 = 该周首日开盘、末日收盘、期间最高最低、成交量求和）
        """
        rule = RESAMPLE_RULE[interval]
        result = {"ok": [], "failed": [], "skipped": []}
        total = len(vt_symbols)

        for i, raw in enumerate(vt_symbols, 1):
            vt_symbol = normalize(raw)
            if progress:
                progress(i, total, vt_symbol)

            daily_path = self._path(vt_symbol, Interval.DAILY)
            if not daily_path.exists():
                logger.warning("缺少日线，无法合成 %s: %s", interval.value, vt_symbol)
                result["failed"].append(vt_symbol)
                continue

            try:
                df = pd.read_parquet(daily_path)
                df = self._normalize_df(df)
                if df.empty:
                    result["failed"].append(vt_symbol)
                    continue

                agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
                if "volume" in df.columns:
                    agg["volume"] = "sum"
                if "amount" in df.columns:
                    agg["amount"] = "sum"

                # 停牌日成交量为 0，参与 first/last 会污染开收盘价，先剔除
                if "volume" in df.columns:
                    df = df[df["volume"] > 0]

                out = df.resample(rule).agg(agg).dropna(subset=["close"])
                out.to_parquet(self._path(vt_symbol, interval))
                result["ok"].append(vt_symbol)
            except Exception:
                logger.exception("合成 %s 失败: %s", interval.value, vt_symbol)
                result["failed"].append(vt_symbol)

        logger.info("合成 %s 完成：成功 %d，失败 %d", interval.value,
                    len(result["ok"]), len(result["failed"]))
        return result

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """把 xtdata 的 DataFrame 统一成 DatetimeIndex + 小写列名"""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if "time" in df.columns and df["time"].notna().any():
            idx = pd.to_datetime(df["time"], unit="ms")
        else:
            idx = pd.to_datetime(df.index.astype(str), errors="coerce")
        df.index = idx
        return df[~df.index.isna()].sort_index()

    # ------------------------------------------------------------ 读取

    def load_bars(self, vt_symbols: list[str], start: str, end: str,
                  interval: Interval = Interval.DAILY) -> list[BarData]:
        bars: list[BarData] = []
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)

        for raw in vt_symbols:
            vt_symbol = normalize(raw)
            path = self._path(vt_symbol, interval)
            if not path.exists():
                logger.warning("本地无数据文件，请先运行 download_data.py: %s %s",
                               vt_symbol, interval.value)
                continue

            df = self._normalize_df(pd.read_parquet(path))
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
                    # xtdata 停牌日成交量为 0，据此标记
                    suspended=(volume == 0),
                    gateway_name="XT",
                ))

        bars.sort(key=lambda b: (b.datetime, b.vt_symbol))
        return bars

    def get_trading_dates(self, start: str, end: str) -> list[datetime]:
        try:
            from xtquant import xtdata
        except ImportError:
            raise NotImplementedError("需要 xtquant 才能获取交易日历")
        dates = xtdata.get_trading_dates("SH", start.replace("-", ""), end.replace("-", ""))
        return [pd.Timestamp(d, unit="ms").to_pydatetime() for d in dates]

    # ------------------------------------------------------------ 本地库存

    def summary(self) -> pd.DataFrame:
        """统计本地已下载的数据，用于确认下载结果"""
        rows = []
        for interval_dir in sorted(self.store_dir.iterdir()):
            if not interval_dir.is_dir():
                continue
            files = list(interval_dir.rglob("*.parquet"))
            if not files:
                continue
            size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
            rows.append({
                "周期": interval_dir.name,
                "标的数": len(files),
                "占用(MB)": round(size_mb, 1),
            })
        return pd.DataFrame(rows)
