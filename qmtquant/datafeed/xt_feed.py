"""基于 xtdata 的行情数据源（miniQMT / 大 QMT）。

xtdata 只有在 QMT 客户端运行时才能取数，因此本类做了两件事：
1. 下载 → 落地为 Parquet，脱离客户端也能回测
2. 优先读本地 Parquet，缺失才回源下载
"""
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..core.constants import Interval
from ..core.objects import BarData
from ..utils.symbol import normalize, split_vt_symbol, to_xt_symbol
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
}


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

    # ------------------------------------------------------------ 下载

    def download_history(self, vt_symbols: list[str], start: str, end: str,
                         interval: Interval = Interval.DAILY) -> None:
        try:
            from xtquant import xtdata
        except ImportError:
            logger.error("未安装 xtquant，无法下载数据。可改用 CsvDataFeed 离线回测")
            return

        period = PERIOD_MAP[interval]
        s, e = start.replace("-", ""), end.replace("-", "")

        for vt_symbol in vt_symbols:
            vt_symbol = normalize(vt_symbol)
            xt_symbol = to_xt_symbol(vt_symbol)
            try:
                # 先补齐本地缓存，再读出来落 Parquet
                xtdata.download_history_data(xt_symbol, period=period,
                                             start_time=s, end_time=e)
                data = xtdata.get_market_data_ex(
                    field_list=[], stock_list=[xt_symbol], period=period,
                    start_time=s, end_time=e,
                    dividend_type=self.dividend_type, fill_data=False,
                )
                df = data.get(xt_symbol)
                if df is None or df.empty:
                    logger.warning("无数据: %s", vt_symbol)
                    continue

                df.to_parquet(self._path(vt_symbol, interval))
                logger.info("已下载 %s %s，%d 条", vt_symbol, period, len(df))
            except Exception:
                logger.exception("下载失败: %s", vt_symbol)

    # ------------------------------------------------------------ 读取

    def load_bars(self, vt_symbols: list[str], start: str, end: str,
                  interval: Interval = Interval.DAILY) -> list[BarData]:
        bars: list[BarData] = []
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)

        for vt_symbol in vt_symbols:
            vt_symbol = normalize(vt_symbol)
            path = self._path(vt_symbol, interval)
            if not path.exists():
                logger.warning("本地无数据文件，请先 download_history: %s", vt_symbol)
                continue

            df = pd.read_parquet(path)
            symbol, exchange = split_vt_symbol(vt_symbol)

            for idx, row in df.iterrows():
                dt = self._parse_dt(idx, row)
                if dt is None or not (start_dt <= dt <= end_dt):
                    continue
                volume = float(row.get("volume", 0))
                bars.append(BarData(
                    symbol=symbol, exchange=exchange, datetime=dt.to_pydatetime(),
                    interval=interval,
                    open_price=float(row["open"]), high_price=float(row["high"]),
                    low_price=float(row["low"]), close_price=float(row["close"]),
                    volume=volume, turnover=float(row.get("amount", 0)),
                    # xtdata 停牌日成交量为 0，据此标记
                    suspended=(volume == 0),
                    gateway_name="XT",
                ))

        bars.sort(key=lambda b: (b.datetime, b.vt_symbol))
        return bars

    @staticmethod
    def _parse_dt(idx, row) -> pd.Timestamp | None:
        """xtdata 的时间可能在 index（'20240102'）或 time 列（毫秒戳）"""
        if "time" in row and pd.notna(row["time"]):
            return pd.Timestamp(int(row["time"]), unit="ms")
        try:
            return pd.Timestamp(str(idx))
        except Exception:
            return None

    def get_trading_dates(self, start: str, end: str) -> list[datetime]:
        try:
            from xtquant import xtdata
        except ImportError:
            raise NotImplementedError("需要 xtquant 才能获取交易日历")
        dates = xtdata.get_trading_dates("SH", start.replace("-", ""), end.replace("-", ""))
        return [pd.Timestamp(d, unit="ms").to_pydatetime() for d in dates]
