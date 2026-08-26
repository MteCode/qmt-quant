"""CSV / 合成数据源。

用途：没有 QMT 环境（比如 Linux CI）时也能跑通回测与单元测试。
CSV 需包含列：datetime, open, high, low, close, volume[, amount]
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.constants import Interval
from ..core.objects import BarData
from ..utils.symbol import normalize, split_vt_symbol
from .base import BaseDataFeed

logger = logging.getLogger(__name__)


class CsvDataFeed(BaseDataFeed):
    """从 CSV 目录读取行情，文件名为 `{symbol}.csv`"""

    def __init__(self, csv_dir: str) -> None:
        self.csv_dir = Path(csv_dir)

    def download_history(self, vt_symbols, start, end, interval=Interval.DAILY) -> None:
        """CSV 数据源为离线源，无需下载"""

    def load_bars(self, vt_symbols: list[str], start: str, end: str,
                  interval: Interval = Interval.DAILY) -> list[BarData]:
        bars: list[BarData] = []
        start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)

        for vt_symbol in vt_symbols:
            vt_symbol = normalize(vt_symbol)
            symbol, exchange = split_vt_symbol(vt_symbol)
            path = self.csv_dir / f"{symbol}.csv"
            if not path.exists():
                logger.warning("CSV 不存在: %s", path)
                continue

            df = pd.read_csv(path, parse_dates=["datetime"])
            df = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
            for _, row in df.iterrows():
                bars.append(BarData(
                    symbol=symbol, exchange=exchange,
                    datetime=row["datetime"].to_pydatetime(), interval=interval,
                    open_price=float(row["open"]), high_price=float(row["high"]),
                    low_price=float(row["low"]), close_price=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                    turnover=float(row.get("amount", 0)),
                    suspended=bool(row.get("volume", 1) == 0),
                    gateway_name="CSV",
                ))

        bars.sort(key=lambda b: (b.datetime, b.vt_symbol))
        return bars


def generate_random_bars(vt_symbol: str, days: int = 500, start_price: float = 10.0,
                         seed: int = 42, annual_vol: float = 0.30) -> list[BarData]:
    """生成几何布朗运动模拟行情，用于框架自测（无真实数据时）。

    注意：这只是随机游走，**任何在它上面跑出的绩效都没有参考意义**。
    """
    rng = np.random.default_rng(seed)
    vt_symbol = normalize(vt_symbol)
    symbol, exchange = split_vt_symbol(vt_symbol)

    daily_vol = annual_vol / np.sqrt(242)
    returns = rng.normal(0.0003, daily_vol, days)
    closes = start_price * np.exp(np.cumsum(returns))
    dates = pd.bdate_range("2022-01-04", periods=days)

    bars: list[BarData] = []
    prev_close = start_price
    for dt, close in zip(dates, closes):
        open_p = prev_close * (1 + rng.normal(0, daily_vol / 3))
        high = max(open_p, close) * (1 + abs(rng.normal(0, daily_vol / 3)))
        low = min(open_p, close) * (1 - abs(rng.normal(0, daily_vol / 3)))
        bars.append(BarData(
            symbol=symbol, exchange=exchange, datetime=dt.to_pydatetime(),
            interval=Interval.DAILY,
            open_price=round(open_p, 2), high_price=round(high, 2),
            low_price=round(low, 2), close_price=round(close, 2),
            volume=float(rng.integers(1_000_000, 10_000_000)),
            gateway_name="MOCK",
        ))
        prev_close = close
    return bars
