"""数据源抽象。"""
from abc import ABC, abstractmethod
from datetime import datetime

from ..core.constants import Interval
from ..core.objects import BarData


class BaseDataFeed(ABC):
    """历史与实时行情数据源"""

    @abstractmethod
    def download_history(self, vt_symbols: list[str], start: str, end: str,
                         interval: Interval = Interval.DAILY) -> None:
        """把历史数据下载到本地，供后续 load_bars 读取"""

    @abstractmethod
    def load_bars(self, vt_symbols: list[str], start: str, end: str,
                  interval: Interval = Interval.DAILY) -> list[BarData]:
        """读取历史 K 线，按时间升序返回"""

    def get_trading_dates(self, start: str, end: str) -> list[datetime]:
        """交易日历。默认不支持，由具体实现覆写"""
        raise NotImplementedError
