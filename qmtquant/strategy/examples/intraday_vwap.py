"""兼容别名，实现已移至 ``qmtquant.strategy.intraday_vwap``。"""
from ..intraday_vwap import (
    MODE_REVERSION,
    MODE_TREND,
    ROLE_ENTRY,
    ROLE_T0,
    IntradayVwapStrategy,
)

__all__ = ["IntradayVwapStrategy", "MODE_TREND", "MODE_REVERSION",
           "ROLE_ENTRY", "ROLE_T0"]
