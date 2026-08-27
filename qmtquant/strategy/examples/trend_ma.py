"""兼容别名，实现已移至 ``qmtquant.strategy.trend_ma``。"""
from ..trend_ma import MODE_REVERSION, MODE_TREND, TrendMaStrategy

__all__ = ["TrendMaStrategy", "MODE_TREND", "MODE_REVERSION"]
