"""旧的示例策略路径，保留为兼容别名。

这四个策略已于 2026-08 升级为正式模块并移出本目录：

===============================  ============================
旧路径                            新路径
===============================  ============================
``examples.momentum_rotation``   ``qmtquant.strategy.momentum``
``examples.trend_ma``            ``qmtquant.strategy.trend_ma``
``examples.ma_cross``            ``qmtquant.strategy.ma_cross``
``examples.intraday_vwap``       ``qmtquant.strategy.intraday_vwap``
===============================  ============================

「升级」的具体含义：参数校验、类型强制转型（参数寻优会传 float）、
买卖限价缓冲不对称、成交计数可持久化，以及各自的独立测试文件。

配置文件里若还写着 ``qmtquant.strategy.examples.*`` 仍可工作，
但建议改成新路径 —— 本兼容层不保证长期保留。
"""
from ..intraday_vwap import IntradayVwapStrategy
from ..ma_cross import MaCrossStrategy
from ..momentum import MomentumRotationStrategy
from ..trend_ma import TrendMaStrategy

__all__ = ["IntradayVwapStrategy", "MaCrossStrategy",
           "MomentumRotationStrategy", "TrendMaStrategy"]
