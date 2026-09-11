"""回撤减仓的数量计算 —— 回测与实盘共用的唯一实现。

## 为什么要有这个模块

同一件事「把持仓削减到 keep_ratio」在回测（`backtest_engine._enforce_drawdown`）
和实盘（`risk_monitor.sell_to_target`）里各写了一遍，而且**已经跑偏**：
回测会发出 451 股这种零股部分委托（实盘必被券商拒单），实盘则 floor 到 400。
两份实现各自演进，迟早有一边先错，而错误无声。抽成一份，改一处两边同改。

## A 股的卖出整手规则

部分卖出须为 100 股的整数倍；**清空整个可卖持仓**时，零股可以一次性卖出
（券商允许「全卖」带零股）。所以取整只作用于「部分卖出」，清仓不取整。

整手约束作用在**真实股数**上。回测持有的是后复权价口径的股数，需按
`factor = 后复权价 / 真实价` 换算回真实口径再取整；实盘本就是真实股数，
`factor` 取默认 1.0。
"""
from __future__ import annotations

#: 一手股数，A 股为 100
DEFAULT_LOT_SIZE = 100


def plan_reduction(volume: float, available: float, keep_ratio: float,
                   lot_size: int = DEFAULT_LOT_SIZE,
                   factor: float = 1.0) -> float:
    """把持仓削减到 keep_ratio，返回应卖出的股数（0 表示不动）。

    返回量的口径与 ``volume`` 相同（回测为后复权股数，实盘为真实股数）。

    :param volume: 当前持仓总量
    :param available: 可卖数量（T+1，当日买入部分不可卖）
    :param keep_ratio: 目标保留比例；<= 0 即清仓
    :param lot_size: 一手股数
    :param factor: 后复权价 / 真实价，用于把整手约束换算到真实股数上
    """
    volume = float(volume)
    if volume <= 0 or keep_ratio >= 1.0:
        return 0.0

    sellable = min(float(available), volume)
    if sellable <= 0:
        return 0.0

    # 清仓：零股可一次性卖出，不取整
    if keep_ratio <= 0:
        return sellable

    excess = volume - volume * keep_ratio
    if excess <= 0:
        return 0.0
    sell = min(excess, sellable)
    if sell <= 0:
        return 0.0

    f = factor if factor and factor > 0 else 1.0
    real = int(sell * f // lot_size) * lot_size
    sell = real / f
    return sell if sell > 0 else 0.0


def should_act(level, acted_level) -> bool:
    """上升沿判定：档位**高于**已执行档位时才动作。

    回测与实盘共用，取代两边各自的 ``_last_enforced_level`` / ``acted_level``
    写法。下降沿由调用方自行把 acted_level 同步回当前档位。
    """
    return level > acted_level
