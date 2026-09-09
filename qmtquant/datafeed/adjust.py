"""后复权价与真实价的换算。

## 为什么需要

本地行情一律是**后复权价**，而且锚在 IPO，所以复权因子可以非常大：

    万科A     真实   3.13 元   后复权  879.1 元   因子 280.9
    平安银行  真实  11.73 元   后复权 1228.7 元   因子 104.7
    海尔智家  真实  21.15 元   后复权  906.8 元   因子  42.9
    工商银行  真实   8.15 元   后复权   13.2 元   因子   1.6

凡是「拿价格去比一个现实世界的阈值」的地方，用后复权价都会错，
而且错得没有规律 —— 因子从 1.6 到 280 不等。已经踩到的两处：

1. **高股价过滤**。规则是「1 手买入超过 5 万的排除」，即真实价 > 500。
   写成 `close > 500` 作用在后复权价上，于是把海尔智家（21 元）、
   新和成（26 元）、北方稀土（40 元）这些**便宜的老蓝筹**排除了 ——
   它们只是分红送转多、因子高。实测 500 只里误排 11 只，
   而真正该排的只有 1 只：错排的比对排的多 10 倍。

2. **整手取整**。一手是 100 **真实**股。按后复权价算手数，
   高因子的标的会被算成不足一手而静默跳过。

## 真实价怎么来

turnover / (volume * 每手股数)。turnover 与 volume 都是不复权的真实值，
相除即当日均价。这里不依赖 `data/1d_raw/` —— 那个目录在本仓库并不存在，
依赖它的代码路径恒退回旧行为，是静默失效的。
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["real_price", "adj_factor", "LOT_SIZE"]

#: A 股一手股数
LOT_SIZE = 100

#: 复权因子的合理区间。低于 1 说明后复权价低于真实价（不可能，
#: 除非单位搞错）；高于这个上限说明数据有问题，不猜。
_MIN_FACTOR = 0.8
_MAX_FACTOR = 500.0


def real_price(df: pd.DataFrame, lot_size: int = LOT_SIZE) -> float | None:
    """从最后一根 bar 反推真实价（当日均价口径）。

    需要 turnover/amount 与 volume 两列。缺任一列返回 None ——
    返回 None 比返回一个猜的数好：调用方能据此选择保守行为，
    而一个错的价格会静默改变过滤与仓位。
    """
    if df is None or len(df) == 0:
        return None
    cols = set(df.columns)
    amt_col = "amount" if "amount" in cols else (
        "turnover" if "turnover" in cols else None)
    if amt_col is None or "volume" not in cols:
        return None

    row = df.iloc[-1]
    try:
        vol = float(row["volume"])
        amt = float(row[amt_col])
    except (TypeError, ValueError):
        return None
    if vol <= 0 or amt <= 0:
        return None

    price = amt / (vol * lot_size)
    return price if price > 0 else None


def adj_factor(df: pd.DataFrame, lot_size: int = LOT_SIZE) -> float | None:
    """后复权价 / 真实价。

    取不到或算出不合理的值返回 None，由调用方决定退化行为。
    """
    if df is None or len(df) == 0 or "close" not in df.columns:
        return None
    real = real_price(df, lot_size)
    if real is None:
        return None
    try:
        adj = float(df["close"].iloc[-1])
    except (TypeError, ValueError, IndexError):
        return None
    if adj <= 0:
        return None

    f = adj / real
    # 略小于 1 是收盘价与当日均价的日内噪声（真实价用的是均价），夹到 1
    if _MIN_FACTOR <= f < 1.0:
        return 1.0
    return f if 1.0 <= f <= _MAX_FACTOR else None
