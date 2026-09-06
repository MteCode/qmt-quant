"""日内分钟级特征 —— 全市场通用，不绑定特定标的。

从 1m K 线计算适用于日内交易策略的因子。设计原则：

1. **纯价量特征**：只用 OHLCV+amount，不依赖外部数据（财报、板块等），
   保证盘中实时可算。
2. **日内归一化**：绝大多数特征以当日开盘价或当日均值为基准做归一化，
   消除股价绝对值差异，使全市场可横截面比较。
3. **无前视**：所有窗口只看过去，shift 方向是 backward。
4. **缺失值处理**：开盘首几根 bar 窗口不足时 fillna(0)，不插值。

特征分组
--------
- 动量类（ret_*）：多周期收益率
- 均线偏离（ma_gap_*）：价格相对移动均线的偏离
- 波动类（vol_*, atr_*）：成交量 z-score、真实波幅
- 微观结构（vwap_gap, spread, order_imbalance）：日内价格效率
- 时间编码（tod, session）：交易时段位置
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 日内分钟级特征名，按计算顺序排列
INTRADAY_FEATURES = [
    # 动量
    "ret1", "ret3", "ret5", "ret10", "ret20",
    # 均线偏离
    "ma_gap_5", "ma_gap_10", "ma_gap_20", "ma_gap_60",
    # 波动
    "vol_z_10", "vol_z_30",
    "atr_10", "atr_20",
    # RSI
    "rsi_14",
    # 微观结构
    "vwap_gap",
    "spread",
    "order_imbalance",
    "amihud",
    # 日内位置
    "day_ret",
    "day_high_pct",
    "day_low_pct",
    # 时间编码
    "tod",
    "session",
]


def make_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """从 1m K 线计算日内特征。

    :param df: 至少包含 open/high/low/close/volume/amount 列，
               DatetimeIndex（分钟级）。
    :return: 与 df 同索引的特征 DataFrame，列名见 INTRADAY_FEATURES。
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    opn = df["open"].astype(float)
    vol = df["volume"].astype(float)
    amt = df["amount"].astype(float)

    out = pd.DataFrame(index=df.index)

    # ---- 动量：多周期收益率 ----
    for n in (1, 3, 5, 10, 20):
        out[f"ret{n}"] = close.pct_change(n)

    # ---- 均线偏离 ----
    for n in (5, 10, 20, 60):
        ma = close.rolling(n, min_periods=max(1, n // 3)).mean()
        out[f"ma_gap_{n}"] = close / ma - 1

    # ---- 成交量 z-score ----
    for n in (10, 30):
        mu = vol.rolling(n, min_periods=max(1, n // 3)).mean()
        sigma = vol.rolling(n, min_periods=max(1, n // 3)).std().replace(0, np.nan)
        out[f"vol_z_{n}"] = (vol - mu) / sigma

    # ---- ATR（Average True Range）归一化 ----
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    for n in (10, 20):
        atr = tr.rolling(n, min_periods=max(1, n // 3)).mean()
        out[f"atr_{n}"] = atr / close

    # ---- RSI ----
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = (up.rolling(14, min_periods=5).mean()
          / down.rolling(14, min_periods=5).mean().replace(0, np.nan))
    out["rsi_14"] = (100 - 100 / (1 + rs)) / 100

    # ---- VWAP 偏离（日内累计）----
    # xtdata 的 volume 单位在分钟线上不统一，直接用 amount/volume 的
    # 日内累计比值作为 VWAP 代理；绝对值不准但偏离方向正确
    day = df.index.normalize()
    cum_amt = amt.groupby(day).cumsum()
    cum_vol = vol.replace(0, np.nan).groupby(day).cumsum()
    vwap = (cum_amt / cum_vol).replace([np.inf, -np.inf], np.nan)
    out["vwap_gap"] = (close / vwap - 1)

    # ---- Spread：(high - low) / close ----
    out["spread"] = (high - low) / close

    # ---- 订单不平衡代理：用 close 与 open 的方向 × volume ----
    direction = np.sign(close - opn)
    signed_vol = direction * vol
    out["order_imbalance"] = (
        signed_vol.rolling(10, min_periods=3).sum()
        / vol.rolling(10, min_periods=3).sum().replace(0, np.nan)
    )

    # ---- Amihud 非流动性：|ret| / amount（日内滚动）----
    abs_ret = close.pct_change().abs()
    out["amihud"] = (
        abs_ret.rolling(20, min_periods=5).mean()
        / amt.rolling(20, min_periods=5).mean().replace(0, np.nan)
    ) * 1e8

    # ---- 日内位置：当日涨幅、距日内高/低点的位置 ----
    day_open = opn.groupby(day).transform("first")
    day_hi = high.groupby(day).cummax()
    day_lo = low.groupby(day).cummin()
    out["day_ret"] = close / day_open - 1
    rng = (day_hi - day_lo).replace(0, np.nan)
    out["day_high_pct"] = (day_hi - close) / rng
    out["day_low_pct"] = (close - day_lo) / rng

    # ---- 时间编码 ----
    minutes = pd.Series(df.index.hour * 60 + df.index.minute, index=df.index)
    # 上午 09:30=570, 下午 15:00=900；归一化到 [0, 1]
    out["tod"] = (minutes - 570).clip(lower=0) / 240
    # session: 0=上午, 1=下午
    out["session"] = (minutes >= 780).astype(float)

    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def make_labels(df: pd.DataFrame, horizon: int = 10,
                threshold: float = 0.0005) -> pd.Series:
    """生成二分类标签：未来 horizon 根 bar 的收益率是否超过阈值。

    :param horizon: 预测窗口（bar 数）
    :param threshold: 正类阈值，过滤噪声
    :return: Series，1=正类（未来上涨），0=负类
    """
    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    return (future_ret > threshold).astype(int)


def compute_features_for_symbol(path: str | pd.DataFrame,
                                vt_symbol: str = "") -> pd.DataFrame | None:
    """为单标的计算特征，返回特征 + 原始价格合并的 DataFrame。

    :param path: parquet 文件路径或已加载的 DataFrame
    :return: 包含 INTRADAY_FEATURES 列 + close/volume/amount 列的 DataFrame，
             或 None（数据不足时）
    """
    if isinstance(path, (str, type(None))):
        import pathlib
        p = pathlib.Path(path)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
    else:
        df = path

    if len(df) < 60:
        return None

    feat = make_intraday_features(df)
    feat["close"] = df["close"].astype(float)
    feat["volume"] = df["volume"].astype(float)
    feat["amount"] = df["amount"].astype(float)
    if vt_symbol:
        feat["symbol"] = vt_symbol
    return feat
