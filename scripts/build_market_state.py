"""构建分钟级市场状态序列 —— 给单票策略提供「大盘现在什么样」。

## 为什么不用指数

本地没有指数的 1m 数据，但有 4750 只个股的 1m 数据 —— 直接算横截面
统计量比用指数更细：指数是市值加权的，几只权重股就能主导它，
而「市场宽度」（多少只在涨）反映的是普涨还是分化，这对日内交易
更有意义。

## 三个指标

- **mkt_ret**：全市场等权分钟收益。大盘整体在涨还是在跌。
- **mkt_breadth**：上涨家数占比。0.8 是普涨，0.5 是分化，0.2 是普跌。
  同样是 +0.3% 的平均涨幅，普涨和「少数暴涨拉起来的」含义完全不同。
- **mkt_vol_z**：全市场成交量相对当日均值的 z-score。放量还是缩量。

## 无前视

每个时点的统计量只用该时点及之前的数据：
- mkt_ret / mkt_breadth 用当根 bar 的 close/open，都在该分钟收盘时可见
- mkt_vol_z 用当日累计到该时刻的均值与标准差，不用全天数据

## 用法

    python scripts/build_market_state.py --sample 500
    python scripts/build_market_state.py --sample 0     # 全市场，慢
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "market_state_1m.parquet"


def main() -> int:
    p = argparse.ArgumentParser(description="构建分钟级市场状态")
    p.add_argument("--sample", type=int, default=500,
                   help="采样多少只标的（0=全市场）。500 只已能稳定估计"
                        "横截面统计量，全市场只是更慢")
    p.add_argument("--min-bars", type=int, default=5000,
                   help="标的至少要有多少根 bar 才纳入")
    p.add_argument("--output", default=str(OUT))
    args = p.parse_args()

    from qmtquant.config import get_config
    cfg = get_config()
    src = Path(cfg.data.store_dir) / "clean" / "1m"
    if not src.exists():
        print(f"找不到 1m 数据: {src}")
        return 1

    # 与训练一致的过滤：排除北交所、科创板
    files = []
    for ex in sorted(d for d in src.iterdir()
                     if d.is_dir() and d.name != "BSE"):
        for f in sorted(ex.glob("*.parquet")):
            if f.stem.startswith("688"):
                continue
            files.append(f)

    if args.sample and len(files) > args.sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(files), args.sample, replace=False)
        files = [files[i] for i in sorted(idx)]

    print(f"读取 {len(files)} 只标的...")
    rets, ups, vols = [], [], []
    t0 = time.time()
    used = 0

    for i, f in enumerate(files, 1):
        try:
            df = pd.read_parquet(f, columns=["open", "close", "volume"])
        except (OSError, ValueError, KeyError):
            continue
        if len(df) < args.min_bars:
            continue

        o = df["open"].astype(np.float32)
        c = df["close"].astype(np.float32)
        # 单根 bar 的收益：收盘/开盘 - 1。只用该 bar 自身，无前视
        r = (c / o.replace(0, np.nan) - 1).astype(np.float32)
        rets.append(r.rename(f.stem))
        ups.append((r > 0).astype(np.float32).rename(f.stem))
        vols.append(df["volume"].astype(np.float32).rename(f.stem))
        used += 1

        if i % 100 == 0:
            print(f"  {i}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

    if not rets:
        print("没有可用数据")
        return 1

    print(f"合并 {used} 只标的的横截面...")
    R = pd.concat(rets, axis=1)
    U = pd.concat(ups, axis=1)
    V = pd.concat(vols, axis=1)
    del rets, ups, vols

    out = pd.DataFrame(index=R.index)
    # 等权平均收益。用 median 而非 mean 也可以，但 mean 对极端行情
    # 更敏感，日内正是要捕捉这种时刻
    out["mkt_ret"] = R.mean(axis=1)
    out["mkt_breadth"] = U.mean(axis=1)
    out["n_symbols"] = R.notna().sum(axis=1).astype(np.int32)

    # 成交量 z-score：相对**当日累计到此刻**的均值与标准差，
    # 不能用全天数据 —— 那是前视
    tot_vol = V.sum(axis=1)
    day = out.index.normalize()
    g = tot_vol.groupby(day)
    mu = g.expanding().mean().reset_index(level=0, drop=True)
    sd = g.expanding().std().reset_index(level=0, drop=True)
    out["mkt_vol_z"] = ((tot_vol - mu) / sd.replace(0, np.nan)).fillna(0)

    # 当日累计市场收益：从开盘到此刻大盘涨了多少
    out["mkt_day_ret"] = out.groupby(day)["mkt_ret"].cumsum()

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["mkt_ret"])
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)

    p_out = Path(args.output)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p_out)

    print(f"\n{'=' * 62}")
    print(f"  市场状态已生成: {p_out}")
    print(f"  {len(out):,} 个时点，{out.index.min()} ~ {out.index.max()}")
    print(f"  平均纳入标的数: {out['n_symbols'].mean():.0f}")
    print()
    print("  指标分布:")
    for c in ("mkt_ret", "mkt_breadth", "mkt_vol_z", "mkt_day_ret"):
        s = out[c]
        print(f"    {c:<14} 中位 {s.median():>8.4f}  "
              f"10% {s.quantile(.1):>8.4f}  90% {s.quantile(.9):>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
