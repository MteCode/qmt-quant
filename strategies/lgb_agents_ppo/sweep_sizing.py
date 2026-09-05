"""仓位与持仓只数的可行性扫描。

## 要回答的问题

之前的回测用 100 万本金、50 只持仓、满仓运行 —— 每只 2 万，能买满一手。
但实盘是 50 万本金，PPO 又把仓位压到 30%，于是每只只剩 5000 元，
科创板/创业板高价股连一手都买不进。

**回测里的持仓在实盘根本建不起来。** 这比 IC 高低重要得多。

本脚本按真实本金扫描 (仓位, 持仓只数)，同时给出两件事：

1. **可行性** —— 该配置下 Top-K 里有几只能买满一手
2. **回测表现** —— 只用能买进的那些标的，成绩还剩多少

第二点是关键：把买不进的标的从回测里剔掉，得到的才是**可执行的成绩**。

## 价格口径

优先用 `data/1d_raw/` 的不复权价（真实价）。没有则退回后复权价 ——
后者被抬高（茅台约 5.4 倍），会**高估**买不进的比例，结论偏保守。

用法::

    python strategies/lgb_agents_ppo/sweep_sizing.py
    python strategies/lgb_agents_ppo/sweep_sizing.py --capital 500000
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import paths  # noqa: E402

RESULT_JSON = paths.BACKTEST_DIR / "sizing.json"

EXPOSURES = [0.3, 0.5, 0.8, 1.0]
HOLDINGS = [5, 10, 15, 20, 30]
LOT = 100


def load_prices(store: Path, use_raw: bool = True):
    """真实价面板。优先不复权，退回后复权。"""
    import pandas as pd

    if use_raw:
        raw_root = store / "1d_raw"
        if raw_root.exists() and any(raw_root.rglob("*.parquet")):
            cols = {}
            for p in raw_root.rglob("*.parquet"):
                vt = f"{p.stem}.{p.parent.name}"
                try:
                    d = pd.read_parquet(p, columns=["close"])
                except (OSError, ValueError, KeyError):
                    continue
                if not isinstance(d.index, pd.DatetimeIndex):
                    d.index = pd.to_datetime(d.index.astype(str),
                                             format="%Y%m%d", errors="coerce")
                cols[vt] = d["close"]
            if cols:
                return pd.DataFrame(cols).sort_index(), "不复权（真实价）"

    # 退回后复权：价格被抬高，会高估买不进的比例，结论偏保守
    import sqlite3
    db = store / "market.db"
    if not db.exists():
        return None, None
    conn = sqlite3.connect(str(db))
    df = pd.read_sql(
        "SELECT date, vt_symbol, close FROM daily_bar", conn)
    conn.close()
    panel = df.pivot(index="date", columns="vt_symbol", values="close")
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index(), "后复权（偏保守）"


def feasibility(scores, prices, capital: float, exposure: float,
                k: int, rebal_dates) -> dict:
    """各调仓日有几只能买满一手。"""
    import numpy as np

    per = capital * exposure / k
    ok, total = [], []
    for d in rebal_dates:
        if d not in scores.index:
            continue
        top = scores.loc[d].dropna().nlargest(k)
        if d not in prices.index:
            continue
        px = prices.loc[d]
        n = 0
        for vt in top.index:
            p = px.get(vt)
            if p and p > 0 and per >= p * LOT:
                n += 1
        ok.append(n)
        total.append(len(top))
    if not ok:
        return {"per_name": per, "feasible": 0.0, "feasible_pct": 0.0}
    return {"per_name": per,
            "feasible": float(np.mean(ok)),
            "feasible_pct": float(np.mean(ok) / max(np.mean(total), 1))}


def main() -> int:
    p = argparse.ArgumentParser(description="仓位与持仓只数扫描")
    p.add_argument("--capital", type=float, default=None)
    p.add_argument("--rebalance", type=int, default=20)
    p.add_argument("--index", default="000852.SH")
    args = p.parse_args()

    import pandas as pd

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.core.constants import Interval
    from qmtquant.datafeed.qlib_init import init_qlib
    from qmtquant.datafeed.xt_feed import XtDataFeed
    from qmtquant.universe.providers import HistoricalUniverse
    from qmtquant.utils.logger import setup_logging

    sys.path.insert(0, str(paths.ROOT / "strategies" / "alstm_ppo_csi1000"))
    from train_ensemble import backtest as run_bt

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    params = paths.load_params()
    capital = args.capital or params["capital"]
    store = Path(cfg.data.store_dir)
    init_qlib(str(store / "qlib_data"), n_expressions=8)

    print("=" * 74)
    print("仓位与持仓只数扫描")
    print(f"  本金 {capital:,.0f}   每 {args.rebalance} 日调仓   一手 {LOT} 股")
    print("=" * 74)

    scores = pd.read_parquet(paths.SCREEN_SCORES).sort_index()
    print(f"\n分数面板 {scores.shape[0]} 期 x {scores.shape[1]} 只")

    prices, src = load_prices(store)
    if prices is None:
        print("无价格数据")
        return 1
    print(f"价格口径 {src}   {prices.shape[0]} 期 x {prices.shape[1]} 只")

    print("\n装载回测行情...")
    weight_csv = store / "universe" / f"index_weight_{args.index}.csv"
    universe = HistoricalUniverse(str(weight_csv))
    symbols = universe.all_symbols()
    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    bars = feed.load_bars(symbols, str(scores.index[0].date()),
                          str(scores.index[-1].date()), Interval.DAILY)
    print(f"  {len(bars):,} 根 K 线")

    rebal = scores.index[::args.rebalance]

    print("\n" + "-" * 74)
    print(f"{'仓位':>6s}{'持仓':>6s}{'单只金额':>10s}{'可买满':>8s}"
          f"{'可行率':>8s}{'总收益':>10s}{'最大回撤':>10s}{'Sharpe':>9s}")
    print("-" * 74)

    rows = []
    t0 = time.time()
    for exp in EXPOSURES:
        for k in HOLDINGS:
            fe = feasibility(scores, prices, capital, exp, k, rebal)
            # 回测：持仓 k 只，本金按实际投入额（仓位 x 本金）
            m = run_bt(scores, cfg, bars, universe, symbols,
                       k, args.rebalance, capital * exp)
            row = {"exposure": exp, "holdings": k, **fe,
                   "total_return": m["total_return"],
                   "max_drawdown": m["max_drawdown"],
                   "sharpe": m["sharpe"],
                   "drawdown_ok": m["drawdown_ok"],
                   "total_trades": m["total_trades"]}
            rows.append(row)
            print(f"{exp:>6.0%}{k:>6d}{fe['per_name']:>10,.0f}"
                  f"{fe['feasible']:>8.1f}{fe['feasible_pct']:>8.0%}"
                  f"{m['total_return']:>+10.2%}{m['max_drawdown']:>10.2%}"
                  f"{m['sharpe']:>+9.3f}")
        print()

    # ---- 结论
    print("-" * 74)
    good = [r for r in rows if r["feasible_pct"] >= 0.9
            and r["drawdown_ok"] and r["sharpe"] > 0]
    if good:
        best = max(good, key=lambda r: r["sharpe"])
        print(f"可执行且达标的配置（可行率>=90%、回撤达标、Sharpe>0）"
              f"共 {len(good)} 个，最优：")
        print(f"  仓位 {best['exposure']:.0%}  持仓 {best['holdings']} 只  "
              f"单只 {best['per_name']:,.0f} 元")
        print(f"  收益 {best['total_return']:+.2%}  "
              f"回撤 {best['max_drawdown']:.2%}  "
              f"Sharpe {best['sharpe']:+.3f}")
    else:
        print("!! 没有配置同时满足：可行率>=90%、回撤达标、Sharpe>0")
        feas = [r for r in rows if r["feasible_pct"] >= 0.9]
        if feas:
            b = max(feas, key=lambda r: r["sharpe"])
            print(f"   可行率达标中最好的：仓位 {b['exposure']:.0%} / "
                  f"{b['holdings']} 只 -> Sharpe {b['sharpe']:+.3f}"
                  f"  回撤 {b['max_drawdown']:.2%}")

    paths.ensure_dirs()
    RESULT_JSON.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "capital": capital, "rebalance": args.rebalance,
        "price_source": src, "lot": LOT,
        "runs": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细: {RESULT_JSON}")
    print(f"耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
