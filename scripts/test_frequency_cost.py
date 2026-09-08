"""频率与成本的双维实验 —— 做 T 年化 50% 需要什么条件。

## 两条路径

做 T 年化 = 每日次数 x 每笔投入 x 净edge x 244 / 总资产
其中 净edge = 毛edge - 往返成本

要提高它只有三个杠杆：**做得更多、做得更准、成本更低**。
「做得更准」已经被 22032 组参数搜索证否（为正 0.1%，样本外 0/20），
所以这里测剩下两个。

## 关键假设必须先验证

「每天做 50 次」的算法默认 **edge 在高频下不衰减**。这个假设很可能
是错的：把阈值放松到每天 50 次，多出来的都是边际信号，
毛 edge 会掉下来。

所以本实验的核心不是「找最优参数」，而是**测出 edge 随频率的衰减曲线**。
如果 edge 随频率单调下降且下降快于频率上升，那高频这条路直接封死。

## 成本这条路

当前往返成本 0.25%（佣金双边 0.05% + 印花税 0.10% + 滑点双边 0.10%）。
可以谈的部分：
- 佣金：万 2.5 -> 万 1，双边省 0.03%
- 滑点：万 5 -> 万 2，双边省 0.06%
- 印花税 0.10%：**法定，不可谈**

所以成本下限约 0.16%，降幅 36%。而净 edge = 毛 - 成本，
毛 edge 0.266% 时净 edge 从 0.016% 涨到 0.106% —— **6.6 倍**。
成本是这里最大的杠杆，这一点必须量化给出。
"""
import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

OUT_DIR = ROOT / "models" / "t0_frequency"


def _engine():
    spec = importlib.util.spec_from_file_location(
        "t0div", ROOT / "scripts" / "optimize_t0_divergence.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cost_scenarios() -> list[dict]:
    """成本情景。印花税是法定的，不能假设它降。"""
    return [
        {"name": "当前（佣金万2.5+滑点万5）",
         "commission": 0.00025, "slippage": 0.0005},
        {"name": "佣金谈到万1",
         "commission": 0.0001, "slippage": 0.0005},
        {"name": "滑点压到万2",
         "commission": 0.00025, "slippage": 0.0002},
        {"name": "两者都优化",
         "commission": 0.0001, "slippage": 0.0002},
        {"name": "[不可实现]零佣金零滑点",
         "commission": 0.0, "slippage": 0.0},
    ]


def annual_from(net_edge: float, trades_per_day: float,
                per_trade: float, init: float, days: int = 244) -> float:
    """由净 edge、频率、单笔金额反推年化。"""
    return net_edge * trades_per_day * per_trade * days / init


def main() -> int:
    p = argparse.ArgumentParser(description="频率与成本双维实验")
    p.add_argument("--symbol", default="600711")
    p.add_argument("--exchange", default="SSE")
    p.add_argument("--base", type=float, default=100000.0)
    p.add_argument("--cash", type=float, default=100000.0)
    p.add_argument("--target", type=float, default=0.50,
                   help="做T年化目标")
    p.add_argument("--entry-time", default="09:35")
    p.add_argument("--exit-time", default="14:50")
    p.add_argument("--output", default=str(OUT_DIR))
    args = p.parse_args()

    def _tm(s):
        h, mm = s.split(":")
        return int(h) * 60 + int(mm)

    m = _engine()
    em, xm = _tm(args.entry_time), _tm(args.exit_time)
    init = args.base + args.cash
    STAMP = 0.001

    print("=" * 80)
    print(f"  频率与成本双维实验 —— {args.symbol}.{args.exchange}")
    print(f"  底仓 {args.base:,.0f} + 现金 {args.cash:,.0f} = {init:,.0f} 元")
    print(f"  目标：做T年化 {args.target:.0%}")
    print("=" * 80)

    df = m.load_bars(args.symbol, args.exchange)
    d = m.prep(df)
    days = pd.DatetimeIndex(d["day"].unique()).sort_values()
    n_days = len(days)
    print(f"  {n_days} 交易日 {days[0].date()} ~ {days[-1].date()}")

    # ---------- 一、edge 随频率如何变化 ----------
    print(f"\n{'=' * 80}")
    print("  一、毛 edge 随交易频率的衰减曲线")
    print("  （核心问题：放松阈值多做交易，每笔的质量会掉多少）")
    print(f"{'=' * 80}")

    # 从严到松扫阈值，观察频率与 edge 的关系
    combos = []
    for sig in ("corr", "obv", "pulse"):
        if sig in ("corr", "obv"):
            thrs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0]
        else:
            thrs = [0.02, 0.015, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0]
        for thr, vt, tp, sl, mt, hd in itertools.product(
                thrs, [0.0, 1.0, 2.0], [0.004, 0.008],
                [0.008, 0.015], [1, 3, 6, 12], [10, 30]):
            combos.append((sig, thr, vt, tp, sl, mt, hd))

    print(f"  扫描 {len(combos)} 组（阈值从严到松，每日次数上限 1~12）")
    rows = []
    t0 = time.time()
    for i, (sig, thr, vt, tp, sl, mt, hd) in enumerate(combos, 1):
        r = m.simulate(d, args.base, args.cash, sig, thr, vt, tp, sl, mt,
                       em, xm, hd, False)
        if "error" in r or r["n_trades"] == 0:
            continue
        rows.append({
            "signal": sig, "thr": thr, "vol_thr": vt, "take_profit": tp,
            "stop_loss": sl, "max_trades": mt, "hold_max": hd,
            "n_trades": r["n_trades"],
            "trades_per_day": r["n_trades"] / n_days,
            "gross_edge": r["gross_edge"], "net_edge": r["net_edge"],
            "t_annual": r["t_annual"], "win_rate": r["win_rate"],
            "coverage": r["coverage"],
        })
        if i % 500 == 0:
            print(f"    {i}/{len(combos)} ({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        print("无有效结果")
        return 1

    g = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "frequency_scan.csv", index=False)

    # 按频率分桶，看 edge 的变化
    print(f"\n  {len(g)} 组有交易。按每日交易次数分桶：")
    print(f"  {'每日次数':<12} {'组数':>6} {'毛edge中位':>12} "
          f"{'毛edge上四分位':>15} {'胜率中位':>10} {'做T年化中位':>12}")
    print("  " + "-" * 72)
    bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 100)]
    freq_rows = []
    for lo, hi in bins:
        s = g[(g["trades_per_day"] >= lo) & (g["trades_per_day"] < hi)]
        if len(s) < 3:
            continue
        label = f"{lo:g}~{hi:g}" if hi < 100 else f">{lo:g}"
        freq_rows.append({
            "bucket": label, "n": len(s),
            "gross_median": float(s["gross_edge"].median()),
            "gross_q75": float(s["gross_edge"].quantile(0.75)),
            "win_median": float(s["win_rate"].median()),
            "t_annual_median": float(s["t_annual"].median()),
            "trades_per_day_median": float(s["trades_per_day"].median()),
        })
        print(f"  {label:<12} {len(s):>6} {s['gross_edge'].median():>12.4%} "
              f"{s['gross_edge'].quantile(0.75):>15.4%} "
              f"{s['win_rate'].median():>10.1%} "
              f"{s['t_annual'].median():>12.2%}")

    if len(freq_rows) >= 3:
        lo_e = freq_rows[0]["gross_median"]
        hi_e = freq_rows[-1]["gross_median"]
        print(f"\n  低频组毛 edge {lo_e:.4%} -> 高频组 {hi_e:.4%}"
              f"（{'衰减' if hi_e < lo_e else '上升'} "
              f"{abs(hi_e/lo_e - 1):.0%}）")
        if hi_e < lo_e:
            print(f"  → edge 随频率衰减。多做的交易质量更差，"
                  f"「靠频率堆收益」这条路走不通。")
        else:
            print(f"  → edge 未随频率衰减，高频路径值得进一步验证。")

    # ---------- 二、成本情景 ----------
    print(f"\n{'=' * 80}")
    print("  二、成本降低的效果")
    print(f"{'=' * 80}")

    # 取实测毛 edge 的代表值：所有组合的中位数与上四分位
    ge_med = float(g["gross_edge"].median())
    ge_q75 = float(g["gross_edge"].quantile(0.75))
    ge_best = float(g["gross_edge"].max())
    print(f"  实测毛 edge：中位 {ge_med:.4%}  上四分位 {ge_q75:.4%}  "
          f"最好 {ge_best:.4%}")
    print(f"  （印花税 {STAMP:.2%} 是法定的，任何情景下都不能省）")

    print(f"\n  {'成本情景':<24} {'往返成本':>10} "
          f"{'净edge(中位)':>13} {'净edge(最好)':>13} {'年化@4次/日':>13}")
    print("  " + "-" * 78)
    per_trade = args.cash / 4
    cost_rows = []
    for c in cost_scenarios():
        rt = 2 * c["commission"] + STAMP + 2 * c["slippage"]
        ne_med = ge_med - rt
        ne_best = ge_best - rt
        ann = annual_from(ne_best, 4, per_trade, init, n_days)
        cost_rows.append({**c, "round_trip": rt,
                          "net_median": ne_med, "net_best": ne_best,
                          "annual_at_4": ann})
        print(f"  {c['name']:<24} {rt:>10.3%} {ne_med:>13.4%} "
              f"{ne_best:>13.4%} {ann:>13.2%}")

    # ---------- 三、达到目标需要什么 ----------
    print(f"\n{'=' * 80}")
    print(f"  三、做T年化 {args.target:.0%} 需要的条件组合")
    print(f"{'=' * 80}")
    print(f"  年化 = 净edge x 每日次数 x 每笔金额 x {n_days} / {init:,.0f}")
    print(f"\n  {'成本情景':<24} {'需要的每日次数':>16} {'可行性'}")
    print("  " + "-" * 60)
    feas = []
    for c in cost_rows:
        ne = c["net_best"]
        if ne <= 0:
            print(f"  {c['name']:<24} {'不可能（净edge≤0）':>16}  --")
            feas.append({**c, "trades_needed": None, "feasible": False})
            continue
        need = args.target * init / (ne * per_trade * n_days)
        # 一天 240 分钟，一次做 T 至少要几分钟走完
        ok = need <= 24
        note = ("可行" if ok else
                f"不可行（一天 240 分钟，{need:.0f} 次意味着"
                f"每 {240/need:.1f} 分钟一个来回）")
        print(f"  {c['name']:<24} {need:>16.0f}  {note}")
        feas.append({**c, "trades_needed": float(need), "feasible": bool(ok)})

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": f"{args.symbol}.{args.exchange}",
        "base_value": args.base, "cash_value": args.cash,
        "total_capital": init, "n_days": n_days,
        "target_annual": args.target,
        "n_combos": len(g),
        "gross_edge": {"median": ge_med, "q75": ge_q75, "best": ge_best},
        "frequency_buckets": freq_rows,
        "cost_scenarios": cost_rows,
        "feasibility": feas,
        "stamp_tax": STAMP,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print(f"\n  结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
