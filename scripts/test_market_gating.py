"""市场门控对照实验 —— 只在大盘配合时做 T，有用吗？

## 为什么用对照实验而不是暴力搜索

「门控有没有用」是一个**因果问题**，不是「找最大值」的问题。
把门控和策略参数一起丢进 2 万组网格里搜，最后拿最优组的门控参数说
「门控有用」，是把多重检验的噪声当成了结论 —— 最优组恰好带门控，
不代表门控起了作用。

正确做法是控制变量：**同一组策略参数**，只改门控，看配对差异。
每组参数自己和自己比，策略本身的好坏被抵消掉，剩下的就是门控的贡献。

## 门控条件

- `min_breadth`：上涨家数占比下限。普跌时不抄底
- `min_day_ret`：大盘当日涨幅下限。大盘在跌就不做多头方向的 T
- `min_vol_z`：全市场量能下限。极度缩量时日内没波动空间

## 判读

看的是**配对差异的分布**，不是最大值：
- 中位数显著为正 → 门控真的有用
- 中位数接近 0、只有个别组变好 → 噪声
- 中位数为负 → 门控反而有害（过滤掉了有效信号）
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

# 控制台是 GBK 时，数学减号、警告符号这类字符会直接抛
# UnicodeEncodeError 让脚本崩在 print 上 —— 算了半小时的结果全丢。
# 降级为替换字符，宁可显示成 ? 也不能因为一个字符丢掉整轮结果。
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass


OUT_DIR = ROOT / "models" / "t0_market"


def _engine():
    spec = importlib.util.spec_from_file_location(
        "t0div", ROOT / "scripts" / "optimize_t0_divergence.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _market():
    spec = importlib.util.spec_from_file_location(
        "t0mkt", ROOT / "scripts" / "optimize_t0_market.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    p = argparse.ArgumentParser(description="市场门控对照实验")
    p.add_argument("--symbol", default="600711")
    p.add_argument("--exchange", default="SSE")
    p.add_argument("--base", type=float, default=100000.0)
    p.add_argument("--cash", type=float, default=100000.0)
    p.add_argument("--entry-time", default="09:35")
    p.add_argument("--exit-time", default="14:50")
    p.add_argument("--output", default=str(OUT_DIR))
    args = p.parse_args()

    def _tm(s):
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    m = _engine()
    mk = _market()
    em, xm = _tm(args.entry_time), _tm(args.exit_time)
    init = args.base + args.cash

    print("=" * 76)
    print(f"  市场门控对照实验 —— {args.symbol}.{args.exchange}")
    print(f"  底仓 {args.base:,.0f} + 现金 {args.cash:,.0f} = {init:,.0f} 元")
    print("=" * 76)

    df = m.load_bars(args.symbol, args.exchange)
    d = m.prep(df)
    d = mk.attach_market(d)
    days = pd.DatetimeIndex(d["day"].unique()).sort_values()
    print(f"  {len(days)} 交易日 {days[0].date()} ~ {days[-1].date()}，"
          f"市场数据覆盖 {d.attrs['market_coverage']:.0%}")

    # ---- 策略参数：覆盖三类信号的代表性组合 ----
    base_params = []
    for sig in ("corr", "obv", "pulse"):
        thrs = ([0.1, 0.2, 0.3] if sig in ("corr", "obv")
                else [0.003, 0.005, 0.01])
        for thr, vt, tp, sl, mt, hd, rv in itertools.product(
                thrs, [1.0, 2.0], [0.006, 0.015], [0.008, 0.012],
                [1, 2], [30, 60], [False, True]):
            base_params.append((sig, thr, vt, tp, sl, mt, hd, rv))

    # ---- 门控档位：逐条单独开，避免多条件耦合 ----
    NO_GATE = (0.0, -1.0, -99.0)
    gates = {
        "无门控": NO_GATE,
        "宽度≥0.30": (0.30, -1.0, -99.0),
        "宽度≥0.40": (0.40, -1.0, -99.0),
        "大盘≥0": (0.0, 0.0, -99.0),
        "大盘≥-0.5%": (0.0, -0.005, -99.0),
        "量能≥-0.8": (0.0, -1.0, -0.8),
        "量能≥-0.3": (0.0, -1.0, -0.3),
        "宽度0.30+大盘≥0": (0.30, 0.0, -99.0),
    }

    print(f"  策略参数 {len(base_params)} 组 × 门控 {len(gates)} 档 "
          f"= {len(base_params) * len(gates)} 次模拟")

    rows = []
    t0 = time.time()
    n = 0
    for sig, thr, vt, tp, sl, mt, hd, rv in base_params:
        for gname, (mb, mdr, mvz) in gates.items():
            r = mk.simulate_gated(m, d, args.base, args.cash, sig, thr, vt,
                                  tp, sl, mt, em, xm, hd, rv, mb, mdr, mvz)
            n += 1
            if "error" in r:
                continue
            r.pop("_daily", None)
            r.pop("_trades", None)
            rows.append({
                "param_id": f"{sig}|{thr}|{vt}|{tp}|{sl}|{mt}|{hd}|{rv}",
                "signal": sig, "gate": gname,
                "min_breadth": mb, "min_day_ret": mdr, "min_vol_z": mvz,
                "annual_return": r["annual_return"],
                "t_annual": r["t_annual"],
                "max_drawdown": r["max_drawdown"],
                "sharpe": r["sharpe"], "n_trades": r["n_trades"],
                "coverage": r["coverage"],
                "gross_edge": r["gross_edge"], "net_edge": r["net_edge"],
                "buyhold_annual": r["buyhold_annual"],
            })
            if n % 400 == 0:
                print(f"    {n}/{len(base_params)*len(gates)} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    g = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "gating_experiment.csv", index=False)

    bh = float(g["buyhold_annual"].iloc[0])
    print(f"\n{'=' * 76}")
    print(f"  完成 {len(g)} 次有效模拟，耗时 {time.time()-t0:.0f}s")
    print(f"  基准 —— 纯持有不做 T: 年化 {bh:.2%}")
    print(f"{'=' * 76}")

    # ---- 各门控档位的整体表现 ----
    print(f"\n  各门控档位（每档 {len(base_params)} 组参数的分布）：")
    print(f"  {'门控':<18} {'做T年化中位':>12} {'均值':>10} {'最好':>10} "
          f"{'为正占比':>9} {'交易数中位':>11}")
    print("  " + "-" * 76)
    agg = []
    for gname in gates:
        s = g[g["gate"] == gname]
        if s.empty:
            continue
        agg.append({
            "gate": gname, "n": len(s),
            "median": float(s["t_annual"].median()),
            "mean": float(s["t_annual"].mean()),
            "best": float(s["t_annual"].max()),
            "pct_pos": float((s["t_annual"] > 0).mean()),
            "median_trades": float(s["n_trades"].median()),
        })
        print(f"  {gname:<18} {s['t_annual'].median():>12.2%} "
              f"{s['t_annual'].mean():>10.2%} {s['t_annual'].max():>10.2%} "
              f"{(s['t_annual'] > 0).mean():>9.1%} "
              f"{s['n_trades'].median():>11.0f}")

    # ---- 配对差异：同一组参数，带门控 vs 不带 ----
    print(f"\n  配对差异（同一组策略参数，门控 - 无门控）：")
    print(f"  这才是门控的真实贡献 —— 策略本身的好坏被抵消掉了")
    print(f"  {'门控':<18} {'差异中位':>10} {'差异均值':>10} "
          f"{'变好占比':>9} {'交易量变化':>11}")
    print("  " + "-" * 66)
    base = g[g["gate"] == "无门控"].set_index("param_id")
    paired = []
    for gname in gates:
        if gname == "无门控":
            continue
        s = g[g["gate"] == gname].set_index("param_id")
        common = base.index.intersection(s.index)
        if len(common) == 0:
            continue
        delta = s.loc[common, "t_annual"] - base.loc[common, "t_annual"]
        tr_ratio = (s.loc[common, "n_trades"]
                    / base.loc[common, "n_trades"].replace(0, np.nan))
        paired.append({
            "gate": gname, "n_pairs": len(common),
            "delta_median": float(delta.median()),
            "delta_mean": float(delta.mean()),
            "pct_improved": float((delta > 0).mean()),
            "trade_ratio_median": float(tr_ratio.median()),
        })
        print(f"  {gname:<18} {delta.median():>10.2%} {delta.mean():>10.2%} "
              f"{(delta > 0).mean():>9.1%} {tr_ratio.median():>11.2f}x")

    # ---- 收益最高的组合 ----
    print(f"\n  收益最高的 10 组（按总年化）：")
    print(f"  {'#':<3} {'信号':<6} {'门控':<18} {'年化':>9} {'做T年化':>9} "
          f"{'回撤':>9} {'夏普':>7} {'交易':>6}")
    print("  " + "-" * 76)
    for i, (_, r) in enumerate(g.nlargest(10, "annual_return").iterrows(), 1):
        print(f"  {i:<3} {r['signal']:<6} {r['gate']:<18} "
              f"{r['annual_return']:>9.2%} {r['t_annual']:>9.2%} "
              f"{r['max_drawdown']:>9.2%} {r['sharpe']:>7.2f} "
              f"{int(r['n_trades']):>6}")

    best = g.nlargest(1, "annual_return").iloc[0]
    verdict = _verdict(paired)

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": f"{args.symbol}.{args.exchange}",
        "base_value": args.base, "cash_value": args.cash,
        "total_capital": init,
        "date_range": [str(days[0].date()), str(days[-1].date())],
        "n_days": len(days),
        "market_coverage": round(d.attrs["market_coverage"], 4),
        "design": "对照实验：同一组策略参数，只改门控，看配对差异",
        "n_param_sets": len(base_params),
        "n_gates": len(gates),
        "n_simulations": len(g),
        "buyhold_annual": bh,
        "by_gate": agg,
        "paired_delta": paired,
        "verdict": verdict,
        "best_overall": {
            "signal": best["signal"], "gate": best["gate"],
            "annual_return": float(best["annual_return"]),
            "t_annual": float(best["t_annual"]),
            "max_drawdown": float(best["max_drawdown"]),
            "sharpe": float(best["sharpe"]),
            "n_trades": int(best["n_trades"]),
            "gross_edge": float(best["gross_edge"]),
            "net_edge": float(best["net_edge"]),
            "vs_buyhold": float(best["annual_return"] - bh),
        },
        "overall_positive_t": int((g["t_annual"] > 0).sum()),
        "overall_pct_positive": round(float((g["t_annual"] > 0).mean()), 4),
        "cost_model": {"round_trip": m.ROUND_TRIP},
    }
    (out / "gating_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print(f"\n{'=' * 76}")
    print(f"  结论：{verdict}")
    print(f"{'=' * 76}")
    print(f"  全体 {len(g)} 次模拟中做 T 为正的: "
          f"{summary['overall_positive_t']} "
          f"({summary['overall_pct_positive']:.1%})")
    print(f"  收益最高组合: {best['signal']} + {best['gate']}，"
          f"年化 {best['annual_return']:.2%}"
          f"（做T {best['t_annual']:+.2%}）")
    print(f"  相对纯持有: {best['annual_return'] - bh:+.2%}")
    print(f"\n  结果已保存: {out}")
    return 0


def _verdict(paired: list) -> str:
    """从配对差异得出结论。看中位数与变好占比，不看最大值。"""
    if not paired:
        return "无有效配对，无法判断"
    meds = [p["delta_median"] for p in paired]
    best_med = max(meds)
    improved = max(p["pct_improved"] for p in paired)
    if best_med > 0.02 and improved > 0.6:
        return ("市场门控有效 —— 配对差异中位数显著为正，"
                "且多数参数组都变好")
    if best_med > 0:
        return ("市场门控效果微弱 —— 配对差异中位数为正但幅度小，"
                "不足以支撑「大盘配合时做T更好」的结论")
    return ("市场门控无效或有害 —— 配对差异中位数不为正，"
            "过滤掉的信号并不比保留的差")


if __name__ == "__main__":
    raise SystemExit(main())
