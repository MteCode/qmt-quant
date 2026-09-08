"""资金配置优化 —— 仓位与做 T 一起定，不是先定仓位再调参数。

## 为什么合在一起

之前的研究把「底仓多少」当成给定条件，只在策略参数上找最优。
但组合收益是两部分之和：

    总收益 = 底仓的持有收益(beta) + 做T的价差收益(alpha)

底仓规模同时决定 beta 的大小和回撤的大小，而做 T 的规模受底仓约束
（正T 买入后要从底仓卖出等量，底仓不够这条腿平不掉）。
两者耦合，分开优化会得到局部最优。

## 输出的是「有效前沿」，不是单一方案

「最优」取决于能承受多大回撤。所以扫遍配置，对每个回撤档位给出
该档位下收益最高的方案 —— 你看到的是一条曲线，可以自己在上面选点，
而不是我替你假定风险偏好。

## 一个必须诚实标注的事

做 T 的 alpha 已被 22032 组参数搜索证否（为正 0.1%，样本外 0/20）。
所以「最优方案」里做 T 的贡献大概率是负的，收益全部来自底仓。
报告会把两部分分开列，不把 beta 伪装成策略业绩。
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

OUT_DIR = ROOT / "models" / "allocation"


def _engine():
    spec = importlib.util.spec_from_file_location(
        "t0div", ROOT / "scripts" / "optimize_t0_divergence.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def buyhold(d: pd.DataFrame, base_value: float, cash_value: float) -> dict:
    """纯持有底仓、其余现金不动的对照组。"""
    px = d["close"].values.astype(float)
    day = d["day"].values
    last_idx = np.r_[np.where(day[:-1] != day[1:])[0], len(px) - 1]
    closes = px[last_idx]
    shares = int(base_value / px[0] / 100) * 100
    eq = cash_value + shares * closes
    init = base_value + cash_value
    if init <= 0:
        return {}
    total = eq[-1] / init - 1
    n = len(eq)
    rm = np.maximum.accumulate(eq)
    mdd = float(((eq - rm) / rm).min())
    af = 244 / n
    ann = (1 + total) ** af - 1 if total > -1 else -1.0
    rets = np.diff(eq) / eq[:-1] if n > 1 else np.array([0.0])
    vol = float(rets.std() * np.sqrt(244)) if n > 1 else 0.0
    return {"total_return": float(total), "annual_return": float(ann),
            "max_drawdown": mdd, "sharpe": float(ann / vol) if vol > 1e-9 else 0.0,
            "volatility": vol, "n_days": n}


def main() -> int:
    p = argparse.ArgumentParser(description="资金配置优化")
    p.add_argument("--symbol", default="600711")
    p.add_argument("--exchange", default="SSE")
    p.add_argument("--capital", type=float, default=200000.0,
                   help="可动用的总资金")
    p.add_argument("--cost-basis", type=float, default=10.8,
                   help="现有持仓成本价（仅用于报告，不影响回测收益率）")
    p.add_argument("--entry-time", default="09:35")
    p.add_argument("--exit-time", default="14:50")
    p.add_argument("--output", default=str(OUT_DIR))
    args = p.parse_args()

    def _tm(s):
        h, mm = s.split(":")
        return int(h) * 60 + int(mm)

    m = _engine()
    em, xm = _tm(args.entry_time), _tm(args.exit_time)
    cap = args.capital

    print("=" * 80)
    print(f"  资金配置优化 —— {args.symbol}.{args.exchange}")
    print(f"  可动用资金 {cap:,.0f} 元   现有持仓成本价 {args.cost_basis} 元")
    print("=" * 80)

    df = m.load_bars(args.symbol, args.exchange)
    d = m.prep(df)
    days = pd.DatetimeIndex(d["day"].unique()).sort_values()
    n_days = len(days)

    # 标的自身属性 —— 决定了配置的可行域
    dc = d.groupby("day")["close"].last()
    eq = dc.values
    rm = np.maximum.accumulate(eq)
    stock_dd = float(((eq - rm) / rm).min())
    stock_ret = float(eq[-1] / eq[0] - 1)
    stock_ann = (1 + stock_ret) ** (244 / n_days) - 1
    drets = np.diff(eq) / eq[:-1]
    stock_vol = float(drets.std() * np.sqrt(244))

    print(f"  {n_days} 交易日 {days[0].date()} ~ {days[-1].date()}")
    print(f"\n  标的自身：区间涨幅 {stock_ret:+.2%}（年化 {stock_ann:+.2%}），"
          f"最大回撤 {stock_dd:.2%}，年化波动 {stock_vol:.1%}")
    print(f"  → 满仓持有的年化就是 {stock_ann:.2%}，回撤 {stock_dd:.2%}。")
    print(f"    任何配置的收益都不会超过它 —— 除非做 T 能创造正 alpha，")
    print(f"    而那已被 22032 组参数搜索证否。")

    # ---------- 一、纯持有的有效前沿 ----------
    print(f"\n{'=' * 80}")
    print("  一、纯持有（不做 T）—— 仓位与收益/回撤的关系")
    print(f"{'=' * 80}")
    print(f"  {'仓位':>6} {'底仓':>10} {'现金':>10} {'年化':>9} "
          f"{'回撤':>9} {'夏普':>7} {'波动':>8}")
    print("  " + "-" * 64)
    bh_rows = []
    for ratio in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        base = round(cap * ratio / 1000) * 1000
        r = buyhold(d, base, cap - base)
        if not r:
            continue
        r.update({"ratio": ratio, "base_value": base,
                  "cash_value": cap - base})
        bh_rows.append(r)
        print(f"  {ratio:>6.0%} {base:>10,.0f} {cap-base:>10,.0f} "
              f"{r['annual_return']:>9.2%} {r['max_drawdown']:>9.2%} "
              f"{r['sharpe']:>7.2f} {r['volatility']:>8.1%}")

    print(f"\n  夏普在各仓位下基本不变 —— 现金不产生收益也不产生波动，")
    print(f"  调仓位只是在同一条射线上滑动，改变不了风险收益比。")

    # ---------- 二、加入做 T 后 ----------
    print(f"\n{'=' * 80}")
    print("  二、加入做 T —— 底仓规模与策略参数联合搜索")
    print(f"{'=' * 80}")

    base_ratios = (0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
    combos = []
    for ratio in base_ratios:
        base = round(cap * ratio / 1000) * 1000
        for sig in ("corr", "obv", "pulse"):
            thrs = ([0.1, 0.2, 0.3] if sig in ("corr", "obv")
                    else [0.003, 0.005, 0.01])
            for c in itertools.product(thrs, [1.0, 2.0], [0.006, 0.010, 0.015],
                                       [0.008, 0.012], [1, 2, 4], [30, 60],
                                       [False, True]):
                combos.append((base, ratio, sig) + c)

    print(f"  搜索 {len(combos)} 组（底仓 {len(base_ratios)} 档 × 参数）")
    rows = []
    t0 = time.time()
    for i, (base, ratio, sig, thr, vt, tp, sl, mt, hd, rv) in \
            enumerate(combos, 1):
        cash = cap - base
        r = m.simulate(d, base, cash, sig, thr, vt, tp, sl, mt,
                       em, xm, hd, rv)
        if "error" in r or r["n_trades"] == 0:
            continue
        r.pop("_daily", None)
        r.pop("_trades", None)
        r.update({"base_value": base, "cash_value": cash, "ratio": ratio,
                  "signal": sig, "thr": thr, "vol_thr": vt,
                  "take_profit": tp, "stop_loss": sl,
                  "max_trades": mt, "hold_max": hd, "reverse": rv})
        rows.append(r)
        if i % 500 == 0:
            print(f"    {i}/{len(combos)} ({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        print("无有效结果")
        return 1

    g = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "grid.csv", index=False)

    npos = int((g["t_annual"] > 0).sum())
    print(f"\n  {len(g)} 组有交易。做 T 为正的: {npos} "
          f"({npos/len(g):.2%})，均值 {g['t_annual'].mean():.2%}")

    # ---------- 三、有效前沿 ----------
    print(f"\n{'=' * 80}")
    print("  三、有效前沿 —— 每个回撤档位下收益最高的方案")
    print(f"{'=' * 80}")
    print(f"  {'回撤上限':>9} {'最优方案':<14} {'底仓':>10} "
          f"{'年化':>9} {'实际回撤':>10} {'做T贡献':>10} {'对比纯持有':>11}")
    print("  " + "-" * 78)

    frontier = []
    for cap_dd in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        # 做 T 方案里满足该回撤的最优
        ok_t = g[g["max_drawdown"].abs() <= cap_dd]
        best_t = (ok_t.nlargest(1, "annual_return").iloc[0]
                  if len(ok_t) else None)
        # 纯持有里满足该回撤的最优
        ok_b = [b for b in bh_rows if abs(b["max_drawdown"]) <= cap_dd]
        best_b = (max(ok_b, key=lambda x: x["annual_return"])
                  if ok_b else None)

        if best_t is None and best_b is None:
            print(f"  {cap_dd:>9.0%} {'无可行方案':<14}")
            frontier.append({"max_dd": cap_dd, "feasible": False})
            continue

        t_ann = best_t["annual_return"] if best_t is not None else -99
        b_ann = best_b["annual_return"] if best_b else -99
        if t_ann >= b_ann and best_t is not None:
            kind, ann = "做T", t_ann
            base_v, real_dd = best_t["base_value"], best_t["max_drawdown"]
            t_contrib = best_t["t_annual"]
        else:
            kind, ann = "纯持有", b_ann
            base_v, real_dd = best_b["base_value"], best_b["max_drawdown"]
            t_contrib = 0.0
        delta = ann - b_ann if best_b else float("nan")
        print(f"  {cap_dd:>9.0%} {kind:<14} {base_v:>10,.0f} "
              f"{ann:>9.2%} {real_dd:>10.2%} {t_contrib:>+10.2%} "
              f"{delta:>+11.2%}")
        frontier.append({
            "max_dd": cap_dd, "feasible": True, "kind": kind,
            "annual_return": float(ann), "base_value": float(base_v),
            "actual_dd": float(real_dd), "t_contribution": float(t_contrib),
            "vs_buyhold": float(delta) if best_b else None,
            "params": ({k: (best_t[k].item()
                            if hasattr(best_t[k], "item") else best_t[k])
                        for k in ("signal", "thr", "vol_thr", "take_profit",
                                  "stop_loss", "max_trades", "hold_max",
                                  "reverse")}
                       if kind == "做T" else None),
        })

    n_t_wins = sum(1 for f in frontier
                   if f.get("feasible") and f.get("kind") == "做T")
    print(f"\n  在 {len(frontier)} 个回撤档位中，做 T 方案胜出 {n_t_wins} 次。")
    if n_t_wins == 0:
        print(f"  → 任何风险偏好下，纯持有都不劣于做 T。")

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": f"{args.symbol}.{args.exchange}",
        "capital": cap, "cost_basis": args.cost_basis,
        "date_range": [str(days[0].date()), str(days[-1].date())],
        "n_days": n_days,
        "stock": {"total_return": stock_ret, "annual_return": float(stock_ann),
                  "max_drawdown": stock_dd, "volatility": stock_vol},
        "buyhold_frontier": bh_rows,
        "n_combos": len(g),
        "n_positive_t": npos,
        "pct_positive_t": round(npos / len(g), 4),
        "t_annual_mean": round(float(g["t_annual"].mean()), 4),
        "frontier": frontier,
        "t_wins": n_t_wins,
        "cost_model": {"round_trip": m.ROUND_TRIP},
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\n  结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
