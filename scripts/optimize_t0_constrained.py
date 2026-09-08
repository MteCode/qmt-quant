"""带回撤约束的做 T 方案搜索 —— 同时搜底仓规模与策略参数。

## 为什么要一起搜

「回撤不超过 X%」这个约束，多数情况下**不是策略参数能满足的** ——
组合回撤主要由底仓贡献：

    组合回撤 ≈ 标的自身回撤 × 底仓占总资产比例 + 做T的损耗

600711 自身最大回撤 48.62%。10 万底仓 / 20 万总资产 = 50% 占比，
组合回撤下限就是 24.3%，无论做 T 参数怎么调都降不下来。

所以搜索空间必须包含**底仓规模**这一维。把它固定住去调策略参数，
等于在一个无解的可行域里找解。

## 输出

    models/t0_constrained/
        grid.csv        全部组合
        summary.json    约束下的最优方案 + 对照基准
        report.html     可读报告

## 用法

    python scripts/optimize_t0_constrained.py --total 200000 --max-dd 0.15
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


OUT_DIR = ROOT / "models" / "t0_constrained"


def _load_engine():
    """复用 optimize_t0_divergence 的模拟器与特征计算。

    那边已经把 T+1 约束、成本模型、量价背离信号都实现好了，
    重写一遍只会引入不一致 —— 两处逻辑分叉时，报告里的数字
    和之前的研究结论就对不上了。
    """
    spec = importlib.util.spec_from_file_location(
        "t0div", ROOT / "scripts" / "optimize_t0_divergence.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def buyhold_curve(d: pd.DataFrame, base_value: float,
                  cash_value: float) -> dict:
    """纯持有底仓不做 T 的对照组。"""
    px = d["close"].values.astype(float)
    day = d["day"].values
    # 每日收盘价
    last_idx = np.r_[np.where(day[:-1] != day[1:])[0], len(px) - 1]
    closes = px[last_idx]
    shares = int(base_value / px[0] / 100) * 100
    eq = cash_value + shares * closes
    init = base_value + cash_value
    total = eq[-1] / init - 1
    n = len(eq)
    rm = np.maximum.accumulate(eq)
    mdd = float(((eq - rm) / rm).min())
    af = 244 / n
    ann = (1 + total) ** af - 1 if total > -1 else -1
    rets = np.diff(eq) / eq[:-1] if n > 1 else np.array([0.0])
    vol = float(rets.std() * np.sqrt(244)) if n > 1 else 0.0
    return {"total_return": round(total, 4),
            "annual_return": round(float(ann), 4),
            "max_drawdown": round(mdd, 4),
            "sharpe": round(float(ann / vol) if vol > 1e-9 else 0.0, 3),
            "n_days": n}


def main() -> int:
    p = argparse.ArgumentParser(description="带回撤约束的做 T 方案搜索")
    p.add_argument("--symbol", default="600711")
    p.add_argument("--exchange", default="SSE")
    p.add_argument("--total", type=float, default=200000.0,
                   help="总资产（底仓 + 现金）")
    p.add_argument("--max-dd", type=float, default=0.15,
                   help="最大回撤上限")
    p.add_argument("--entry-time", default="09:35")
    p.add_argument("--exit-time", default="14:50")
    p.add_argument("--output", default=str(OUT_DIR))
    args = p.parse_args()

    def _tm(s):
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    m = _load_engine()
    em, xm = _tm(args.entry_time), _tm(args.exit_time)

    print("=" * 74)
    print(f"  带回撤约束的做 T 方案搜索 —— {args.symbol}.{args.exchange}")
    print(f"  总资产 {args.total:,.0f} 元   回撤上限 {args.max_dd:.0%}")
    print("=" * 74)

    df = m.load_bars(args.symbol, args.exchange)
    d = m.prep(df)
    days = pd.DatetimeIndex(d["day"].unique()).sort_values()

    # ---- 标的自身的回撤，决定底仓上限 ----
    daily_close = d.groupby("day")["close"].last().values
    rm = np.maximum.accumulate(daily_close)
    stock_dd = float(((daily_close - rm) / rm).min())
    max_base = args.total * args.max_dd / abs(stock_dd)

    print(f"  数据 {len(df):,} bar，{len(days)} 交易日，"
          f"{days[0].date()} ~ {days[-1].date()}")
    print(f"  标的自身最大回撤 {stock_dd:.2%}")
    print(f"  → 仅靠底仓就会产生的回撤 = 自身回撤 × 底仓占比")
    print(f"  → 要满足 {args.max_dd:.0%} 上限，底仓需 ≤ "
          f"{max_base:,.0f} 元（占 {max_base/args.total:.0%}）")
    print()

    # ---- 搜索空间：底仓规模 × 策略参数 ----
    base_values = sorted({
        round(args.total * r / 1000) * 1000
        for r in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1)
    })
    signals = ["corr", "obv", "pulse"]
    combos = []
    for base in base_values:
        for sig in signals:
            thrs = ([0.1, 0.2, 0.3] if sig in ("corr", "obv")
                    else [0.003, 0.005, 0.01])
            for c in itertools.product(thrs, [1.0, 2.0], [0.006, 0.010, 0.015],
                                       [0.008, 0.012], [1, 2], [30, 60],
                                       [False, True]):
                combos.append((base,) + (sig,) + c)

    print(f"  搜索 {len(combos)} 组"
          f"（底仓 {len(base_values)} 档 × 信号 {len(signals)} 类 × 参数）")

    rows = []
    t0 = time.time()
    for i, (base, sig, thr, vt, tp, sl, mt, hd, rv) in enumerate(combos, 1):
        cash = args.total - base
        r = m.simulate(d, base, cash, sig, thr, vt, tp, sl, mt,
                       em, xm, hd, rv)
        if "error" in r or r["n_trades"] == 0:
            continue
        r.pop("_daily", None)
        r.pop("_trades", None)
        r.update({"base_value": base, "cash_value": cash,
                  "base_pct": base / args.total,
                  "signal": sig, "thr": thr, "vol_thr": vt,
                  "take_profit": tp, "stop_loss": sl,
                  "max_trades": mt, "hold_max": hd, "reverse": rv})
        rows.append(r)
        if i % 300 == 0:
            print(f"    {i}/{len(combos)} ({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        print("无有效结果")
        return 1

    g = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "grid.csv", index=False)

    ok = g[g["max_drawdown"].abs() <= args.max_dd]
    print(f"\n{'=' * 74}")
    print(f"  搜索完成（{len(g)} 组有交易，耗时 {time.time()-t0:.0f}s）")
    print(f"  满足回撤 ≤{args.max_dd:.0%} 的: {len(ok)} 组")
    print(f"{'=' * 74}")

    # ---- 各底仓档位的对照基准（纯持有不做 T）----
    print(f"\n  各底仓档位 —— 纯持有不做 T 的表现：")
    print(f"  {'底仓':>10} {'占比':>6} {'年化':>9} {'回撤':>9} "
          f"{'夏普':>7}  {'满足约束'}")
    print(f"  {'-'*10} {'-'*6} {'-'*9} {'-'*9} {'-'*7}  {'-'*8}")
    bh_rows = []
    for base in base_values:
        bh = buyhold_curve(d, base, args.total - base)
        bh["base_value"] = base
        bh["base_pct"] = base / args.total
        bh_rows.append(bh)
        mark = "是" if abs(bh["max_drawdown"]) <= args.max_dd else "否"
        print(f"  {base:>10,} {base/args.total:>6.0%} "
              f"{bh['annual_return']:>8.2%} {bh['max_drawdown']:>9.2%} "
              f"{bh['sharpe']:>7.2f}  {mark}")

    best_bh = max((b for b in bh_rows
                   if abs(b["max_drawdown"]) <= args.max_dd),
                  key=lambda x: x["annual_return"], default=None)

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": f"{args.symbol}.{args.exchange}",
        "total_capital": args.total,
        "max_drawdown_limit": args.max_dd,
        "date_range": [str(days[0].date()), str(days[-1].date())],
        "n_days": len(days),
        "stock_max_drawdown": round(stock_dd, 4),
        "max_base_for_constraint": round(max_base, 0),
        "n_combos": len(g),
        "n_satisfying": len(ok),
        "buyhold_by_base": bh_rows,
        "best_buyhold_under_constraint": best_bh,
        "cost_model": {"commission": m.COMMISSION, "stamp_tax": m.STAMP_TAX,
                       "slippage": m.SLIPPAGE, "round_trip": m.ROUND_TRIP},
    }

    if len(ok):
        best = ok.nlargest(1, "annual_return").iloc[0].to_dict()
        summary["best_under_constraint"] = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in best.items() if not k.startswith("_")
        }
        print(f"\n  约束下最优做 T 方案：")
        print(f"    底仓 {best['base_value']:,.0f} "
              f"({best['base_pct']:.0%}) + 现金 "
              f"{best['cash_value']:,.0f}")
        print(f"    信号 {best['signal']} thr={best['thr']} "
              f"止盈{best['take_profit']:.1%} 止损{best['stop_loss']:.1%} "
              f"×{int(best['max_trades'])}")
        print(f"    年化 {best['annual_return']:.2%}  "
              f"回撤 {best['max_drawdown']:.2%}  "
              f"做T贡献 {best['t_annual']:+.2%}")
        if best_bh:
            delta = best["annual_return"] - best_bh["annual_return"]
            print(f"\n  对照：同约束下纯持有最优 年化 "
                  f"{best_bh['annual_return']:.2%}"
                  f"（底仓 {best_bh['base_value']:,.0f}）")
            print(f"  做 T 相对纯持有: {delta:+.2%}")
            summary["t_vs_buyhold"] = round(float(delta), 4)
    else:
        print(f"\n  没有任何做 T 组合能满足 {args.max_dd:.0%} 回撤约束。")
        if best_bh:
            print(f"  同约束下纯持有最优：年化 "
                  f"{best_bh['annual_return']:.2%}，"
                  f"回撤 {best_bh['max_drawdown']:.2%}"
                  f"（底仓 {best_bh['base_value']:,.0f}）")

    # 做 T 为正的比例 —— 判断是噪声还是真信号的关键
    summary["n_positive_t"] = int((g["t_annual"] > 0).sum())
    summary["pct_positive_t"] = round(float((g["t_annual"] > 0).mean()), 4)
    summary["t_annual_mean"] = round(float(g["t_annual"].mean()), 4)
    print(f"\n  全体组合中做 T 为正的: "
          f"{summary['n_positive_t']}/{len(g)} "
          f"({summary['pct_positive_t']:.1%})，"
          f"均值 {summary['t_annual_mean']:.2%}")

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\n  结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
