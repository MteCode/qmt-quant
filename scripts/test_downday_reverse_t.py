"""验证「只在下跌时做反 T」—— 用实时可观测的日内跌幅做门控。

## 为什么不能用「下跌日」定义

归因分析发现盈利日的唯一区分特征是「当天股价在跌」（盈利日涨幅中位
-2.75%，亏损日 +0.33%，分布重叠仅 12%）。但那是**收盘后才知道的结果**，
拿它当条件是前视 —— 开盘时你不知道今天会不会跌。

唯一诚实的做法是用**当前时刻相对开盘的跌幅**做门控：这个量在每一分钟
都是已知的，可以真实执行。

代价是它比事后的「下跌日」弱得多：日内跌 2% 的时刻，今天最终可能收涨。
这正是要验证的 —— 实时可观测的版本还剩多少效果。

## 为什么只做反 T

反 T = 先从底仓卖出、跌下去再买回。下跌中它天然占优：
卖在相对高位、买在相对低位。正 T（先买后卖）在下跌中则是逆势接刀。

归因显示的「跌的日子做 T 赚钱」，本质就是反 T 方向占了便宜。
所以这里只测反 T，把方向固定住，看门控本身的贡献。

## 对照组

1. 无门控 + 双向（原策略）
2. 无门控 + 只反 T
3. 日内跌幅门控 + 只反 T   ← 要验证的
4. 事后下跌日 + 只反 T     ← 前视上界，用来看「如果能预知」最多能赚多少

第 4 组不可实现，但它给出天花板：如果连它都不赚钱，那这条路彻底不通。
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

OUT_DIR = ROOT / "models" / "t0_downday"


def _engine():
    spec = importlib.util.spec_from_file_location(
        "t0div", ROOT / "scripts" / "optimize_t0_divergence.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gate_by_intraday_drop(d: pd.DataFrame, threshold: float,
                          lookahead: bool = False) -> pd.Series:
    """门控掩码：允许交易的时刻。

    :param threshold: 日内跌幅门槛（负数，如 -0.01 表示跌 1% 才开始做）
    :param lookahead: True 时用当日收盘涨幅判断（**前视，仅作上界参照**）
    """
    if lookahead:
        # 用当日最终涨跌决定整天是否交易 —— 不可实现，只用来看天花板
        day_final = d.groupby("day")["close"].transform("last")
        day_open = d.groupby("day")["open"].transform("first")
        return (day_final / day_open - 1) <= threshold
    # 实时可观测：当前收盘价相对当日开盘的涨跌
    return d["day_ret"] <= threshold


def run_variant(m, d, base, cash, sig, thr, vt, tp, sl, mt, em, xm, hd,
                reverse_only: bool, gate: pd.Series | None):
    """跑一个变体。

    reverse_only 通过屏蔽正 T 的开仓条件实现：把「价跌」信号方向的
    触发条件置为不可能，只留下「价涨→卖出」这一侧。
    """
    dd = d.copy()
    if gate is not None:
        # 门控外的时刻信号置中性，模拟器据此判定无信号
        for col, neutral in (("pv_corr", 0.0), ("obv_div", 0.0),
                             ("vol_z", -99.0), ("ret_n", 0.0)):
            if col in dd.columns:
                dd.loc[~gate, col] = neutral

    if reverse_only:
        # 模拟器里 ret_n > 0 触发卖出（反T）、ret_n < 0 触发买入（正T）。
        # 把负的 ret_n 抬到 0，正 T 就再也不会触发
        dd["ret_n"] = dd["ret_n"].clip(lower=0.0)
        # corr/obv 信号同样按 ret_n 的符号分方向，clip 后自动只剩反T

    return m.simulate(dd, base, cash, sig, thr, vt, tp, sl, mt,
                      em, xm, hd, False)


def main() -> int:
    p = argparse.ArgumentParser(description="下跌反 T 验证")
    p.add_argument("--symbol", default="600711")
    p.add_argument("--exchange", default="SSE")
    p.add_argument("--base", type=float, default=100000.0)
    p.add_argument("--cash", type=float, default=100000.0)
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

    print("=" * 78)
    print(f"  下跌反 T 验证 —— {args.symbol}.{args.exchange}")
    print(f"  底仓 {args.base:,.0f} + 现金 {args.cash:,.0f} = {init:,.0f} 元")
    print("=" * 78)

    df = m.load_bars(args.symbol, args.exchange)
    d = m.prep(df)
    days = pd.DatetimeIndex(d["day"].unique()).sort_values()
    print(f"  {len(days)} 交易日 {days[0].date()} ~ {days[-1].date()}")

    # 门控覆盖率 —— 先看这个，覆盖太低就没有统计意义
    print(f"\n  各门控档位的时刻覆盖率：")
    for th in (-0.005, -0.01, -0.02, -0.03):
        g = gate_by_intraday_drop(d, th)
        gl = gate_by_intraday_drop(d, th, lookahead=True)
        print(f"    日内跌幅 <= {th:>6.1%}: 实时 {g.mean():>6.1%} 的时刻   "
              f"（事后下跌日口径 {gl.mean():>6.1%}）")

    # ---- 策略参数：沿用之前搜索里的代表组合 ----
    params = []
    for sig in ("corr", "obv", "pulse"):
        thrs = ([0.1, 0.2, 0.3] if sig in ("corr", "obv")
                else [0.003, 0.005, 0.01])
        for thr, vt, tp, sl, mt, hd in itertools.product(
                thrs, [1.0, 2.0], [0.006, 0.015], [0.008, 0.012],
                [1, 2], [30, 60]):
            params.append((sig, thr, vt, tp, sl, mt, hd))

    variants = [
        ("原策略（双向、无门控）", False, None, False),
        ("只反T、无门控", True, None, False),
        ("只反T + 日内跌0.5%", True, -0.005, False),
        ("只反T + 日内跌1%", True, -0.01, False),
        ("只反T + 日内跌2%", True, -0.02, False),
        ("[前视]只反T + 下跌日", True, -0.0, True),
    ]

    print(f"\n  策略参数 {len(params)} 组 × 变体 {len(variants)} 个 "
          f"= {len(params) * len(variants)} 次模拟")

    rows = []
    t0 = time.time()
    n = 0
    for vname, rev_only, gth, la in variants:
        gate = None if gth is None else gate_by_intraday_drop(d, gth, la)
        for sig, thr, vt, tp, sl, mt, hd in params:
            r = run_variant(m, d, args.base, args.cash, sig, thr, vt, tp, sl,
                            mt, em, xm, hd, rev_only, gate)
            n += 1
            if "error" in r or r["n_trades"] == 0:
                continue
            r.pop("_daily", None)
            r.pop("_trades", None)
            rows.append({
                "variant": vname, "lookahead": la,
                "param_id": f"{sig}|{thr}|{vt}|{tp}|{sl}|{mt}|{hd}",
                "signal": sig,
                "annual_return": r["annual_return"],
                "t_annual": r["t_annual"],
                "max_drawdown": r["max_drawdown"],
                "sharpe": r["sharpe"], "n_trades": r["n_trades"],
                "win_rate": r["win_rate"], "coverage": r["coverage"],
                "gross_edge": r["gross_edge"], "net_edge": r["net_edge"],
                "buyhold_annual": r["buyhold_annual"],
            })
            if n % 300 == 0:
                print(f"    {n}/{len(params)*len(variants)} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        print("无有效结果")
        return 1

    g = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "variants.csv", index=False)
    bh = float(g["buyhold_annual"].iloc[0])

    print(f"\n{'=' * 78}")
    print(f"  完成 {len(g)} 次有效模拟，耗时 {time.time()-t0:.0f}s")
    print(f"  基准 —— 纯持有不做 T: 年化 {bh:.2%}")
    print(f"{'=' * 78}")

    print(f"\n  {'变体':<26} {'做T年化中位':>12} {'最好':>10} "
          f"{'为正占比':>9} {'胜率中位':>9} {'交易中位':>9} {'单笔':>10}")
    print("  " + "-" * 88)
    agg = []
    for vname, _, _, la in variants:
        s = g[g["variant"] == vname]
        if s.empty:
            continue
        per = (s["t_annual"] / s["n_trades"].replace(0, np.nan)).median()
        agg.append({"variant": vname, "lookahead": la, "n": len(s),
                    "median": float(s["t_annual"].median()),
                    "best": float(s["t_annual"].max()),
                    "pct_pos": float((s["t_annual"] > 0).mean()),
                    "win_rate": float(s["win_rate"].median()),
                    "median_trades": float(s["n_trades"].median()),
                    "per_trade": float(per)})
        tag = " [前视]" if la else ""
        print(f"  {vname:<26} {s['t_annual'].median():>+12.2%} "
              f"{s['t_annual'].max():>+10.2%} "
              f"{(s['t_annual'] > 0).mean():>9.1%} "
              f"{s['win_rate'].median():>9.1%} "
              f"{s['n_trades'].median():>9.0f} {per:>10.5f}{tag}")

    # ---- 配对差异：每组参数自己和「只反T、无门控」比 ----
    print(f"\n  配对差异（同一组参数，门控版 - 只反T无门控版）：")
    print(f"  {'变体':<26} {'差异中位':>10} {'变好占比':>9} "
          f"{'交易量比':>9} {'单笔亏损比':>11}")
    print("  " + "-" * 70)
    base = g[g["variant"] == "只反T、无门控"].set_index("param_id")
    base_per = base["t_annual"] / base["n_trades"].replace(0, np.nan)
    paired = []
    for vname, _, gth, la in variants:
        if vname == "只反T、无门控" or gth is None:
            continue
        s = g[g["variant"] == vname].set_index("param_id")
        common = base.index.intersection(s.index)
        if not len(common):
            continue
        delta = s.loc[common, "t_annual"] - base.loc[common, "t_annual"]
        tr = (s.loc[common, "n_trades"]
              / base.loc[common, "n_trades"].replace(0, np.nan))
        per = s.loc[common, "t_annual"] / s.loc[common, "n_trades"].replace(0, np.nan)
        pr = per / base_per.loc[common].replace(0, np.nan)
        paired.append({"variant": vname, "lookahead": la,
                       "delta_median": float(delta.median()),
                       "pct_improved": float((delta > 0).mean()),
                       "trade_ratio": float(tr.median()),
                       "per_trade_ratio": float(pr.median())})
        tag = " [前视]" if la else ""
        print(f"  {vname:<26} {delta.median():>+10.2%} "
              f"{(delta > 0).mean():>9.1%} {tr.median():>8.2f}x "
              f"{pr.median():>10.2f}x{tag}")

    print(f"\n  判读：**单笔亏损比**显著小于 1 才说明门控筛出了更好的交易；")
    print(f"  接近 1 说明只是少交易了。前视版给出天花板 —— 连它都不行，")
    print(f"  实时版更不可能。")

    best = g[~g["lookahead"]].nlargest(1, "t_annual")
    la_best = g[g["lookahead"]].nlargest(1, "t_annual")
    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": f"{args.symbol}.{args.exchange}",
        "base_value": args.base, "cash_value": args.cash,
        "date_range": [str(days[0].date()), str(days[-1].date())],
        "n_days": len(days),
        "buyhold_annual": bh,
        "by_variant": agg,
        "paired_vs_reverse_only": paired,
        "best_realizable": best.iloc[0].to_dict() if len(best) else None,
        "best_lookahead_ceiling": (la_best.iloc[0].to_dict()
                                   if len(la_best) else None),
        "n_simulations": len(g),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print(f"\n{'=' * 78}")
    if len(best):
        b = best.iloc[0]
        print(f"  可实现的最好结果: {b['variant']} / {b['signal']}")
        print(f"    做T年化 {b['t_annual']:+.2%}  总年化 "
              f"{b['annual_return']:.2%}  vs 纯持有 {bh:.2%}")
    if len(la_best):
        lb = la_best.iloc[0]
        print(f"  前视天花板: 做T年化 {lb['t_annual']:+.2%}")
    print(f"\n  结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
