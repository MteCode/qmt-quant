"""市场门控的单票做 T 优化 —— 只在大盘配合时做 T。

## 想法

之前的做 T 研究只看个股自身的量价背离，结论是负收益。一个未测过的
可能：**做 T 的成败可能取决于大盘状态**。

同样是「价跌量增」，大盘普跌时它多半是跟随下杀（抄底会接飞刀），
大盘企稳时它才可能是个股的错杀（抄底才有意义）。个股信号不带这个
上下文，等于把两种完全不同的情形当成一种。

所以在原有信号之外加一层市场门控：

- `mkt_breadth`：上涨家数占比。普跌时不抄底
- `mkt_day_ret`：大盘当日累计涨幅。大盘在跌就不做多头方向的 T
- `mkt_vol_z`：全市场量能。极度缩量时日内波动小，做 T 没空间

市场状态由 `build_market_state.py` 从全市场 1m 数据算出，无前视。

## 目标：最高收益，不设回撤约束

调用方明确要求不考虑回撤。但报告里仍然会给出回撤 —— 不看不等于
不存在，把它藏起来是另一种误导。

## 输出

    models/t0_market/
        grid.csv        全部组合
        summary.json    最优方案 + 样本外验证 + 对照基准
        best_daily.csv  最优方案的每日净值
        best_trades.csv 最优方案的成交明细
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

OUT_DIR = ROOT / "models" / "t0_market"
MARKET_FILE = ROOT / "data" / "market_state_1m.parquet"


def _load_engine():
    """复用 optimize_t0_divergence 的模拟器 —— T+1 约束、成本模型、
    量价背离信号都在那边实现好了，重写会引入不一致。"""
    spec = importlib.util.spec_from_file_location(
        "t0div", ROOT / "scripts" / "optimize_t0_divergence.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def attach_market(d: pd.DataFrame) -> pd.DataFrame:
    """把市场状态并到个股的分钟索引上。

    用 reindex 而非 merge_asof：两边都是分钟级且对齐，缺失的时点
    （市场数据没覆盖到）填中性值而不是前向填充 —— 前向填充会把
    几小时前的市场状态当成当前状态用。
    """
    if not MARKET_FILE.exists():
        raise FileNotFoundError(
            f"市场状态文件不存在: {MARKET_FILE}\n"
            f"先运行 python scripts/build_market_state.py")
    mkt = pd.read_parquet(MARKET_FILE)
    cols = ["mkt_ret", "mkt_breadth", "mkt_vol_z", "mkt_day_ret"]
    aligned = mkt[cols].reindex(d.index)

    # 缺失时点填中性值：宽度 0.5（分化）、其余 0。
    # 这样门控条件在无市场数据时不会误判为「极端行情」
    aligned["mkt_breadth"] = aligned["mkt_breadth"].fillna(0.5)
    for c in ("mkt_ret", "mkt_vol_z", "mkt_day_ret"):
        aligned[c] = aligned[c].fillna(0.0)

    out = d.copy()
    for c in cols:
        out[c] = aligned[c].values.astype(np.float64)
    cover = float(mkt[cols[0]].reindex(d.index).notna().mean())
    out.attrs["market_coverage"] = cover
    return out


def simulate_gated(m, d, base_value, cash_value, signal, thr, vol_thr,
                   take_profit, stop_loss, max_trades, entry_min, exit_min,
                   hold_max, reverse,
                   min_breadth, min_day_ret, min_vol_z):
    """带市场门控的模拟。

    门控只影响**开仓**，不影响平仓 —— 已开的仓无论大盘怎样都要按
    止盈止损与尾盘规则了结。市场转坏时锁死出场会让风险敞口失控。
    """
    # 门控通过屏蔽信号实现：把不满足市场条件的时点的信号强度置为
    # 无法触发的值，这样不必改动模拟器内部逻辑
    dd = d.copy()
    gate = ((dd["mkt_breadth"] >= min_breadth)
            & (dd["mkt_day_ret"] >= min_day_ret)
            & (dd["mkt_vol_z"] >= min_vol_z))

    # 各信号列在不满足门控时置中性值（模拟器据此判定「无信号」）
    for col, neutral in (("pv_corr", 0.0), ("obv_div", 0.0),
                         ("vol_z", -99.0), ("ret_n", 0.0)):
        if col in dd.columns:
            dd.loc[~gate, col] = neutral

    r = m.simulate(dd, base_value, cash_value, signal, thr, vol_thr,
                   take_profit, stop_loss, max_trades, entry_min, exit_min,
                   hold_max, reverse)
    if "error" not in r:
        r["gate_pass_pct"] = round(float(gate.mean()), 4)
    return r


def main() -> int:
    p = argparse.ArgumentParser(description="市场门控的单票做 T 优化")
    p.add_argument("--symbol", default="600711")
    p.add_argument("--exchange", default="SSE")
    p.add_argument("--base", type=float, default=100000.0, help="底仓市值")
    p.add_argument("--cash", type=float, default=100000.0, help="可用现金")
    p.add_argument("--entry-time", default="09:35")
    p.add_argument("--exit-time", default="14:50")
    p.add_argument("--output", default=str(OUT_DIR))
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    def _tm(s):
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    m = _load_engine()
    em, xm = _tm(args.entry_time), _tm(args.exit_time)
    init = args.base + args.cash

    print("=" * 78)
    print(f"  市场门控做 T 优化 —— {args.symbol}.{args.exchange}")
    print(f"  底仓 {args.base:,.0f} + 现金 {args.cash:,.0f} "
          f"= 总资产 {init:,.0f} 元")
    print(f"  目标：最高收益（不设回撤约束）")
    print("=" * 78)

    df = m.load_bars(args.symbol, args.exchange)
    d = m.prep(df)
    d = attach_market(d)
    days = pd.DatetimeIndex(d["day"].unique()).sort_values()
    print(f"  数据 {len(df):,} bar，{len(days)} 交易日，"
          f"{days[0].date()} ~ {days[-1].date()}")
    print(f"  市场状态覆盖率 {d.attrs['market_coverage']:.1%}")
    print(f"  市场指标: breadth 中位 {d['mkt_breadth'].median():.3f}，"
          f"day_ret 中位 {d['mkt_day_ret'].median():+.4f}，"
          f"vol_z 中位 {d['mkt_vol_z'].median():+.2f}")

    # ---- 搜索空间 ----
    signals = ["corr", "obv", "pulse"]
    # 门控档位：从「不设限」到「只在明显普涨时做」
    breadths = [0.0, 0.30, 0.40]        # 0 = 不看宽度
    day_rets = [-1.0, -0.005, 0.0]      # -1 = 不看大盘涨跌
    vol_zs = [-99.0, -0.8, -0.3]        # -99 = 不看量能

    combos = []
    for sig in signals:
        thrs = ([0.1, 0.2, 0.3] if sig in ("corr", "obv")
                else [0.003, 0.005, 0.01])
        for c in itertools.product(thrs, [1.0, 2.0], [0.006, 0.010, 0.015],
                                   [0.008, 0.012], [1, 2], [30, 60],
                                   [False, True], breadths, day_rets, vol_zs):
            combos.append((sig,) + c)

    print(f"\n  搜索 {len(combos)} 组"
          f"（信号 {len(signals)} 类 × 策略参数 × 门控 "
          f"{len(breadths)}×{len(day_rets)}×{len(vol_zs)}）")

    rows = []
    t0 = time.time()
    for i, (sig, thr, vt, tp, sl, mt, hd, rv, mb, mdr, mvz) in \
            enumerate(combos, 1):
        r = simulate_gated(m, d, args.base, args.cash, sig, thr, vt, tp, sl,
                           mt, em, xm, hd, rv, mb, mdr, mvz)
        if "error" in r or r["n_trades"] == 0:
            continue
        r.pop("_daily", None)
        r.pop("_trades", None)
        r.update({"signal": sig, "thr": thr, "vol_thr": vt,
                  "take_profit": tp, "stop_loss": sl, "max_trades": mt,
                  "hold_max": hd, "reverse": rv,
                  "min_breadth": mb, "min_day_ret": mdr, "min_vol_z": mvz,
                  "gated": (mb > 0 or mdr > -1 or mvz > -99)})
        rows.append(r)
        if i % 1000 == 0:
            print(f"    {i}/{len(combos)} ({time.time()-t0:.0f}s)",
                  flush=True)

    if not rows:
        print("无有效结果")
        return 1

    g = pd.DataFrame(rows).sort_values("annual_return", ascending=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "grid.csv", index=False)

    bh = float(g["buyhold_annual"].iloc[0])
    npos = int((g["t_annual"] > 0).sum())

    print(f"\n{'=' * 78}")
    print(f"  搜索完成（{len(g)} 组有交易，耗时 {time.time()-t0:.0f}s）")
    print(f"  基准 —— 纯持有不做 T: 年化 {bh:.2%}")
    print(f"  做 T 为正的组合: {npos}/{len(g)} ({npos/len(g):.1%})")
    print(f"{'=' * 78}")

    # 门控 vs 不门控 —— 这是本轮要回答的核心问题
    gated = g[g["gated"]]
    plain = g[~g["gated"]]
    print(f"\n  门控是否有用：")
    print(f"    {'':16} {'组数':>6} {'做T年化均值':>12} {'最好':>10} "
          f"{'为正占比':>9}")
    for name, sub in (("不带门控", plain), ("带门控", gated)):
        if len(sub):
            print(f"    {name:16} {len(sub):>6} "
                  f"{sub['t_annual'].mean():>12.2%} "
                  f"{sub['t_annual'].max():>10.2%} "
                  f"{(sub['t_annual'] > 0).mean():>9.1%}")

    print(f"\n  收益最高的 {args.top} 组：")
    print(f"  {'#':<3} {'信号':<6} {'门控(宽度/涨跌/量能)':<22} "
          f"{'年化':>9} {'做T年化':>9} {'回撤':>9} {'夏普':>7} "
          f"{'交易':>6} {'覆盖':>7}")
    print("  " + "-" * 100)
    for i, (_, r) in enumerate(g.head(args.top).iterrows(), 1):
        gate = (f"{r['min_breadth']:.2f}/"
                f"{'不限' if r['min_day_ret'] <= -1 else f'{r[chr(39)+chr(39)] if False else r.min_day_ret:+.3f}'}/"
                f"{'不限' if r['min_vol_z'] <= -98 else f'{r.min_vol_z:+.1f}'}")
        print(f"  {i:<3} {r['signal']:<6} {gate:<22} "
              f"{r['annual_return']:>9.2%} {r['t_annual']:>9.2%} "
              f"{r['max_drawdown']:>9.2%} {r['sharpe']:>7.2f} "
              f"{int(r['n_trades']):>6} {r['coverage']:>7.1%}")

    # ---- 样本外验证 ----
    print(f"\n{'=' * 78}")
    print("  样本外验证 —— 前半段选参数，后半段检验")
    print(f"{'=' * 78}")
    mid = days[len(days) // 2]
    d_in, d_out = d[d["day"] < mid], d[d["day"] >= mid]
    print(f"  样本内 {days[0].date()} ~ {mid.date()} ({len(days)//2} 天)")
    print(f"  样本外 {mid.date()} ~ {days[-1].date()} "
          f"({len(days)-len(days)//2} 天)")

    ins = []
    for sig, thr, vt, tp, sl, mt, hd, rv, mb, mdr, mvz in combos:
        r = simulate_gated(m, d_in, args.base, args.cash, sig, thr, vt, tp,
                           sl, mt, em, xm, hd, rv, mb, mdr, mvz)
        if "error" in r or r["n_trades"] == 0:
            continue
        ins.append((r["t_annual"], sig, thr, vt, tp, sl, mt, hd, rv,
                    mb, mdr, mvz))
    ins.sort(reverse=True)

    wf = []
    print(f"\n  {'#':<3} {'样本内做T':>10} {'样本外做T':>10}  参数")
    print("  " + "-" * 74)
    for i, t in enumerate(ins[:10], 1):
        tin, sig, thr, vt, tp, sl, mt, hd, rv, mb, mdr, mvz = t
        ro = simulate_gated(m, d_out, args.base, args.cash, sig, thr, vt, tp,
                            sl, mt, em, xm, hd, rv, mb, mdr, mvz)
        tout = ro.get("t_annual", float("nan"))
        wf.append({"rank": i, "in_sample_t_annual": tin,
                   "out_sample_t_annual": tout, "signal": sig,
                   "min_breadth": mb, "min_day_ret": mdr, "min_vol_z": mvz})
        print(f"  {i:<3} {tin:>9.2%} {tout:>10.2%}  {sig} "
              f"门控 {mb:.2f}/{mdr:+.3f}/{mvz:+.1f}")

    surv = sum(1 for w in wf if w["out_sample_t_annual"] > 0)
    print(f"\n  样本内前 10 组中样本外仍为正: {surv}/10")

    # ---- 最优方案明细 ----
    b = g.iloc[0]
    br = simulate_gated(m, d, args.base, args.cash, b["signal"], b["thr"],
                        b["vol_thr"], b["take_profit"], b["stop_loss"],
                        int(b["max_trades"]), em, xm, int(b["hold_max"]),
                        bool(b["reverse"]), b["min_breadth"],
                        b["min_day_ret"], b["min_vol_z"])
    bdl = br.pop("_daily")
    btr = br.pop("_trades")
    bdl.to_csv(out / "best_daily.csv", index=False)
    if not btr.empty:
        btr.to_csv(out / "best_trades.csv", index=False)

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": f"{args.symbol}.{args.exchange}",
        "base_value": args.base, "cash_value": args.cash,
        "date_range": [str(days[0].date()), str(days[-1].date())],
        "n_days": len(days),
        "market_coverage": round(d.attrs["market_coverage"], 4),
        "objective": "最高收益（不设回撤约束）",
        "best_params": {
            "signal": b["signal"], "thr": float(b["thr"]),
            "vol_thr": float(b["vol_thr"]),
            "take_profit": float(b["take_profit"]),
            "stop_loss": float(b["stop_loss"]),
            "max_trades": int(b["max_trades"]),
            "hold_max": int(b["hold_max"]),
            "reverse": bool(b["reverse"]),
            "min_breadth": float(b["min_breadth"]),
            "min_day_ret": float(b["min_day_ret"]),
            "min_vol_z": float(b["min_vol_z"]),
            "entry_time": args.entry_time, "exit_time": args.exit_time,
        },
        "performance": br,
        "buyhold_annual": bh,
        "gating_effect": {
            "plain_mean_t_annual": round(float(plain["t_annual"].mean()), 4)
            if len(plain) else None,
            "gated_mean_t_annual": round(float(gated["t_annual"].mean()), 4)
            if len(gated) else None,
            "plain_best": round(float(plain["t_annual"].max()), 4)
            if len(plain) else None,
            "gated_best": round(float(gated["t_annual"].max()), 4)
            if len(gated) else None,
        },
        "robustness": {
            "n_configs": len(g), "n_positive_t": npos,
            "pct_positive_t": round(npos / len(g), 4),
            "t_annual_mean": round(float(g["t_annual"].mean()), 4),
            "walkforward_survivors": surv,
        },
        "walkforward": wf,
        "cost_model": {"commission": m.COMMISSION, "stamp_tax": m.STAMP_TAX,
                       "slippage": m.SLIPPAGE, "round_trip": m.ROUND_TRIP},
        "top_configs": g.head(args.top).to_dict("records"),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print(f"\n{'=' * 78}")
    print("  最优方案")
    print(f"{'=' * 78}")
    print(f"  信号 {b['signal']}  阈值 {b['thr']}  量能门槛 {b['vol_thr']}")
    print(f"  止盈 {b['take_profit']:.1%}  止损 {b['stop_loss']:.1%}  "
          f"每日 {int(b['max_trades'])} 次  最长持有 {int(b['hold_max'])} 分钟")
    print(f"  市场门控: 宽度≥{b['min_breadth']:.2f}  "
          f"大盘涨幅≥{'不限' if b['min_day_ret'] <= -1 else f'{b.min_day_ret:+.3f}'}  "
          f"量能≥{'不限' if b['min_vol_z'] <= -98 else f'{b.min_vol_z:+.1f}'}")
    print()
    print(f"  总资产 {init:,.0f} → {br['total_return'] * init + init:,.0f} 元")
    print(f"  年化 {br['annual_return']:.2%}"
          f"（其中做T贡献 {br['t_annual']:+.2%}）")
    print(f"  最大回撤 {br['max_drawdown']:.2%}  夏普 {br['sharpe']:.2f}")
    print(f"  交易 {br['n_trades']} 笔，覆盖 {br['coverage']:.1%} 的交易日")
    print(f"  毛 edge {br['gross_edge']:.4%}  成本 {m.ROUND_TRIP:.3%}  "
          f"净 edge {br['net_edge']:+.4%}")
    print(f"\n  对照 —— 纯持有不做 T: 年化 {bh:.2%}")
    print(f"  做 T 相对纯持有: {br['annual_return'] - bh:+.2%}")
    print(f"\n  结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
