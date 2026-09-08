"""做 T 深度归因 —— 赚钱的日子为什么赚，与市场、行业是什么关系。

回答四个问题：

1. **少数几天赚了大钱，其余亏损 —— 那几天有什么不同？**
   如果赚钱集中在极少数日子，且那些日子有共同特征（比如大波动），
   策略的价值就取决于「能否识别这种日子」，而不是「参数调得好」。
   如果找不到共同特征，那就是运气。

2. **市场门控到底有没有用？**
   门控会同时减少亏损和减少交易。要区分「找到了更好的交易」和
   「只是少交易了」—— 看**单笔亏损**：如果门控后单笔亏损没变小，
   那就只是交易少了，不是信号变好了。

3. **这只票和大盘什么关系？** beta、相关性、独立行情的比例。

4. **和行业什么关系？** 同行业标的的共同走势解释了多少。

## 一个方法论要点

「赚钱的日子有什么特征」这个问题极易得出虚假结论：事后挑出赚钱的
日子，总能找到它们的某个共同点。所以这里的做法是**先定义候选特征，
再看它们在赚钱日与亏钱日的分布差异**，并给出两组的重叠程度 ——
分布高度重叠就说明这个特征区分不了，哪怕均值差异看着显著。
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

OUT_DIR = ROOT / "models" / "t0_analysis"
MARKET_FILE = ROOT / "data" / "market_state_1m.parquet"


def _engine():
    spec = importlib.util.spec_from_file_location(
        "t0div", ROOT / "scripts" / "optimize_t0_divergence.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------- 1. 赚钱日归因

def analyze_winning_days(daily: pd.DataFrame, d: pd.DataFrame,
                         mkt_daily: pd.DataFrame) -> dict:
    """赚钱的日子 vs 亏钱的日子，特征分布对比。"""
    print("\n" + "=" * 78)
    print("  一、做 T 的收益分布 —— 是稳定盈利还是靠几天？")
    print("=" * 78)

    pnl = daily["day_t_pnl"].astype(float)
    n = len(pnl)
    win = pnl > 0
    print(f"  {n} 个交易日：盈利 {win.sum()} 天（{win.mean():.1%}），"
          f"亏损 {(~win).sum()} 天")
    print(f"  做 T 日收益：中位 {pnl.median():+,.0f}  "
          f"均值 {pnl.mean():+,.0f}  合计 {pnl.sum():+,.0f} 元")

    # 收益集中度：前 N 天贡献了多少
    srt = pnl.sort_values(ascending=False)
    total = pnl.sum()
    print(f"\n  收益集中度：")
    for k in (3, 5, 7, 10, 20):
        if k > n:
            break
        top = srt.head(k).sum()
        share = (top / abs(total) * 100) if total != 0 else float("nan")
        print(f"    最赚的 {k:>2} 天合计 {top:>+10,.0f} 元"
              f"（占总盈亏 {share:>7.0f}%）")

    pos_sum = pnl[pnl > 0].sum()
    neg_sum = pnl[pnl <= 0].sum()
    print(f"\n    盈利日合计 {pos_sum:+,.0f}，亏损日合计 {neg_sum:+,.0f}")
    if pos_sum > 0:
        print(f"    最赚的 7 天占全部盈利的 "
              f"{srt.head(7).clip(lower=0).sum() / pos_sum:.0%}")

    # ---- 特征对比 ----
    print(f"\n  盈利日 vs 亏损日的特征分布：")
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date")

    # 个股当日特征
    g = d.groupby("day")
    feat = pd.DataFrame({
        "振幅": (g["high"].max() - g["low"].min()) / g["close"].last(),
        "日涨幅": g["close"].last() / g["open"].first() - 1,
        "成交量": g["volume"].sum(),
        "分钟波动": g["close"].apply(lambda x: x.pct_change().std()),
    })
    feat.index = pd.to_datetime(feat.index)
    if mkt_daily is not None:
        feat = feat.join(mkt_daily, how="left")

    j = daily.join(feat, how="inner")
    j["win"] = j["day_t_pnl"] > 0

    print(f"  {'特征':<14} {'盈利日中位':>12} {'亏损日中位':>12} "
          f"{'差异':>10} {'分布重叠':>10}")
    print("  " + "-" * 62)
    rows = []
    for c in feat.columns:
        if c not in j.columns:
            continue
        a, b = j.loc[j["win"], c].dropna(), j.loc[~j["win"], c].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        ma, mb = a.median(), b.median()
        # 重叠度：两组分布的共同覆盖范围占比。高重叠 = 区分不了
        lo = max(a.quantile(.25), b.quantile(.25))
        hi = min(a.quantile(.75), b.quantile(.75))
        span = max(a.quantile(.75), b.quantile(.75)) - \
            min(a.quantile(.25), b.quantile(.25))
        overlap = max(0.0, (hi - lo) / span) if span > 0 else 1.0
        fmt = ".4f" if abs(ma) < 1 else ",.0f"
        print(f"  {c:<14} {format(ma, fmt):>12} {format(mb, fmt):>12} "
              f"{format(ma - mb, fmt):>10} {overlap:>9.0%}")
        rows.append({"feature": c, "win_median": float(ma),
                     "lose_median": float(mb), "overlap": float(overlap)})

    print(f"\n  判读：分布重叠 >60% 说明该特征区分不了盈亏日 ——")
    print(f"  哪怕中位数看着有差异，实际用它做过滤也筛不出赚钱的日子。")

    return {"n_days": int(n), "win_days": int(win.sum()),
            "win_rate": float(win.mean()),
            "total_t_pnl": float(total),
            "top7_share_of_gains": float(
                srt.head(7).clip(lower=0).sum() / pos_sum)
            if pos_sum > 0 else None,
            "features": rows}


# ------------------------------------------------------- 2. 门控是否有用

def analyze_gating() -> dict:
    """从对照实验数据判断门控的真实贡献。"""
    p = ROOT / "models" / "t0_market" / "gating_experiment.csv"
    if not p.exists():
        print("\n（未找到门控实验数据，跳过）")
        return {}

    print("\n" + "=" * 78)
    print("  二、市场门控有没有用？—— 配对差异分析")
    print("=" * 78)

    g = pd.read_csv(p)
    base = g[g["gate"] == "无门控"].set_index("param_id")

    print(f"  {'门控':<18} {'做T年化差异':>12} {'变好占比':>9} "
          f"{'交易量比':>9} {'单笔亏损':>10} {'单笔亏损比':>11}")
    print("  " + "-" * 72)

    # 单笔亏损 = 做T总盈亏 / 交易笔数。门控若真的筛出更好的交易，
    # 这个数应该变小；若只是少交易了，它不变
    base_per = (base["t_annual"] / base["n_trades"].replace(0, np.nan))
    rows = []
    for gname in g["gate"].unique():
        if gname == "无门控":
            print(f"  {'无门控（基准）':<18} {'—':>12} {'—':>9} "
                  f"{'1.00x':>9} {base_per.median():>10.5f} {'1.00x':>11}")
            continue
        s = g[g["gate"] == gname].set_index("param_id")
        common = base.index.intersection(s.index)
        if not len(common):
            continue
        delta = s.loc[common, "t_annual"] - base.loc[common, "t_annual"]
        tr = (s.loc[common, "n_trades"]
              / base.loc[common, "n_trades"].replace(0, np.nan))
        per = (s.loc[common, "t_annual"]
               / s.loc[common, "n_trades"].replace(0, np.nan))
        per_ratio = per / base_per.loc[common].replace(0, np.nan)
        rows.append({"gate": gname,
                     "delta_median": float(delta.median()),
                     "pct_improved": float((delta > 0).mean()),
                     "trade_ratio": float(tr.median()),
                     "per_trade": float(per.median()),
                     "per_trade_ratio": float(per_ratio.median())})
        print(f"  {gname:<18} {delta.median():>+12.2%} "
              f"{(delta > 0).mean():>9.1%} {tr.median():>8.2f}x "
              f"{per.median():>10.5f} {per_ratio.median():>10.2f}x")

    print(f"\n  关键判读：")
    print(f"  「做T年化差异」为正只说明亏得少了，可能只是因为交易少了。")
    print(f"  要看**单笔亏损比** —— 接近 1.00x 说明每笔交易的质量没变，")
    print(f"  门控只是减少了交易次数；显著小于 1 才说明筛出了更好的交易。")

    if rows:
        best = min(rows, key=lambda r: r["per_trade_ratio"])
        worst_overlap = all(abs(r["per_trade_ratio"] - 1) < 0.5 for r in rows)
        print(f"\n  单笔亏损改善最好的: {best['gate']} "
              f"({best['per_trade_ratio']:.2f}x)")
        if worst_overlap:
            print(f"  → 所有门控的单笔亏损比都在 1.0 附近，"
                  f"说明门控**没有改善交易质量**，只是少做了。")
    return {"paired": rows}


# ------------------------------------------------------- 3. 与市场的关系

def analyze_market_relation(d: pd.DataFrame,
                            mkt_daily: pd.DataFrame) -> dict:
    print("\n" + "=" * 78)
    print("  三、这只票和大盘什么关系？")
    print("=" * 78)
    if mkt_daily is None or "mkt_day_ret" not in mkt_daily.columns:
        print("  无市场数据，跳过")
        return {}

    g = d.groupby("day")
    stock = pd.DataFrame({
        "stock_ret": g["close"].last() / g["open"].first() - 1,
        "amp": (g["high"].max() - g["low"].min()) / g["close"].last(),
    })
    stock.index = pd.to_datetime(stock.index)
    j = stock.join(mkt_daily, how="inner").dropna(
        subset=["stock_ret", "mkt_day_ret"])
    if len(j) < 20:
        print("  重叠交易日太少")
        return {}

    x, y = j["mkt_day_ret"].values, j["stock_ret"].values
    beta = float(np.polyfit(x, y, 1)[0])
    corr = float(np.corrcoef(x, y)[0, 1])
    r2 = corr ** 2

    print(f"  重叠 {len(j)} 个交易日（日频口径）")
    print(f"  相关系数 {corr:+.3f}   R² {r2:.3f}   beta {beta:+.2f}")
    print(f"  → 大盘只解释了个股 {r2:.0%} 的日涨跌，"
          f"{1-r2:.0%} 是个股自己的行情")

    same = float(np.mean(np.sign(x) == np.sign(y)))
    print(f"  同向天数占比 {same:.0%}")

    # 分市场状态看个股表现
    print(f"\n  不同大盘状态下，个股的表现：")
    print(f"  {'大盘状态':<16} {'天数':>5} {'个股涨幅中位':>13} "
          f"{'个股振幅中位':>13}")
    print("  " + "-" * 52)
    q = j["mkt_day_ret"].quantile([0.33, 0.67])
    buckets = [("大盘下跌", j[j["mkt_day_ret"] <= q.iloc[0]]),
               ("大盘震荡", j[(j["mkt_day_ret"] > q.iloc[0])
                              & (j["mkt_day_ret"] < q.iloc[1])]),
               ("大盘上涨", j[j["mkt_day_ret"] >= q.iloc[1]])]
    bstats = []
    for name, sub in buckets:
        if sub.empty:
            continue
        print(f"  {name:<16} {len(sub):>5} "
              f"{sub['stock_ret'].median():>+13.2%} "
              f"{sub['amp'].median():>13.2%}")
        bstats.append({"bucket": name, "n": len(sub),
                       "stock_ret_median": float(sub["stock_ret"].median()),
                       "amp_median": float(sub["amp"].median())})

    return {"corr": corr, "r2": r2, "beta": beta,
            "same_direction_pct": same, "buckets": bstats,
            "n_days": len(j)}


# ------------------------------------------------------- 4. 与行业的关系

def analyze_industry(symbol: str) -> dict:
    print("\n" + "=" * 78)
    print("  四、和行业什么关系？")
    print("=" * 78)

    from qmtquant.config import get_config
    cfg = get_config()
    ind_file = Path(cfg.data.store_dir) / "universe" / "industry.parquet"
    if not ind_file.exists():
        print(f"  无行业数据（{ind_file} 不存在），跳过")
        return {}

    try:
        ind = pd.read_parquet(ind_file)
    except (OSError, ValueError) as e:
        print(f"  行业数据读取失败: {e}")
        return {}

    vt = f"{symbol}.SSE"
    row = ind[ind["vt_symbol"] == vt]
    if row.empty:
        print(f"  {vt} 不在行业表里")
        return {}

    my_ind = row.iloc[0].get("industry", "")
    name = row.iloc[0].get("name", "")
    peers = ind[ind["industry"] == my_ind]["vt_symbol"].tolist()
    peers = [p for p in peers if p != vt]
    print(f"  {vt} {name}  行业：{my_ind}")
    print(f"  同行业标的 {len(peers)} 只")

    if len(peers) < 3:
        print("  同行业标的太少，无法做行业因子分析")
        return {"industry": my_ind, "n_peers": len(peers)}

    # 用日线算行业等权指数
    base = Path(cfg.data.store_dir) / "clean" / "1d"
    series = {}
    for p in [vt] + peers[:40]:
        try:
            code, ex = p.split(".")
        except ValueError:
            continue
        f = base / ex / f"{code}.parquet"
        if not f.exists():
            continue
        try:
            s = pd.read_parquet(f, columns=["close"])["close"]
        except (OSError, ValueError, KeyError):
            continue
        if len(s) > 100:
            series[p] = s.pct_change()

    if vt not in series or len(series) < 4:
        print(f"  可用日线数据太少（{len(series)} 只），跳过")
        return {"industry": my_ind, "n_peers": len(peers)}

    R = pd.DataFrame(series).dropna(how="all")
    peer_cols = [c for c in R.columns if c != vt]
    ind_ret = R[peer_cols].mean(axis=1)
    j = pd.DataFrame({"stock": R[vt], "ind": ind_ret}).dropna()
    if len(j) < 50:
        print("  重叠日数太少")
        return {"industry": my_ind, "n_peers": len(peers)}

    corr = float(j["stock"].corr(j["ind"]))
    beta = float(np.polyfit(j["ind"].values, j["stock"].values, 1)[0])
    print(f"  用 {len(peer_cols)} 只同业股构建等权行业指数，"
          f"重叠 {len(j)} 个交易日")
    print(f"  与行业相关系数 {corr:+.3f}   R² {corr**2:.3f}   "
          f"beta {beta:+.2f}")
    print(f"  → 行业解释了个股 {corr**2:.0%} 的日涨跌")

    return {"industry": my_ind, "name": name, "n_peers": len(peers),
            "n_used": len(peer_cols), "corr": corr, "r2": corr ** 2,
            "beta": beta, "n_days": len(j)}


# ------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description="做 T 深度归因分析")
    p.add_argument("--symbol", default="600711")
    p.add_argument("--exchange", default="SSE")
    p.add_argument("--daily",
                   default=str(ROOT / "models" / "t0_divergence"
                               / "600711_best_daily.csv"),
                   help="最优方案的每日结果 CSV")
    p.add_argument("--output", default=str(OUT_DIR))
    args = p.parse_args()

    m = _engine()
    df = m.load_bars(args.symbol, args.exchange)
    d = m.prep(df)

    mkt_daily = None
    if MARKET_FILE.exists():
        mkt = pd.read_parquet(MARKET_FILE)
        mkt = mkt.copy()
        mkt["day"] = mkt.index.normalize()
        mkt_daily = mkt.groupby("day").agg(
            mkt_day_ret=("mkt_day_ret", "last"),
            mkt_breadth=("mkt_breadth", "mean"),
            mkt_vol_z=("mkt_vol_z", "mean"))
        mkt_daily.index = pd.to_datetime(mkt_daily.index)

    print("=" * 78)
    print(f"  做 T 深度归因 —— {args.symbol}.{args.exchange}")
    print("=" * 78)

    out = {"symbol": f"{args.symbol}.{args.exchange}",
           "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

    dp = Path(args.daily)
    if dp.exists():
        daily = pd.read_csv(dp)
        out["winning_days"] = analyze_winning_days(daily, d, mkt_daily)
    else:
        print(f"\n（找不到 {dp}，跳过收益分布分析）")

    out["gating"] = analyze_gating()
    out["market"] = analyze_market_relation(d, mkt_daily)
    out["industry"] = analyze_industry(args.symbol)

    o = Path(args.output)
    o.mkdir(parents=True, exist_ok=True)
    (o / "deep_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\n  分析结果已保存: {o / 'deep_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
