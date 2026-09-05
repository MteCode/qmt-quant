"""信号衰减诊断 —— IC 到底能撑多少天。

## 为什么需要这个

回测出现了一个直接矛盾：

    IC  +0.0296  t(NW) +7.66     排序明明有预测力
    关掉风控后的回测            20 个配置全亏，回撤 29%~60%

IC 是按 Alpha158 的默认标签算的，那是**约 2 日**的前瞻收益。
而策略每 20 日调一次仓。如果信号的预测力在两三天内就衰减完，
两个数字就不矛盾了：IC 是真的，只是持有 20 天时它早已消失。

本脚本直接测这件事：同一份分数，对 1/2/3/5/10/20/40 日前瞻收益
分别算 IC，看衰减曲线。

## 怎么读结果

- **IC 在 20 日仍显著为正** —— 衰减不是原因，问题在别处
  （成本、Top-K 的尾部行为、或组合构建）
- **IC 几天内归零** —— 调仓周期必须缩短到信号还活着的窗口内，
  或者换一个更长周期的标签重新训练

后者才是真问题：训练标签的期限决定了模型学的是什么。
用 2 日标签训出来的模型拿去做 20 日调仓，是把短周期信号
硬套在长周期决策上。

用法::

    python strategies/lgb_agents_ppo/ic_decay.py
    python strategies/lgb_agents_ppo/ic_decay.py --horizons 1 5 20 60
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import paths  # noqa: E402

DEFAULT_HORIZONS = [1, 2, 3, 5, 10, 20, 40, 60]


def newey_west_t(ic, lag: int) -> float:
    """重叠窗口的 t 值修正。

    N 日前瞻收益在相邻交易日之间高度重叠 —— 今天和明天的
    20 日收益共享 19 天。不修正会把 t 值放大数倍。
    """
    import numpy as np

    x = ic.values - ic.mean()
    n = len(x)
    if n < 3:
        return 0.0
    var = float(x @ x) / n
    for k in range(1, min(max(lag - 1, 1), n - 1) + 1):
        var += 2.0 * (1 - k / lag) * float(x[k:] @ x[:-k]) / n
    return float(ic.mean() / (var / n) ** 0.5) if var > 0 else 0.0


def forward_returns(prices, h: int):
    """h 日前瞻收益。用后复权价 —— 算收益率必须复权，
    否则除权日会凭空冒出一个几个点的假跌幅。"""
    return prices.shift(-h) / prices - 1.0


def main() -> int:
    p = argparse.ArgumentParser(description="信号衰减诊断")
    p.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    p.add_argument("--min-names", type=int, default=30,
                    help="截面至少多少只才计入")
    args = p.parse_args()

    import numpy as np
    import pandas as pd

    from qmtquant.config import get_config

    cfg = get_config()
    store = Path(cfg.data.store_dir)

    scores = pd.read_parquet(paths.SCREEN_SCORES).sort_index()
    print("=" * 66)
    print("信号衰减诊断")
    print("=" * 66)
    print(f"分数面板 {scores.shape[0]} 期 x {scores.shape[1]} 只")
    print(f"区间     {scores.index[0].date()} ~ {scores.index[-1].date()}")

    # 后复权价：算收益率必须复权
    import sqlite3
    db = store / "market.db"
    if not db.exists():
        print("缺少 data/market.db，先跑 scripts/build_database.py")
        return 1
    conn = sqlite3.connect(str(db))
    df = pd.read_sql("SELECT date, vt_symbol, close FROM daily_bar", conn)
    conn.close()
    prices = df.pivot(index="date", columns="vt_symbol", values="close")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    shared_d = scores.index.intersection(prices.index)
    shared_s = scores.columns.intersection(prices.columns)
    sc = scores.loc[shared_d, shared_s]
    px = prices.loc[shared_d, shared_s]
    print(f"对齐后     {len(shared_d)} 期 x {len(shared_s)} 只\n")

    print(f"{'前瞻天数':>8s}{'IC均值':>10s}{'IC标准差':>10s}{'ICIR':>8s}"
          f"{'t(NW)':>9s}{'IC>0占比':>10s}{'多空年化':>10s}")
    print("-" * 66)

    rows = []
    for h in args.horizons:
        fwd = forward_returns(px, h)
        ics, spreads = [], []
        for d in sc.index:
            s = sc.loc[d].dropna()
            r = fwd.loc[d].dropna()
            common = s.index.intersection(r.index)
            if len(common) < args.min_names:
                continue
            ics.append(s[common].corr(r[common], method="spearman"))
            # 多空价差：前 20% 减后 20%，衡量可交易的那部分收益
            k = max(int(len(common) * 0.2), 1)
            top = s[common].nlargest(k).index
            bot = s[common].nsmallest(k).index
            spreads.append(r[top].mean() - r[bot].mean())

        ic = pd.Series(ics).dropna()
        if len(ic) < 10:
            print(f"{h:>8d}       样本不足")
            continue
        sp = pd.Series(spreads).dropna()
        # 单期收益是 h 日的，年化按不重叠的期数折算
        ann = (1 + sp.mean()) ** (244 / h) - 1 if sp.mean() > -1 else -1.0
        t = newey_west_t(ic, h)
        rows.append({"h": h, "ic": ic.mean(), "t": t,
                     "spread_ann": ann})
        print(f"{h:>8d}{ic.mean():>+10.4f}{ic.std():>10.4f}"
              f"{ic.mean() / ic.std():>+8.3f}{t:>+9.2f}"
              f"{(ic > 0).mean():>10.1%}{ann:>+10.2%}")

    print("-" * 66)
    if not rows:
        return 1

    # ---- 分档收益：IC 好不代表最顶端那一小撮好
    #
    # IC 是全截面的秩相关，衡量的是「整体排序对不对」。
    # 但策略只买最高的 15~30 只 —— 在 1800 只里那是**顶端 1%**。
    # 排序整体单调，不保证极端尾部也单调；A 股尤其常见
    # 「最高分那批是最妖的票」：涨得猛也跌得狠。
    reb = paths.load_params().get("rebalance_days", 20)
    fwd = forward_returns(px, reb)
    buckets = [("Top 15只", "n", 15), ("Top 30只", "n", 30),
               ("Top 50只", "n", 50), ("Top 1%", "p", 0.01),
               ("Top 5%", "p", 0.05), ("Top 10%", "p", 0.10),
               ("Top 20%", "p", 0.20), ("第2个20%", "q", 1),
               ("第3个20%", "q", 2), ("第4个20%", "q", 3),
               ("Bottom 20%", "q", 4), ("全市场等权", "all", 0)]

    print(f"\n\n{reb} 日前瞻收益按分数分档（年化）")
    print("-" * 66)
    print(f"{'档位':<14s}{'年化收益':>11s}{'单期均值':>11s}"
          f"{'单期波动':>11s}{'胜率':>9s}")
    print("-" * 66)

    res = {}
    for label, kind, arg in buckets:
        vals = []
        for d in sc.index:
            s = sc.loc[d].dropna()
            r = fwd.loc[d].dropna()
            common = s.index.intersection(r.index)
            if len(common) < args.min_names:
                continue
            s2, r2 = s[common], r[common]
            if kind == "all":
                vals.append(r2.mean())
                continue
            if kind == "n":
                sel = s2.nlargest(min(arg, len(s2))).index
            elif kind == "p":
                sel = s2.nlargest(max(int(len(s2) * arg), 1)).index
            else:   # 五等分，arg=1 是第二档
                ranks = s2.rank(pct=True, ascending=False)
                sel = s2.index[(ranks > arg * 0.2) & (ranks <= (arg + 1) * 0.2)]
            if len(sel) == 0:
                continue
            vals.append(r2[sel].mean())
        v = pd.Series(vals).dropna()
        if v.empty:
            continue
        ann = (1 + v.mean()) ** (244 / reb) - 1 if v.mean() > -1 else -1.0
        res[label] = ann
        print(f"{label:<14s}{ann:>+11.2%}{v.mean():>+11.3%}"
              f"{v.std():>11.3%}{(v > 0).mean():>9.1%}")

    print("-" * 66)
    mkt = res.get("全市场等权")
    top_n = res.get("Top 15只")
    top20 = res.get("Top 20%")
    if mkt is not None and top_n is not None and top20 is not None:
        print(f"\nTop 20% 超额 {top20 - mkt:+.2%}   "
              f"Top 15 只超额 {top_n - mkt:+.2%}")
        if top20 > mkt and top_n <= mkt:
            print("\n**排序在宽档位上有效，最顶端那一小撮却不行。**")
            print("IC 为正而只做多前 15 只亏钱，原因就在这里 ——")
            print("模型把最高分给了最极端的票，它们波动大、回撤深，")
            print("平均收益反而不如宽一点的档位。")
            print("\n对策是放宽持仓档位（买 Top 10% 里的一批而非 Top 15），")
            print("或在选股时对波动率/换手率做二次约束。")

    sig = [r for r in rows if abs(r["t"]) >= 2 and r["ic"] > 0]
    if not sig:
        print("\n所有期限的 IC 都不显著为正 —— 问题不在衰减，在信号本身")
        return 0

    last = max(r["h"] for r in sig)
    print(f"\nIC 显著为正的最长期限：{last} 日")

    reb = paths.load_params().get("rebalance_days", 20)
    if last < reb:
        print(f"当前调仓周期 {reb} 日 > {last} 日 —— **信号在调仓前就死了**")
        print("\n这解释了「IC 为正但回测亏钱」：模型学的是短周期信号，")
        print("拿去做长周期决策，持有期里预测力早已消失。")
        print("\n两条路：")
        print(f"  1. 调仓周期缩到 {last} 日以内（换手与成本会上升）")
        print(f"  2. 用 {reb} 日前瞻收益重新训练（模型学的东西才对得上）")
        print("     第 2 条更根本 —— 标签期限决定模型学什么")
    else:
        print(f"当前调仓周期 {reb} 日 <= {last} 日 —— 衰减不是主因，")
        print("问题在成本、Top-K 尾部行为或组合构建，需另行排查")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
