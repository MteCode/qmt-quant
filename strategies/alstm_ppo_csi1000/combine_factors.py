"""多因子合成 —— ALSTM + 资金流因子。

## 结论先行：合成没有带来提升

五个候选因子中只有 `dragon_count_20` 通过评估（其余因输入数据的前视偏差
或严重缺失被剔除，见 eval_factors）。加权合成后：

    仅 ALSTM          Sharpe +0.650   回撤 17.06%
    仅龙虎榜           Sharpe +0.145   回撤 16.34%
    ALSTM 8 : 龙虎 2   Sharpe +0.631   回撤 16.94%
    ALSTM 5 : 龙虎 5   Sharpe +0.630   回撤 16.90%

**任何权重组合都没有超过 ALSTM 单独使用。** 且所有含 ALSTM 的配置
在分段检验中都是「前段为正、后段为负」，不具跨期稳定性。

原因不在合成方式，而在 IC 无法转化为组合收益 —— 龙虎榜因子 IC 达
-0.0627（t=-7.22，比 ALSTM 强 3 倍），单独使用却只有 Sharpe 0.145。
这与 ALSTM 的老问题同源：IC 与 Sharpe 的相关性本就只有 0.338。

另一层原因是该因子取值过于离散：6451 只标的中 5432 只的 20 日上榜次数
为 0，截面排名只有 12 个不同取值。用于 Top-50 选股时它无法细分，
实际效果仅是「排除约 1000 只过热标的」。

## 合成方法

各路先转截面百分位排名再加权。三路信号量纲完全不同（ALSTM 是 0~1 的
百分位、龙虎榜是次数），直接加权会被量纲大的一路主导。排名化后量纲
问题消失，且下游 SignalRankStrategy 本就只用相对顺序。

资金流因子 IC 为负（上榜频繁者后续跑输，即「过热则回落」），
合成时取负号统一成「越大越好」。

缺失值填 0.5 而非 0：填 0 等于断言「该股此项排名垫底」，
会把从未上榜的标的全部打成最差 —— 而没上榜恰恰意味着不过热。

用法::

    python strategies/alstm_ppo_csi1000/combine_factors.py
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
from eval_factors import load_money_factors  # noqa: E402
from train_alstm import TEST  # noqa: E402
from train_ensemble import backtest  # noqa: E402

RESULT_JSON = paths.BACKTEST_DIR / "multifactor.json"
COMBINED_SCORES = paths.MODELS_DIR / "multifactor_scores.parquet"

#: 参与合成的资金流因子及方向。-1 表示取负（IC 为负的反转因子）
#: 只保留通过评估的因子。margin_* 因 2023 年后每年仅 24 天数据
#: （覆盖 10%）被剔除 —— 详见 eval_factors.MIN_PERIOD_RATIO
MONEY_FACTORS = {
    "dragon_count_20": -1,
    # 公告因子：方向在看数据前按常识设定，评估结果与之一致
    "ann_reduce_20": -1,      # 减持利空
    "ann_risk_20": -1,        # 诉讼/处罚/问询/质押利空
    "ann_incentive_20": +1,   # 股权激励利好
    "ann_net_score_20": +1,   # 公告净分（利好数 - 利空数）
}

#: 缺失值填的中性排名。仅用于**非计数型**因子 —— 那类缺失确实是「未知」
NEUTRAL = 0.5

#: 计数型因子：缺失的语义是「该事件从未发生」，即计数为 0，而非未知。
#:
#: 这类因子的面板只包含**曾经发生过该事件**的标的，从未发生的根本不在
#: 面板里。若按 NEUTRAL=0.5 填排名，会把「从未减持」排到「减持过」之后 ——
#: 方向完全反了。实测 ann_reduce_20 缺失率 24%，而有值者中位排名 0.556，
#: 填 0.5 使得最干净的那批标的反而垫底。
#:
#: 正确做法是在**排名之前**把原始值补 0，让它们参与同一次截面排序。
COUNT_LIKE = {
    "dragon_count_20", "ann_reduce_20", "ann_risk_20",
    "ann_incentive_20", "ann_net_score_20",
}


def to_rank(panel):
    """截面百分位排名，统一量纲到 0~1。"""
    return panel.rank(axis=1, pct=True)


def to_rank_aligned(panel, sign: int, index, columns, count_like: bool):
    """先对齐到完整标的池，再做截面排名。

    对齐必须在排名**之前**：计数型因子的缺失意味着计数为 0，
    补 0 后与其余标的一起排序才能得到正确的相对位置。
    先排名再填充等于让它们缺席了那次排序。
    """
    p = panel.reindex(index=index, columns=columns)
    if count_like:
        p = p.fillna(0.0)
    return to_rank(p * sign)


def load_alstm():
    """ALSTM 分数，列名转成 qlib 格式以便与因子对齐。"""
    import pandas as pd

    from qmtquant.datafeed.qlib_export import to_qlib_code

    if not paths.ALSTM_SCORES.exists():
        return None
    a = pd.read_parquet(paths.ALSTM_SCORES).sort_index()
    ren = {}
    for c in a.columns:
        try:
            ren[c] = to_qlib_code(str(c))
        except ValueError:
            pass
    return a.rename(columns=ren)


def combine(alstm, factors: dict, weights: dict):
    """加权合成。各路先转截面排名，缺失填中性值。"""
    import pandas as pd

    # 对齐到 ALSTM 的日期与标的 —— 它覆盖最全，是组合的基准
    idx, cols = alstm.index, alstm.columns

    parts = {"alstm": to_rank(alstm)}
    for name, sign in MONEY_FACTORS.items():
        if name in factors:
            parts[name] = to_rank_aligned(
                factors[name], sign, idx, cols, name in COUNT_LIKE)

    total = None
    wsum = 0.0
    for name, panel in parts.items():
        w = weights.get(name, 0.0)
        if w <= 0:
            continue
        p = panel.reindex(index=idx, columns=cols)
        # 计数型已在排名前补 0，这里剩下的缺失才是真的未知
        p = p.fillna(NEUTRAL)
        total = p * w if total is None else total + p * w
        wsum += w
    if total is None or wsum <= 0:
        return None

    out = (total / wsum).sort_index()
    # 合成过程在 qlib 代码格式下进行（因子面板用的是这个格式），
    # 但回测引擎与下游 SignalRankStrategy 要的是 vt 格式 ——
    # 不转回去，分数面板与标的池对不上，回测会全程空仓
    from qmtquant.datafeed.qlib_export import from_qlib_code

    ren = {}
    for c in out.columns:
        try:
            ren[c] = from_qlib_code(str(c))
        except ValueError:
            pass
    return out.rename(columns=ren)


def run_backtest(scores, cfg, bars, universe, symbols,
                 holdings: int, rebalance: int, capital: float,
                 start: str = None, end: str = None) -> dict:
    s = scores
    if start:
        s = s[s.index >= start]
    if end:
        s = s[s.index <= end]
    if s.empty:
        return {"error": "该区间无分数"}
    return backtest(s, cfg, bars, universe, symbols,
                    holdings, rebalance, capital)


def main() -> int:
    p = argparse.ArgumentParser(description="多因子合成")
    p.add_argument("--index", default="000852.SH")
    p.add_argument("--holdings", type=int, default=50)
    p.add_argument("--rebalance", type=int, default=20)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--split", default="2024-03-01")
    args = p.parse_args()

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.core.constants import Interval
    from qmtquant.datafeed.qlib_init import init_qlib
    from qmtquant.datafeed.xt_feed import XtDataFeed
    from qmtquant.universe.providers import HistoricalUniverse
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    uri = str(Path(cfg.data.store_dir) / "qlib_data")
    init_qlib(uri, n_expressions=8)

    print("=" * 72)
    print("多因子合成")
    print("=" * 72)

    alstm = load_alstm()
    if alstm is None:
        print("缺少 ALSTM 分数，请先训练")
        return 1
    factors = load_money_factors(cfg.data.store_dir)
    factors = {k: v for k, v in factors.items() if k in MONEY_FACTORS}
    print(f"ALSTM     : {alstm.shape[0]} 期 x {alstm.shape[1]} 只")
    for k, v in factors.items():
        print(f"{k:<10s}: {v.shape[0]} 期 x {v.shape[1]} 只"
              f"（已按可获得时点滞后）")
    if not factors:
        print("没有可用的资金流因子")
        return 1

    print("\n装载回测行情...")
    weight_csv = (Path(cfg.data.store_dir) / "universe"
                  / f"index_weight_{args.index}.csv")
    universe = HistoricalUniverse(str(weight_csv))
    symbols = universe.all_symbols()
    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    bars = feed.load_bars(symbols, TEST[0], TEST[1], Interval.DAILY)
    if not bars:
        print("测试段没有行情数据")
        return 1
    print(f"  {len(bars):,} 根 K 线")

    # ---- 待比较的几种配置
    ann = {"ann_reduce_20": 1.0, "ann_risk_20": 1.0,
           "ann_incentive_20": 1.0, "ann_net_score_20": 1.0}
    variants = {
        "仅 ALSTM": {"alstm": 1.0},
        "仅龙虎榜": {"dragon_count_20": 1.0},
        "仅公告(4因子)": dict(ann),
        "ALSTM + 龙虎": {"alstm": 0.8, "dragon_count_20": 0.2},
        "ALSTM + 公告": {"alstm": 0.8, **{k: 0.05 for k in ann}},
        "ALSTM + 龙虎 + 公告": {"alstm": 0.7, "dragon_count_20": 0.15,
                                **{k: 0.0375 for k in ann}},
        "等权全部": {"alstm": 1.0, "dragon_count_20": 1.0, **ann},
    }

    print("\n" + "-" * 72)
    print(f"{'配置':<18s}{'总收益':>11s}{'年化':>9s}{'最大回撤':>10s}"
          f"{'Sharpe':>9s}{'成交':>8s}{'合规':>6s}")
    print("-" * 72)

    t0 = time.time()
    rows = []
    panels = {}
    for label, w in variants.items():
        sc = combine(alstm, factors, w)
        if sc is None:
            print(f"{label:<18s}  合成失败")
            continue
        panels[label] = sc
        m = run_backtest(sc, cfg, bars, universe, symbols,
                         args.holdings, args.rebalance, args.capital)
        if "error" in m:
            print(f"{label:<18s}  {m['error']}")
            continue
        m["label"] = label
        m["weights"] = w
        rows.append(m)
        print(f"{label:<18s}{m['total_return']:>+11.2%}"
              f"{m['annual_return']:>+9.2%}{m['max_drawdown']:>10.2%}"
              f"{m['sharpe']:>+9.3f}{m['total_trades']:>8d}"
              f"{'达标' if m['drawdown_ok'] else '超限':>6s}")

    # ---- 分段稳定性：这是之前踩过的坑，必须查
    print("\n" + "-" * 72)
    print("分段检验（同一配置在前后两段是否都成立）")
    print("-" * 72)
    print(f"{'配置':<18s}{'前段 Sharpe':>14s}{'后段 Sharpe':>14s}{'一致性':>10s}")
    print("-" * 72)

    sub = []
    for label, sc in panels.items():
        h1 = run_backtest(sc, cfg, bars, universe, symbols,
                          args.holdings, args.rebalance, args.capital,
                          end=args.split)
        h2 = run_backtest(sc, cfg, bars, universe, symbols,
                          args.holdings, args.rebalance, args.capital,
                          start=args.split)
        if "error" in h1 or "error" in h2:
            continue
        flip = (h1["sharpe"] > 0) != (h2["sharpe"] > 0)
        sub.append({"label": label, "h1": h1["sharpe"], "h2": h2["sharpe"],
                    "flip": bool(flip),
                    "h1_dd": h1["max_drawdown"], "h2_dd": h2["max_drawdown"]})
        print(f"{label:<18s}{h1['sharpe']:>+14.3f}{h2['sharpe']:>+14.3f}"
              f"{('符号翻转' if flip else '一致'):>10s}")

    # ---- 结论
    print("\n" + "-" * 72)
    print("结论")
    print("-" * 72)
    base = next((r for r in rows if r["label"] == "仅 ALSTM"), None)
    best = max((r for r in rows), key=lambda r: r["sharpe"], default=None)
    if base and best:
        print(f"  基线（仅 ALSTM）  Sharpe {base['sharpe']:+.3f}  "
              f"回撤 {base['max_drawdown']:.2%}")
        print(f"  最佳（{best['label']}）  Sharpe {best['sharpe']:+.3f}  "
              f"回撤 {best['max_drawdown']:.2%}")
        print(f"  提升 {best['sharpe'] - base['sharpe']:+.3f}")

    stable = [s for s in sub if not s["flip"] and s["h1"] > 0 and s["h2"] > 0]
    if stable:
        print(f"\n  两段均为正的配置（{len(stable)} 个）:")
        for s in sorted(stable, key=lambda x: -(x["h1"] + x["h2"])):
            print(f"    {s['label']:<18s} 前段 {s['h1']:+.3f}  后段 {s['h2']:+.3f}")
    else:
        print("\n  !! 没有配置在两段都为正 —— 不具备跨期稳定性")

    # ---- 保存最优合成分数供下游使用
    if best and best["label"] in panels:
        paths.ensure_dirs()
        panels[best["label"]].to_parquet(COMBINED_SCORES)
        print(f"\n最优合成分数已保存: {COMBINED_SCORES}")
        print(f"  （配置：{best['label']}，权重 {best['weights']}）")

    RESULT_JSON.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"holdings": args.holdings, "rebalance": args.rebalance,
                   "capital": args.capital, "split": args.split,
                   "period": list(TEST), "money_factors": MONEY_FACTORS},
        "variants": rows,
        "subperiod": sub,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细: {RESULT_JSON}")
    print(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
