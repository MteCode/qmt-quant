"""ALSTM 集成 —— 用多个种子的截面排名平均消掉训练随机性。

## 为什么要做

8 种子实验（seed_experiment.json）显示单模型的表现分布横跨合格线两侧：

    Sharpe    -0.491 ~ +0.532   中位数 -0.027
    最大回撤   17.95% ~ 30.27%   中位数 21.07%
    回撤达标   4/8

任何单次训练的结果都是从这个分布里抽一次签。这不只是「成绩不稳」——
更要命的是**后续任何改动的效果都会被随机性淹没**，无法判断改得对不对。
集成先把这层噪声压下去，实验结论才谈得上可信。

## 合成方式：截面排名平均，不是原始分数平均

不同种子训练出的网络输出量纲不一致（有的分数集中在 ±0.01，有的散在 ±0.1）。
直接平均会让数值范围大的模型主导结果，等于加权不均。

改为每个交易日先在截面上做百分位排名（0~1），再跨模型平均。
排名对单调变换不敏感，量纲差异被消除，且下游 SignalRankStrategy
本来就只用分数的相对顺序。

## 集成本身稳不稳？

降方差是理论，得实测。脚本会把种子拆成两组不相交的子集各做一次集成，
比较两者差距 —— 若两个集成的结果仍然差很多，说明种子数不够，
集成没起到该有的作用。

用法::

    python strategies/alstm_ppo_csi1000/train_ensemble.py --seeds 8
    python strategies/alstm_ppo_csi1000/train_ensemble.py --reuse   # 复用已训练的种子
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
from train_alstm import (  # noqa: E402
    HYPER, TEST, build_dataset, evaluate_ic, to_score_panel,
)

RESULT_JSON = paths.BACKTEST_DIR / "ensemble_result.json"


def train_seed(seed: int, dataset, n_epochs: int, gpu: int):
    """训练一个种子，保存权重与分数面板，返回 (分数面板, IC 字典)。"""
    import pandas as pd
    import torch
    from qlib.contrib.model.pytorch_alstm import ALSTM

    w_path = paths.ensemble_weights(seed)
    s_path = paths.ensemble_scores(seed)

    if w_path.exists() and s_path.exists():
        print(f"  复用已有产物: {w_path.name}")
        scores = pd.read_parquet(s_path)
        meta_path = paths.ENSEMBLE_DIR / f"meta_seed{seed}.json"
        ic = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return scores, ic

    t0 = time.time()
    model = ALSTM(**HYPER, n_epochs=n_epochs, lr=0.001, early_stop=20,
                  batch_size=2000, loss="mse", GPU=gpu, seed=seed)
    model.fit(dataset)

    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    label = dataset.prepare("test", col_set="label", data_key="raw").iloc[:, 0]
    ic = evaluate_ic(pred, label)

    scores = to_score_panel(pred).sort_index()

    paths.ensure_dirs()
    torch.save(model.ALSTM_model.state_dict(), w_path)
    scores.to_parquet(s_path)
    (paths.ENSEMBLE_DIR / f"meta_seed{seed}.json").write_text(
        json.dumps(ic, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  训练完成 {time.time() - t0:.0f}s   "
          f"IC {ic.get('IC均值', 0):+.4f} (t={ic.get('t值', 0):+.2f})")
    return scores, ic


def combine(panels: list):
    """截面排名平均。

    每个交易日在横截面上做百分位排名再跨模型取均值，
    消除不同模型输出量纲不一致带来的加权不均。
    """
    import pandas as pd

    ranked = [p.rank(axis=1, pct=True) for p in panels]
    stacked = pd.concat(ranked)
    return stacked.groupby(stacked.index).mean().sort_index()


def backtest(scores, cfg, bars, universe, symbols,
             holdings: int, rebalance: int, capital: float,
             use_risk: bool = True, phase: int = 0,
             exposure: float = 1.0) -> dict:
    """跑组合回测，返回指标字典。

    ``use_risk=False`` 关掉回撤控制器。**这只是诊断用途** ——
    实盘必须开着。它的作用是把两件事分开：信号本身赚不赚钱，
    和风控层在反复砍仓时自己损耗了多少。两者混在一起时，
    看到一个差的 Sharpe 无法判断该改信号还是该改风控参数。

    ``exposure`` 修正一处回测与实盘的不一致。仓位在回测里是靠
    **缩小本金**实现的（30% 仓位 = 拿 15 万跑满仓），于是回撤控制器
    看到的是投入部分的净值曲线。但实盘账户里那 35 万现金是在的，
    控制器看到的是 50 万口径 —— 同一段行情，两边算出的回撤差 3.3 倍。

    实测后果：档位是 6%/9%/12%，30% 仓位下账户只回撤 3.6% 就被全平，
    而硬约束是 20%。回测里风控疯狂砍仓，实盘根本不会触发。
    传入 exposure 后按 1/exposure 放大阈值，让控制器在**账户口径**
    的同一位置触发。
    """
    from qmtquant.engine.backtest_engine import BacktestEngine
    from qmtquant.risk.drawdown import DrawdownConfig, DrawdownController
    from qmtquant.strategy.signal_rank import SignalRankStrategy

    r = cfg.risk
    # 阈值按仓位放大，还原到账户口径。上限 0.95 —— 再高就等于没有风控，
    # 而且投入部分亏 95% 这种情形本来也不该靠回撤控制器兜底
    sc = 1.0 / max(min(exposure, 1.0), 0.05)
    drawdown = DrawdownController(DrawdownConfig(
        close_only_threshold=min(r.drawdown_close_only * sc, 0.95),
        reduce_threshold=min(r.drawdown_reduce * sc, 0.95),
        reduce_keep_ratio=r.drawdown_reduce_keep,
        flat_threshold=min(r.drawdown_flat * sc, 0.95),
        recovery_ratio=r.drawdown_recovery_ratio,
        min_observations=r.drawdown_min_observations,
        max_freeze_observations=r.drawdown_max_freeze)) if use_risk else None

    engine = BacktestEngine(initial_capital=capital, cost=cfg.cost,
                            drawdown=drawdown)
    engine.load_data(bars)
    engine.set_universe(universe)
    engine.add_strategy(SignalRankStrategy, symbols, {
        "max_holdings": holdings, "rebalance_days": rebalance,
        "rebalance_phase": phase})
    engine.strategy.scores = scores
    engine.strategy._score_dates = scores.index.to_numpy()
    stats = engine.run()

    return {
        "total_return": stats.total_return,
        "annual_return": stats.annual_return,
        "max_drawdown": abs(stats.max_drawdown),
        "sharpe": stats.sharpe_ratio,
        "calmar": stats.calmar_ratio,
        "win_rate": stats.win_rate,
        "total_trades": stats.total_trades,
        "drawdown_ok": stats.drawdown_ok,
        "peak_resets": drawdown.state.peak_resets if drawdown else 0,
        "risk_exit_orders": getattr(engine, "risk_exit_orders", 0),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="ALSTM 集成")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--market", default="csi1000")
    p.add_argument("--index", default="000852.SH")
    p.add_argument("--holdings", type=int, default=50)
    p.add_argument("--rebalance", type=int, default=20)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    import numpy as np

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.core.constants import Interval
    from qmtquant.datafeed.qlib_init import init_qlib
    from qmtquant.datafeed.xt_feed import XtDataFeed
    from qmtquant.universe.providers import HistoricalUniverse
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    uri = str(Path(cfg.data.store_dir) / "qlib_data")
    init_qlib(uri, n_expressions=360)

    print("=" * 66)
    print("ALSTM 集成 —— 截面排名平均")
    print("=" * 66)
    print(f"种子数    : {args.seeds}")
    print(f"持仓      : {args.holdings} 只，每 {args.rebalance} 日调仓")
    print(f"测试段    : {TEST[0]} ~ {TEST[1]}（样本外）")
    print()

    t_all = time.time()

    print("构建 Alpha360 特征（只做一次）...")
    t0 = time.time()
    dataset = build_dataset(uri, args.market)
    print(f"  完成，耗时 {time.time() - t0:.0f}s")

    print("\n装载回测行情（只做一次）...")
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

    # ---- 逐个种子训练
    panels, ics, singles = [], [], []
    for seed in range(args.seeds):
        print(f"\n--- 种子 {seed} ({seed + 1}/{args.seeds}) ---")
        scores, ic = train_seed(seed, dataset, args.n_epochs, args.gpu)
        panels.append(scores)
        ics.append(ic)

        m = backtest(scores, cfg, bars, universe, symbols,
                     args.holdings, args.rebalance, args.capital)
        m["seed"] = seed
        m["ic_mean"] = ic.get("IC均值")
        m["ic_t"] = ic.get("t值")
        singles.append(m)
        print(f"  单模型: 收益 {m['total_return']:+.2%}  "
              f"回撤 {m['max_drawdown']:.2%}  Sharpe {m['sharpe']:+.3f}")

    # ---- 全量集成
    print("\n" + "=" * 66)
    print("集成回测")
    print("=" * 66)
    ens_scores = combine(panels)
    ens = backtest(ens_scores, cfg, bars, universe, symbols,
                   args.holdings, args.rebalance, args.capital)
    print(f"\n全部 {args.seeds} 个种子集成:")
    print(f"  收益 {ens['total_return']:+.2%}   "
          f"回撤 {ens['max_drawdown']:.2%}   "
          f"Sharpe {ens['sharpe']:+.3f}   "
          f"{'达标' if ens['drawdown_ok'] else '回撤超限'}")

    # ---- 集成稳定性：两组不相交的种子各做一次
    half = args.seeds // 2
    sub_results = []
    if half >= 2:
        print(f"\n集成稳定性检验（两组不相交子集，各 {half} 个种子）:")
        for name, idx in [("前半", range(half)),
                          ("后半", range(half, args.seeds))]:
            sub = combine([panels[i] for i in idx])
            r = backtest(sub, cfg, bars, universe, symbols,
                         args.holdings, args.rebalance, args.capital)
            r["group"] = name
            r["seeds"] = list(idx)
            sub_results.append(r)
            print(f"  {name} {list(idx)}: 收益 {r['total_return']:+.2%}  "
                  f"回撤 {r['max_drawdown']:.2%}  Sharpe {r['sharpe']:+.3f}")

    # ---- 对比
    sh = np.array([s["sharpe"] for s in singles])
    dd = np.array([s["max_drawdown"] for s in singles])
    ret = np.array([s["total_return"] for s in singles])

    print("\n" + "-" * 66)
    print("单模型分布  vs  集成")
    print("-" * 66)
    print(f"{'':12s}{'单模型区间':>26s}{'中位数':>12s}{'集成':>12s}")
    print(f"{'总收益':12s}{ret.min():+.2%} ~ {ret.max():+.2%}".ljust(38)
          + f"{np.median(ret):+.2%}".rjust(12)
          + f"{ens['total_return']:+.2%}".rjust(12))
    print(f"{'最大回撤':12s}{dd.min():.2%} ~ {dd.max():.2%}".ljust(38)
          + f"{np.median(dd):.2%}".rjust(12)
          + f"{ens['max_drawdown']:.2%}".rjust(12))
    print(f"{'Sharpe':12s}{sh.min():+.3f} ~ {sh.max():+.3f}".ljust(38)
          + f"{np.median(sh):+.3f}".rjust(12)
          + f"{ens['sharpe']:+.3f}".rjust(12))

    print("\n判断:")
    print(f"  集成 Sharpe {ens['sharpe']:+.3f}  "
          f"vs 单模型中位数 {np.median(sh):+.3f}  "
          f"（提升 {ens['sharpe'] - np.median(sh):+.3f}）")
    better = int((ens["sharpe"] > sh).sum())
    print(f"  集成优于 {better}/{len(sh)} 个单模型")
    if ens["drawdown_ok"]:
        print(f"  回撤 {ens['max_drawdown']:.2%} 达标（<=20%）")
    else:
        print(f"  !! 回撤 {ens['max_drawdown']:.2%} 仍超 20% 硬约束")

    if len(sub_results) == 2:
        gap = abs(sub_results[0]["sharpe"] - sub_results[1]["sharpe"])
        span = sh.max() - sh.min()
        print(f"  两组子集成的 Sharpe 差距 {gap:.3f}，"
              f"单模型极差 {span:.3f}")
        if gap < span / 3:
            print("  -> 集成显著降低了随机性")
        else:
            print("  !! 集成后差距仍大，种子数不足以稳定结果")

    # ---- 落盘
    paths.ensure_dirs()
    ens_scores.to_parquet(paths.ALSTM_SCORES)
    print(f"\n集成分数已写入: {paths.ALSTM_SCORES}")
    print("  （下游 train_ppo.py / generate_signal.py 直接读这个文件）")

    RESULT_JSON.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"seeds": args.seeds, "holdings": args.holdings,
                   "rebalance": args.rebalance, "capital": args.capital,
                   "combine": "cross_sectional_rank_mean", "test": list(TEST)},
        "singles": singles,
        "ensemble": ens,
        "subsets": sub_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细: {RESULT_JSON}")
    print(f"\n总耗时 {(time.time() - t_all) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
