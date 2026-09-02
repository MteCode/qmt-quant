"""多种子实验 —— 量化 ALSTM 的随机性方差。

## 为什么要做这个

同一份代码、同一份数据、同一组超参，仅因神经网络随机初始化不同，
两次训练得到的组合回测差异巨大：

    第一份   总收益 +44.37%   Sharpe 1.008
    第二份   总收益 +24.33%   Sharpe 0.558

单次回测的数字因此无法回答「这个策略到底有没有 alpha」——
+44.37% 可能只是分布右尾的一次抽样。

本脚本用固定种子重复训练 N 次，给出 IC 与回测指标的**分布**而非点估计。
判断依据：

- IC 均值的分布若横跨 0，说明选股信号本身不稳定，谈不上 alpha
- 回测 Sharpe 的中位数才是对策略的合理预期，不是历史最好那次
- 若分布区间宽到覆盖「能上实盘」与「不能上实盘」两侧，结论是样本不足

## 成本优化

Alpha360 特征构建约 10 分钟，且与种子无关，因此只构建一次在种子间复用。
每个种子的增量成本 = 训练 ~10 分钟 + 回测 ~2 分钟。

用法::

    python strategies/alstm_ppo_csi1000/seed_experiment.py --seeds 5
    python strategies/alstm_ppo_csi1000/seed_experiment.py --seeds 10 --holdings 30
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

#: 结果落盘位置。每跑完一个种子就追加写入，中途中断不丢已有结果
RESULT_JSON = paths.BACKTEST_DIR / "seed_experiment.json"


def run_one_seed(seed: int, dataset, cfg, bars, universe, symbols,
                 holdings: int, rebalance: int, capital: float,
                 n_epochs: int, gpu: int) -> dict:
    """训练一个种子并回测，返回该次的全部指标。"""
    import pandas as pd
    from qlib.contrib.model.pytorch_alstm import ALSTM

    from qmtquant.engine.backtest_engine import BacktestEngine
    from qmtquant.risk.drawdown import DrawdownConfig, DrawdownController
    from qmtquant.strategy.signal_rank import SignalRankStrategy

    t0 = time.time()
    model = ALSTM(**HYPER, n_epochs=n_epochs, lr=0.001, early_stop=20,
                  batch_size=2000, loss="mse", GPU=gpu, seed=seed)
    model.fit(dataset)
    train_sec = time.time() - t0

    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    label = dataset.prepare("test", col_set="label", data_key="raw").iloc[:, 0]
    ic = evaluate_ic(pred, label)

    # ---- 组合回测。每个种子用全新的回撤控制器，避免状态串味
    r = cfg.risk
    drawdown = DrawdownController(DrawdownConfig(
        close_only_threshold=r.drawdown_close_only,
        reduce_threshold=r.drawdown_reduce,
        reduce_keep_ratio=r.drawdown_reduce_keep,
        flat_threshold=r.drawdown_flat,
        recovery_ratio=r.drawdown_recovery_ratio,
        min_observations=r.drawdown_min_observations,
        max_freeze_observations=r.drawdown_max_freeze))

    engine = BacktestEngine(initial_capital=capital, cost=cfg.cost,
                            drawdown=drawdown)
    engine.load_data(bars)
    engine.set_universe(universe)
    engine.add_strategy(SignalRankStrategy, symbols, {
        "max_holdings": holdings, "rebalance_days": rebalance})
    scores = to_score_panel(pred).sort_index()
    engine.strategy.scores = scores
    engine.strategy._score_dates = scores.index.to_numpy()
    stats = engine.run()

    return {
        "seed": seed,
        "ic_mean": ic.get("IC均值"),
        "ic_ir": ic.get("ICIR"),
        "ic_t": ic.get("t值"),
        "ic_positive_ratio": ic.get("IC>0占比"),
        "total_return": stats.total_return,
        "annual_return": stats.annual_return,
        "max_drawdown": abs(stats.max_drawdown),
        "sharpe": stats.sharpe_ratio,
        "calmar": stats.calmar_ratio,
        "volatility": stats.volatility,
        "win_rate": stats.win_rate,
        "profit_factor": stats.profit_factor,
        "total_trades": stats.total_trades,
        "drawdown_ok": stats.drawdown_ok,
        "peak_resets": drawdown.state.peak_resets,
        "train_sec": round(train_sec),
    }


def summarize(rows: list, drawdown_limit: float = 0.20) -> str:
    """把多次结果汇总成分布描述。"""
    import numpy as np

    if not rows:
        return "无结果"

    def dist(key, pct=True, fmt="{:+.2%}"):
        v = np.array([r[key] for r in rows if r.get(key) is not None],
                     dtype=float)
        if len(v) == 0:
            return "  (无数据)"
        f = (lambda x: fmt.format(x)) if pct else (lambda x: f"{x:+.3f}")
        return (f"  {f(v.min())} ~ {f(v.max())}   "
                f"中位数 {f(np.median(v))}   均值 {f(v.mean())}   "
                f"标准差 {f(v.std(ddof=1)) if len(v) > 1 else '—'}")

    n = len(rows)
    lines = [
        "=" * 66,
        f"多种子实验汇总  N = {n}",
        "=" * 66,
        "",
        "IC 均值（选股信号强度）",
        dist("ic_mean", fmt="{:+.4f}"),
        "",
        "IC t 值（显著性，|t|>2 才算显著）",
        dist("ic_t", pct=False),
        "",
        "总收益",
        dist("total_return"),
        "",
        "年化收益",
        dist("annual_return"),
        "",
        "最大回撤",
        dist("max_drawdown", fmt="{:.2%}"),
        "",
        "Sharpe",
        dist("sharpe", pct=False),
        "",
    ]

    # ---- 结论性判断
    import numpy as np
    ic_v = np.array([r["ic_mean"] for r in rows], dtype=float)
    sh_v = np.array([r["sharpe"] for r in rows], dtype=float)
    dd_v = np.array([abs(r["max_drawdown"]) for r in rows], dtype=float)
    t_v = np.array([abs(r["ic_t"]) for r in rows], dtype=float)

    lines += ["-" * 66, "判断", "-" * 66]

    if ic_v.min() > 0:
        lines.append(f"  IC 全部为正（{n}/{n}），选股方向稳定")
    else:
        neg = int((ic_v <= 0).sum())
        lines.append(f"  !! IC 有 {neg}/{n} 次非正，选股方向不稳定")

    sig = int((t_v >= 2).sum())
    lines.append(f"  IC 显著（|t|>=2）的次数: {sig}/{n}")

    ok = int((dd_v <= drawdown_limit).sum())
    lines.append(f"  回撤达标（<={drawdown_limit:.0%}）的次数: {ok}/{n}")

    pos = int((sh_v > 0).sum())
    lines.append(f"  Sharpe 为正的次数: {pos}/{n}")

    if n > 1:
        spread = sh_v.max() - sh_v.min()
        lines.append(f"  Sharpe 极差: {spread:.3f}"
                     f"（中位数 {np.median(sh_v):.3f}）")
        if spread > abs(np.median(sh_v)):
            lines.append("  !! 极差大于中位数本身 —— 单次回测数字不可作为决策依据")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="ALSTM 多种子方差实验")
    p.add_argument("--seeds", type=int, default=5, help="跑多少个种子")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--market", default="csi1000")
    p.add_argument("--index", default="000852.SH")
    p.add_argument("--holdings", type=int, default=50)
    p.add_argument("--rebalance", type=int, default=20)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
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
    init_qlib(uri, n_expressions=360)

    print("=" * 66)
    print("ALSTM 多种子方差实验")
    print("=" * 66)
    print(f"种子      : {args.seed_start} ~ {args.seed_start + args.seeds - 1}"
          f"（共 {args.seeds} 次）")
    print(f"持仓      : {args.holdings} 只，每 {args.rebalance} 日调仓")
    print(f"测试段    : {TEST[0]} ~ {TEST[1]}（样本外）")
    print()

    t_all = time.time()

    # ---- 特征只构建一次，种子间复用（每个种子省约 10 分钟）
    print("构建 Alpha360 特征（只做一次，种子间复用）...")
    t0 = time.time()
    dataset = build_dataset(uri, args.market)
    print(f"  完成，耗时 {time.time() - t0:.0f}s")

    # ---- 行情与标的池同样只装载一次
    print("\n装载回测行情（只做一次）...")
    t0 = time.time()
    weight_csv = (Path(cfg.data.store_dir) / "universe"
                  / f"index_weight_{args.index}.csv")
    universe = HistoricalUniverse(str(weight_csv))
    symbols = universe.all_symbols()
    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    bars = feed.load_bars(symbols, TEST[0], TEST[1], Interval.DAILY)
    if not bars:
        print("测试段没有行情数据")
        return 1
    print(f"  {len(bars):,} 根 K 线，耗时 {time.time() - t0:.0f}s")

    rows = []
    paths.ensure_dirs()

    for i in range(args.seeds):
        seed = args.seed_start + i
        print("\n" + "=" * 66)
        print(f"种子 {seed}  ({i + 1}/{args.seeds})")
        print("=" * 66)
        try:
            row = run_one_seed(
                seed, dataset, cfg, bars, universe, symbols,
                args.holdings, args.rebalance, args.capital,
                args.n_epochs, args.gpu)
        except Exception as e:
            print(f"  [FAIL] 种子 {seed} 失败: {type(e).__name__}: {e}")
            continue

        rows.append(row)
        print(f"  IC {row['ic_mean']:+.4f} (t={row['ic_t']:+.2f})   "
              f"收益 {row['total_return']:+.2%}   "
              f"回撤 {row['max_drawdown']:.2%}   "
              f"Sharpe {row['sharpe']:+.3f}   "
              f"[{row['train_sec']}s]")

        # 每完成一个就落盘，中途中断不丢
        RESULT_JSON.write_text(json.dumps({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "market": args.market, "holdings": args.holdings,
                "rebalance": args.rebalance, "capital": args.capital,
                "n_epochs": args.n_epochs, "test": list(TEST),
                "hyper": HYPER,
            },
            "runs": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + summarize(rows, cfg.risk_max_drawdown
                           if hasattr(cfg, "risk_max_drawdown") else 0.20))
    print(f"\n明细已保存: {RESULT_JSON}")
    print(f"总耗时 {(time.time() - t_all) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
