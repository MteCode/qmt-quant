"""组合构造参数扫描 —— 持仓只数 x 调仓周期。

## 为什么先扫这个，而不是先改模型

8 种子实验给出的诊断很明确：

- IC 中位数 +0.0183，6/8 显著 —— 信号存在
- IC 与 Sharpe 的相关系数只有 0.338 —— 信号没能转化成收益
- seed 7 是信号最强的一个（IC +0.0271, t=6.75），回测 Sharpe 却是 -0.347

**最强的信号产出了最差的组合**。这说明瓶颈不在信号强度，而在从信号到
持仓的这一步。此时去换模型架构（提升信号）解决不了问题。

集成分数是当前最好的一份信号，用它扫参数，能把「组合构造能挽回多少」
这件事量化出来。回测约 4 秒一次，整个网格几分钟跑完，成本远低于换架构。

## 扫哪两个维度

- **持仓只数**：越少越集中，信号浓度高但个股风险大
- **调仓周期**：越长换手越低，成本省下来但信号衰减

两者都直接决定换手率，而换手率决定成本拖累。当前配置（50 只 / 20 日）
在 4.7 年里成交 2385 笔，调仓 56 次几乎每次全换。

用法::

    python strategies/alstm_ppo_csi1000/sweep_portfolio.py
    python strategies/alstm_ppo_csi1000/sweep_portfolio.py --scores models/ensemble/scores_seed0.parquet
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
from train_alstm import TEST  # noqa: E402
from train_ensemble import backtest  # noqa: E402

RESULT_JSON = paths.BACKTEST_DIR / "sweep_portfolio.json"

HOLDINGS = [10, 20, 30, 50, 80]
REBALANCE = [5, 10, 20, 40, 60]


def main() -> int:
    p = argparse.ArgumentParser(description="组合构造参数扫描")
    p.add_argument("--scores", default=None,
                    help="分数面板路径，默认用 models/alstm_scores.parquet（集成）")
    p.add_argument("--index", default="000852.SH")
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--holdings", type=int, nargs="*", default=HOLDINGS)
    p.add_argument("--rebalance", type=int, nargs="*", default=REBALANCE)
    p.add_argument("--start", default=TEST[0],
                    help="回测起始日，用于分段检验参数是否跨期稳定")
    p.add_argument("--end", default=TEST[1])
    p.add_argument("--tag", default="", help="结果文件后缀，避免分段结果互相覆盖")
    args = p.parse_args()

    import pandas as pd

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.core.constants import Interval
    from qmtquant.datafeed.qlib_init import init_qlib
    from qmtquant.datafeed.xt_feed import XtDataFeed
    from qmtquant.universe.providers import HistoricalUniverse
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    uri = str(Path(cfg.data.store_dir) / "qlib_data")
    init_qlib(uri, n_expressions=32)

    scores_path = Path(args.scores) if args.scores else paths.ALSTM_SCORES
    if not scores_path.exists():
        print(f"分数文件不存在: {scores_path}")
        return 1
    scores = pd.read_parquet(scores_path).sort_index()

    print("=" * 74)
    print("组合构造参数扫描")
    print("=" * 74)
    print(f"分数      : {scores_path.name}  ({scores.shape[0]} 期 x {scores.shape[1]} 只)")
    print(f"回测段    : {args.start} ~ {args.end}")
    print(f"网格      : 持仓 {args.holdings} x 调仓 {args.rebalance}"
          f" = {len(args.holdings) * len(args.rebalance)} 组")

    print("\n装载回测行情...")
    weight_csv = (Path(cfg.data.store_dir) / "universe"
                  / f"index_weight_{args.index}.csv")
    universe = HistoricalUniverse(str(weight_csv))
    symbols = universe.all_symbols()
    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    bars = feed.load_bars(symbols, args.start, args.end, Interval.DAILY)
    if not bars:
        print("测试段没有行情数据")
        return 1
    print(f"  {len(bars):,} 根 K 线")

    rows = []
    t0 = time.time()

    print("\n" + "-" * 74)
    print(f"{'持仓':>5s}{'调仓':>6s}{'总收益':>11s}{'年化':>9s}"
          f"{'最大回撤':>10s}{'Sharpe':>9s}{'成交':>8s}{'合规':>6s}")
    print("-" * 74)

    for h in args.holdings:
        for rb in args.rebalance:
            m = backtest(scores, cfg, bars, universe, symbols,
                         h, rb, args.capital)
            m["holdings"] = h
            m["rebalance"] = rb
            rows.append(m)
            print(f"{h:>5d}{rb:>6d}"
                  f"{m['total_return']:>+11.2%}{m['annual_return']:>+9.2%}"
                  f"{m['max_drawdown']:>10.2%}{m['sharpe']:>+9.3f}"
                  f"{m['total_trades']:>8d}"
                  f"{'✓' if m['drawdown_ok'] else '✗':>6s}")
        print()

    # ---- 汇总
    ok = [r for r in rows if r["drawdown_ok"]]
    print("-" * 74)
    print(f"回撤达标: {len(ok)}/{len(rows)} 组")

    if ok:
        best = max(ok, key=lambda r: r["sharpe"])
        print(f"\n达标组中 Sharpe 最高:")
        print(f"  持仓 {best['holdings']} 只 / 每 {best['rebalance']} 日调仓")
        print(f"  收益 {best['total_return']:+.2%}   "
              f"年化 {best['annual_return']:+.2%}   "
              f"回撤 {best['max_drawdown']:.2%}   "
              f"Sharpe {best['sharpe']:+.3f}   "
              f"成交 {best['total_trades']} 笔")

        cur = next((r for r in rows
                    if r["holdings"] == 50 and r["rebalance"] == 20), None)
        if cur:
            print(f"\n对比当前配置（50 只 / 20 日）:")
            print(f"  Sharpe {cur['sharpe']:+.3f} -> {best['sharpe']:+.3f}"
                  f"  ({best['sharpe'] - cur['sharpe']:+.3f})")
            print(f"  成交   {cur['total_trades']} -> {best['total_trades']} 笔"
                  f"  ({best['total_trades'] - cur['total_trades']:+d})")

    out = (RESULT_JSON.with_name(f"sweep_portfolio{args.tag}.json")
           if args.tag else RESULT_JSON)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scores": str(scores_path),
        "period": [args.start, args.end],
        "runs": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细: {out}")
    print(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
