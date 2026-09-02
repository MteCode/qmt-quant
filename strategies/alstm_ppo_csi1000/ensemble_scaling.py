"""集成规模 vs 稳定性 —— 到底需要几个种子。

8 种子集成的稳定性检验没通过：两组各 4 个种子的子集成，Sharpe 差了 0.576，
达到单模型极差(1.023)的 56%。这说明 4 个不够，但没回答**几个才够**。

本脚本不重新训练 —— 直接复用 models/ensemble/ 下已保存的 8 份分数面板，
对每个规模 k 抽样若干个组合做回测，测量该规模下结果的离散程度。

若离散度随 k 单调下降并收敛，就能读出「加到第几个种子之后收益递减」；
若到 k=8 仍未收敛，说明 8 个种子远远不够，需要训练更多。

一次回测约 4 秒，全部组合几分钟跑完，成本远低于重训。

用法::

    python strategies/alstm_ppo_csi1000/ensemble_scaling.py
    python strategies/alstm_ppo_csi1000/ensemble_scaling.py --samples 20
"""
import argparse
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import paths  # noqa: E402
from train_alstm import TEST  # noqa: E402
from train_ensemble import backtest, combine  # noqa: E402

RESULT_JSON = paths.BACKTEST_DIR / "ensemble_scaling.json"


def load_panels(max_seeds: int):
    """加载已保存的各种子分数面板。"""
    import pandas as pd

    panels = {}
    for seed in range(max_seeds):
        p = paths.ensemble_scores(seed)
        if p.exists():
            panels[seed] = pd.read_parquet(p)
    return panels


def main() -> int:
    p = argparse.ArgumentParser(description="集成规模 vs 稳定性")
    p.add_argument("--max-seeds", type=int, default=8)
    p.add_argument("--samples", type=int, default=12,
                    help="每个规模最多抽样多少个组合")
    p.add_argument("--index", default="000852.SH")
    p.add_argument("--holdings", type=int, default=50)
    p.add_argument("--rebalance", type=int, default=20)
    p.add_argument("--capital", type=float, default=1_000_000)
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
    init_qlib(uri, n_expressions=32)

    panels = load_panels(args.max_seeds)
    if len(panels) < 2:
        print(f"可用分数面板不足（{len(panels)} 个）。"
              f"请先运行 train_ensemble.py")
        return 1

    seeds = sorted(panels)
    print("=" * 70)
    print("集成规模 vs 稳定性")
    print("=" * 70)
    print(f"可用种子  : {seeds}")
    print(f"每规模抽样: 最多 {args.samples} 个组合")
    print(f"测试段    : {TEST[0]} ~ {TEST[1]}（样本外）")

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

    rng = random.Random(42)
    rows = []
    t_all = time.time()

    print("\n" + "-" * 70)
    print(f"{'规模':>4s}{'组合数':>8s}{'Sharpe 区间':>26s}"
          f"{'中位数':>10s}{'标准差':>10s}{'达标率':>8s}")
    print("-" * 70)

    for k in range(1, len(seeds) + 1):
        combos = list(itertools.combinations(seeds, k))
        if len(combos) > args.samples:
            combos = rng.sample(combos, args.samples)

        sharpes, dds, rets, oks = [], [], [], []
        for combo in combos:
            sc = combine([panels[s] for s in combo])
            m = backtest(sc, cfg, bars, universe, symbols,
                         args.holdings, args.rebalance, args.capital)
            sharpes.append(m["sharpe"])
            dds.append(m["max_drawdown"])
            rets.append(m["total_return"])
            oks.append(bool(m["drawdown_ok"]))
            rows.append({"k": k, "seeds": list(combo), **m})

        sh = np.array(sharpes)
        sd = sh.std(ddof=1) if len(sh) > 1 else 0.0
        print(f"{k:>4d}{len(combos):>8d}"
              f"{f'{sh.min():+.3f} ~ {sh.max():+.3f}':>26s}"
              f"{np.median(sh):>+10.3f}{sd:>10.3f}"
              f"{sum(oks)}/{len(oks)}".rjust(8))

    # ---- 收敛判断
    print("-" * 70)
    by_k = {}
    for r in rows:
        by_k.setdefault(r["k"], []).append(r["sharpe"])

    sd1 = np.std(by_k[1], ddof=1) if len(by_k.get(1, [])) > 1 else None
    print("\n判断:")
    if sd1:
        print(f"  单模型 Sharpe 标准差: {sd1:.3f}")
        for k in sorted(by_k):
            if k == 1 or len(by_k[k]) < 2:
                continue
            sdk = np.std(by_k[k], ddof=1)
            print(f"    k={k}: 标准差 {sdk:.3f}"
                  f"  （降至单模型的 {sdk / sd1:.0%}）")

    ks = sorted(x for x in by_k if len(by_k[x]) > 1)
    if len(ks) >= 2:
        last_k = ks[-1]
        last_sd = np.std(by_k[last_k], ddof=1)
        print(f"\n  最大规模 k={last_k} 的标准差仍有 {last_sd:.3f}")
        if sd1 and last_sd > sd1 * 0.35:
            print(f"  !! 未收敛 —— 仍达单模型的 {last_sd / sd1:.0%}，"
                  f"{last_k} 个种子不够，需要训练更多")
        else:
            print("  -> 已趋于收敛")

    RESULT_JSON.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"max_seeds": args.max_seeds, "samples": args.samples,
                   "holdings": args.holdings, "rebalance": args.rebalance,
                   "test": list(TEST)},
        "runs": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细: {RESULT_JSON}")
    print(f"总耗时 {(time.time() - t_all) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
