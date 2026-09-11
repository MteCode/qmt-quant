"""PPO 多种子实验 —— 量化择时策略的随机性方差。

## 为什么要做这个

PPO 自身的随机性**从未被度量**。已有的直接证据表明它不稳定：同一套代码，
仅把选股分数从 ALSTM 换成 LightGBM，学出的平均仓位就从 30% 跳到 70%、
回撤从 13.96% 恶化到 33.29%。所以那个 Sharpe 0.899 可信度存疑 ——
它可能只是分布右尾的一次抽样。

本脚本**不重训**：直接加载 train_ppo.py --seed N --tag _sN 产出的
models/ppo_model_s0..sN.zip，在同一测试段上做确定性推理，给出指标的
**分布**而非点估计。判断依据与 ALSTM 版一致：

- Sharpe 分布若横跨 0，说明择时没有稳定 alpha
- Sharpe 极差盖过中位数本身 → 单次回测数字不可作决策依据
- 平均仓位种子间差异过大 → 择时行为不稳定

用法::

    python strategies/alstm_ppo_csi1000/ppo_seed_experiment.py --seeds 5
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402
from train_ppo import (  # noqa: E402
    HOLD_K, REBAL_PERIOD, TEST, build_env, evaluate_ppo, prepare_data,
)

RESULT_JSON = paths.BACKTEST_DIR / "ppo_seed_experiment.json"

#: 平均仓位的种子间极差超过它，判定择时行为不稳定
EXPOSURE_SPREAD_LIMIT = 0.30


def run_one_seed(seed: int, env) -> dict | None:
    """加载一个种子的权重并在测试环境上确定性推理。不训练。"""
    from stable_baselines3 import PPO

    model_path = paths.ppo_seed_model(seed)
    if not model_path.exists():
        print(f"  [SKIP] 种子 {seed}: 无权重 {model_path.name}，"
              f"先跑 train_ppo.py --seed {seed} --tag _s{seed}")
        return None

    model = PPO.load(str(model_path), device="cpu")
    obs_shape = tuple(model.observation_space.shape)
    if obs_shape != (env.state_dim,):
        raise RuntimeError(
            f"种子 {seed} 的观测维度 {obs_shape} 与环境 {(env.state_dim,)} 不符 —— "
            f"权重是用不同特征集训练的，加载它得到的结论不作数")

    t0 = time.time()
    m = evaluate_ppo(model, env)
    return {
        "seed": seed,
        "model": model_path.name,
        "total_return": m["total_return"],
        "annual_return": m["annual_return"],
        "annual_vol": m["annual_vol"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "mean_exposure": m["mean_exposure"],
        "exposure_std": m["exposure_std"],
        "final_value": m["final_value"],
        "n_days": m["n_days"],
        "eval_sec": round(time.time() - t0, 1),
    }


def summarize(rows: list, drawdown_limit: float = 0.20) -> str:
    """把多个种子的结果汇总成分布描述 + 稳定性结论。"""
    if not rows:
        return "无结果"

    def dist(key, pct=True, fmt="{:+.2%}"):
        v = np.array([r[key] for r in rows if r.get(key) is not None],
                     dtype=float)
        if len(v) == 0:
            return "  (无数据)"
        f = (lambda x: fmt.format(x)) if pct else (lambda x: f"{x:+.3f}")
        return (f"  {f(v.min())} ~ {f(v.max())}   中位数 {f(np.median(v))}   "
                f"均值 {f(v.mean())}   标准差 "
                f"{f(v.std(ddof=1)) if len(v) > 1 else '—'}")

    n = len(rows)
    lines = [
        "=" * 66,
        f"PPO 多种子实验汇总  N = {n}",
        "=" * 66,
        "",
        "总收益", dist("total_return"),
        "",
        "年化收益", dist("annual_return"),
        "",
        "最大回撤", dist("max_drawdown", fmt="{:.2%}"),
        "",
        "Sharpe", dist("sharpe", pct=False),
        "",
        "平均仓位", dist("mean_exposure", fmt="{:.1%}"),
        "",
    ]

    sh = np.array([r["sharpe"] for r in rows], dtype=float)
    dd = np.array([abs(r["max_drawdown"]) for r in rows], dtype=float)
    ex = np.array([r["mean_exposure"] for r in rows], dtype=float)

    lines += ["-" * 66, "判断", "-" * 66]
    lines.append(f"  Sharpe 为正的次数: {int((sh > 0).sum())}/{n}")
    dd_ok = int((dd <= drawdown_limit).sum())
    lines.append(f"  回撤达标（<={drawdown_limit:.0%}）的次数: {dd_ok}/{n}")

    stable = dd_ok == n
    if n > 1:
        spread = sh.max() - sh.min()
        lines.append(f"  Sharpe 极差: {spread:.3f}"
                     f"（中位数 {np.median(sh):.3f}）")
        if spread > abs(np.median(sh)):
            lines.append("  !! 极差大于中位数本身 —— 单次回测数字不可作为决策依据")
            stable = False

        ex_spread = ex.max() - ex.min()
        lines.append(f"  平均仓位极差: {ex_spread:.1%}")
        if ex_spread > EXPOSURE_SPREAD_LIMIT:
            lines.append("  !! 平均仓位种子间差异过大 —— 择时行为不稳定")
            stable = False

    # 退化检测：所有种子都几乎空仓时，方差为 0（或极小）是**退化**的结果，
    # 不是「稳定」。空仓策略的 Sharpe/回撤恒为 0，方差自然是 0 —— 若不加这条，
    # 一个不学任何东西的策略会被判成「表现稳定」，恰好把结论说反。
    if n > 0 and ex.max() < 0.01:
        lines.append("  !! 所有种子平均仓位≈0 —— 策略退化为恒定空仓，"
                     "指标方差为 0 是退化所致，不代表稳定")
        stable = False

    lines += ["", "  结论: " + ("PPO 多种子表现稳定" if stable
                                else "PPO 单次结果不可信，上实盘前需更多样本")]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="PPO 多种子方差实验（不重训）")
    p.add_argument("--seeds", type=int, default=5, help="评估多少个种子")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-list", default=None,
                   help="逗号分隔的显式种子列表，覆盖 --seeds/--seed-start")
    p.add_argument("--market", default="csi1000")
    p.add_argument("--capital", type=float, default=500_000)
    p.add_argument("--scores", default=None,
                   help="选股分数面板路径，默认用本策略的 ALSTM 分数")
    p.add_argument("--report", default=str(paths.BACKTEST_DIR))
    args = p.parse_args()

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.datafeed.qlib_init import init_qlib
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    paths.ensure_dirs()
    # 特征数与模型观测空间绑定，必须与 train_ppo.py 一致（32，不是 ALSTM 的 360）
    init_qlib(str(Path(cfg.data.store_dir) / "qlib_data"), n_expressions=32)

    print("=" * 66)
    print("PPO 多种子实验（直接评估已有权重，不重训）")
    print("=" * 66)
    print(f"市场: {args.market}  本金: {args.capital:,.0f}  "
          f"测试: {TEST[0]}~{TEST[1]}")
    print(f"持仓: {HOLD_K} 只  调仓: 每 {REBAL_PERIOD} 日\n")

    t0 = time.time()
    close_df, feat_df, label_df = prepare_data(args.market)
    print(f"数据准备完成: {feat_df.shape[0]:,} 行 x {feat_df.shape[1]} 列")

    scores_path = Path(args.scores) if args.scores else paths.ALSTM_SCORES
    scores = None
    if scores_path.exists():
        scores = pd.read_parquet(scores_path)
        print(f"加载分数: {scores.shape[0]} 期 x {scores.shape[1]} 只")
    else:
        print("未找到分数，使用等权全市场")

    test_env = build_env(
        close_df, feat_df, label_df, TEST[0], TEST[1],
        initial_amount=args.capital, scores=scores)
    print(f"测试日数: {len(test_env.dates)}  State dim: {test_env.state_dim}\n")

    out_json = Path(args.report) / RESULT_JSON.name
    seeds = ([int(x) for x in args.seed_list.split(",") if x.strip()]
             if args.seed_list
             else list(range(args.seed_start, args.seed_start + args.seeds)))
    runs: list = []
    for i, seed in enumerate(seeds, 1):
        print(f"种子 {seed}  ({i}/{len(seeds)})")
        try:
            row = run_one_seed(seed, test_env)
            if row is None:
                continue
            runs.append(row)
            print(f"  总收益 {row['total_return']:+.2%}  "
                  f"回撤 {row['max_drawdown']:.2%}  "
                  f"Sharpe {row['sharpe']:+.3f}  "
                  f"平均仓位 {row['mean_exposure']:.1%}  "
                  f"({row['eval_sec']}s)")
        except Exception as e:  # noqa: BLE001 —— 单个种子失败不应中断整批
            print(f"  [FAIL] 种子 {seed}: {type(e).__name__}: {e}")

        # 每跑完一个种子就落盘，中途中断不丢已有结果
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "market": args.market, "capital": args.capital,
                "test": list(TEST), "hold_k": HOLD_K,
                "rebal_period": REBAL_PERIOD,
                "scores": str(scores_path), "device": "cpu",
            },
            "runs": runs,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(summarize(runs))
    print(f"\n结果 -> {out_json}")
    print(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
