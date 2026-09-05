"""每日信号 —— LightGBM 选股 + PPO 择时。

## 与 alstm_ppo_csi1000 的差别

选股模型换成树模型（第 1 层），其余不变。换的理由是稳定性：

              ALSTM(8种子)   LightGBM(5种子)
    IC 中位数    +0.0183        +0.0257
    IC 标准差     0.0120         0.0007      小 17 倍
    出现负 IC     1/8            无
    单次训练     10 分钟         90 秒

神经网络 5 万参数从随机初始化出发，每次收敛到不同局部解；
树模型的分裂点由数据决定，随机性只来自采样。

## 三层怎么串

1. **初筛**：读 `models/screen_scores.parquet`，取 Top-K 候选
2. **精研**：TradingAgents 逻辑验真 —— 因前视问题暂未实施，
   当前直接透传（见 DESIGN.md）
3. **择时**：PPO 给出仓位水平

## 择时层当前不可用

五种子实验显示 PPO 在清洗后的数据上一致输出 0 仓位 ——
训练段（2016-2019 中证 1000）满仓亏 35.29%，在负期望环境里
学出「永远空仓」是理性决策，不是 bug。

因此本脚本默认用固定仓位（`--exposure`），并明确打印该状态。
等训练环境改成相对基准的超额收益口径后再启用 PPO。

用法::

    python strategies/lgb_agents_ppo/generate_signal.py
    python strategies/lgb_agents_ppo/generate_signal.py --exposure 0.3
    python strategies/lgb_agents_ppo/generate_signal.py --use-ppo
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import paths  # noqa: E402


def load_scores():
    """第 1 层的打分面板。"""
    import pandas as pd

    if not paths.SCREEN_SCORES.exists():
        return None
    return pd.read_parquet(paths.SCREEN_SCORES).sort_index()


def pick(scores, date: str, k: int):
    """取某日 Top-K。日期不在面板里时用最近的一期。"""
    import pandas as pd

    ts = pd.Timestamp(date)
    pos = scores.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None, None
    row = scores.iloc[pos].dropna().sort_values(ascending=False)
    return row.head(k), scores.index[pos]


def ppo_exposure(date: str, market: str) -> float | None:
    """PPO 建议的仓位。模型不存在或环境未就绪时返回 None。"""
    model_path = paths.MODELS_DIR / "ppo_model.zip"
    if not model_path.exists():
        return None
    try:
        import numpy as np
        from stable_baselines3 import PPO

        # 状态构造与 train_ppo 的 TimingEnv 必须一致，
        # 否则模型读到的是另一套语义。此处复用该策略的实现
        sys.path.insert(0, str(paths.ROOT / "strategies" / "alstm_ppo_csi1000"))
        import train_ppo as T

        close_df, feat_df, label_df = T.prepare_data(market)
        env = T.build_env(close_df, feat_df, label_df, date, date,
                          initial_amount=paths.load_params()["capital"])
        obs, _ = env.reset()
        model = PPO.load(str(model_path), device="cpu")
        action, _ = model.predict(obs, deterministic=True)
        return float(np.clip(action[0], 0.0, 1.0))
    except Exception as e:
        print(f"  PPO 推理失败（{type(e).__name__}），改用固定仓位")
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="LightGBM 选股 + 择时")
    p.add_argument("--date", default=None, help="信号日期，默认今天")
    p.add_argument("--exposure", type=float, default=0.30,
                    help="固定仓位水平。PPO 不可用时使用")
    p.add_argument("--use-ppo", action="store_true",
                    help="启用 PPO 择时。当前它在清洗后数据上一致输出 0 仓位")
    p.add_argument("--hold-k", type=int, default=None,
                    help="持仓只数，默认读 strategy.yaml")
    args = p.parse_args()

    import pandas as pd

    from qmtquant.config import get_config
    from qmtquant.datafeed.qlib_init import init_qlib

    cfg = get_config()
    params = paths.load_params()
    date = args.date or datetime.now().strftime("%Y-%m-%d")
    hold_k = args.hold_k or params["hold_k"]
    capital = params["capital"]

    print("=" * 58)
    print(f"信号生成  {date}")
    print(f"  选股 LightGBM 集成 · 持仓 {hold_k} 只 · 本金 {capital:,.0f}")
    print("=" * 58)

    print("\n[1/3] 初筛 —— LightGBM 打分")
    scores = load_scores()
    if scores is None:
        print("  缺少分数面板，请先运行 train_screen.py")
        return 1
    top, used = pick(scores, date, hold_k)
    if top is None or top.empty:
        print(f"  无可用分数（{date} 早于面板起始）")
        return 1
    print(f"  分数日期 {used.date()}（面板 {scores.shape[0]} 期 x "
          f"{scores.shape[1]} 只）")
    print(f"  选出 Top-{len(top)}")

    print("\n[2/3] 精研 —— TradingAgents")
    print("  未启用：LLM 读研报做历史回测有无法修正的前视偏差，")
    print("  且研报数据源尚未接入。候选池直接透传。见 DESIGN.md")

    print("\n[3/3] 择时 —— 仓位水平")
    exposure = None
    if args.use_ppo:
        init_qlib(str(Path(cfg.data.store_dir) / "qlib_data"),
                  n_expressions=32)
        exposure = ppo_exposure(date, params["market"])
        if exposure is not None:
            print(f"  PPO 建议仓位 {exposure:.1%}")
            if exposure < 0.01:
                print("  [注意] PPO 输出接近零仓位。这是它在负期望训练环境")
                print("         下的理性结果，非故障。详见 DESIGN.md")
    if exposure is None:
        exposure = args.exposure
        print(f"  使用固定仓位 {exposure:.1%}"
              + ("" if args.use_ppo else "（未启用 PPO）"))

    if exposure <= 0:
        print("\n仓位为 0，不生成持仓")
        return 0

    # ---- 组合：等权
    rows = []
    w = exposure / len(top)
    for vt, score in top.items():
        rows.append({
            "vt_symbol": str(vt),
            "score": round(float(score), 6),
            "weight": round(w, 6),
            "target_value": round(capital * w, 2),
        })
    df = pd.DataFrame(rows)

    paths.ensure_dirs()
    out = paths.SIGNALS_DIR / f"target_{date}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    latest = paths.SIGNALS_DIR / "target_latest.csv"
    df.to_csv(latest, index=False, encoding="utf-8-sig")

    print(f"\n持仓 {len(df)} 只   总仓位 {df['weight'].sum():.1%}   "
          f"金额 {df['target_value'].sum():,.0f} 元")
    per = df["target_value"].iloc[0]
    print(f"单只 {per:,.0f} 元", end="")
    if per < 10_000:
        print(f"  [注意] 股价高于 {per / 100:.0f} 元的买不满一手，下单时会被跳过")
    else:
        print()

    print(f"\n前 10 只:")
    for _, r in df.head(10).iterrows():
        print(f"  {r['vt_symbol']:>12s}  分数 {r['score']:+.4f}  "
              f"{r['target_value']:>10,.0f} 元")

    print(f"\n已保存: {out}")
    print(f"        {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
