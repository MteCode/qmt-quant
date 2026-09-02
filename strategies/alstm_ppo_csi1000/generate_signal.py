"""每日信号生成 —— ALSTM 选股 + PPO 择时。

收盘后运行，输出目标持仓文件 (target_portfolio.csv)，
供虚拟盘/实盘读取执行。

流程：
1. 从 Qlib 本地数据提取最新因子
2. 加载 ALSTM 模型生成选股分数
3. 加载 PPO 模型决定仓位水平
4. 输出目标持仓：股票列表 + 等权权重 × 仓位水平

用法::

    python strategies/alstm_ppo_csi1000/generate_signal.py
    python strategies/alstm_ppo_csi1000/generate_signal.py --date 2026-09-01
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402  （必须在 sys.path 设好后再导入）

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

PARAMS = paths.load_params()
HOLD_K = PARAMS["hold_k"]


def get_market_features(market: str, end_date: str, lookback: int = 60):
    """提取最近 lookback 天的市场截面特征（供 PPO state 用）。"""
    from qlib.data import D

    start = (pd.Timestamp(end_date) - pd.Timedelta(days=lookback * 2)).strftime("%Y-%m-%d")
    instruments = D.instruments(market)

    factor_exprs = [
        "$close/Ref($close,1)-1",
        "$close/Ref($close,5)-1",
        "$close/Ref($close,20)-1",
        "Std($close,5)/Mean($close,5)",
        "Std($close,20)/Mean($close,20)",
        "$volume/Ref($volume,1)-1",
        "Mean($volume,5)/Mean($volume,20)-1",
        "($high-$low)/$close",
        "($close-$low)/($high-$low+1e-8)",
        "Mean($close,5)/Mean($close,20)-1",
    ]
    factor_names = [
        "ret1", "mom5", "mom20", "vol5", "vol20",
        "vol_chg", "vol_ratio", "amplitude", "close_pos", "ma_dev",
    ]

    feat_df = D.features(instruments, factor_exprs,
                         start_time=start, end_time=end_date)
    feat_df.columns = factor_names

    # 加载大资金因子
    from qmtquant.config import get_config
    cfg = get_config()
    factor_dir = Path(cfg.data.store_dir) / "qlib_data" / "money_factors"
    dragon_path = factor_dir / "dragon_count_20.parquet"
    if dragon_path.exists():
        s = pd.read_parquet(dragon_path).iloc[:, 0].swaplevel().sort_index()
        aligned = s.reindex(feat_df.index)
        feat_df["dragon_count_20"] = aligned.values

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

    # 截面 ZScore
    feat_df = feat_df.groupby(level="datetime").transform(
        lambda x: ((x - x.mean()) / (x.std() + 1e-8)).clip(-3, 3)
    ).fillna(0)

    return feat_df


def get_alstm_scores(date: str) -> pd.Series:
    """从已保存的 ALSTM 分数面板取当日分数。

    实盘时需要每日重新推理，此处先用离线分数演示流程。
    """
    scores_path = paths.ALSTM_SCORES
    if not scores_path.exists():
        print("  !! ALSTM 分数文件不存在")
        return pd.Series(dtype=float)

    scores = pd.read_parquet(scores_path)
    ts = pd.Timestamp(date)

    # asof 取最近分数
    pos = scores.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        print(f"  !! 无可用分数（日期 {date} 早于分数起始）")
        return pd.Series(dtype=float)

    row = scores.iloc[pos].dropna().sort_values(ascending=False)
    print(f"  ALSTM 分数日期: {scores.index[pos].date()}, {len(row)} 只有分数")
    return row


def get_ppo_exposure(feat_df, date: str) -> float:
    """用 PPO 模型预测当日仓位水平。"""
    model_path = paths.PPO_MODEL
    if not model_path.exists():
        print("  !! PPO 模型不存在，默认仓位 30%")
        return 0.30

    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")

    # 构建 state：市场聚合特征 + 组合状态
    ts = pd.Timestamp(date)
    if ts in feat_df.index.get_level_values("datetime"):
        day_feat = feat_df.xs(ts, level="datetime")
    else:
        dates = feat_df.index.get_level_values("datetime").unique()
        closest = dates[dates <= ts]
        if len(closest) == 0:
            print("  !! 无可用特征数据")
            return 0.30
        day_feat = feat_df.xs(closest[-1], level="datetime")

    n_features = day_feat.shape[1]
    means = day_feat.mean().values.astype(np.float32)
    stds = day_feat.std().values.astype(np.float32)
    # 组合状态（初始状态：无持仓）
    port_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state = np.concatenate([means, stds, port_state])

    action, _ = model.predict(state, deterministic=True)
    exposure = float(np.clip(action[0], 0, 1))
    return exposure


def generate_signal(date: str, market: str = "csi1000",
                    capital: float = 500_000) -> pd.DataFrame:
    """生成目标持仓。"""
    print(f"\n{'='*50}")
    print(f"信号生成: {date}")
    print(f"{'='*50}")

    # 1. ALSTM 选股
    print("\n[1/3] ALSTM 选股...")
    scores = get_alstm_scores(date)
    if scores.empty:
        print("  无信号，跳过")
        return pd.DataFrame()

    top_stocks = scores.head(HOLD_K)
    print(f"  选出 Top-{len(top_stocks)} 只")

    # 2. PPO 择时
    print("\n[2/3] PPO 择时...")
    feat_df = get_market_features(market, date)
    exposure = get_ppo_exposure(feat_df, date)
    print(f"  PPO 建议仓位: {exposure:.1%}")

    # 3. 构建目标持仓
    print("\n[3/3] 构建目标持仓...")

    from qmtquant.datafeed.qlib_export import to_qlib_code
    target = []
    equal_weight = 1.0 / len(top_stocks)

    for vt_symbol, score in top_stocks.items():
        try:
            qlib_code = to_qlib_code(str(vt_symbol))
        except ValueError:
            continue

        weight = equal_weight * exposure
        target_value = capital * weight

        target.append({
            "vt_symbol": str(vt_symbol),
            "score": round(float(score), 6),
            "weight": round(weight, 4),
            "target_value": round(target_value, 2),
        })

    df = pd.DataFrame(target)
    if df.empty:
        print("  无目标持仓")
        return df

    print(f"\n  持仓只数: {len(df)}")
    print(f"  总仓位: {df['weight'].sum():.1%}")
    print(f"  总金额: {df['target_value'].sum():,.0f} 元")
    print(f"  单只均值: {df['target_value'].mean():,.0f} 元")

    return df


def main():
    p = argparse.ArgumentParser(description="每日信号生成")
    p.add_argument("--date", default=None,
                    help="信号日期，默认今天")
    p.add_argument("--market", default=PARAMS["market"])
    p.add_argument("--capital", type=float, default=PARAMS["capital"])
    args = p.parse_args()

    date = args.date or datetime.now().strftime("%Y-%m-%d")

    from qmtquant.config import get_config
    cfg = get_config()
    uri = str(Path(cfg.data.store_dir) / "qlib_data")

    from qmtquant.datafeed.qlib_init import init_qlib
    init_qlib(uri, n_expressions=32)

    df = generate_signal(date, args.market, args.capital)
    if df.empty:
        return 1

    # 保存
    paths.ensure_dirs()
    out_file = paths.signal_file(date)
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {out_file}")

    # 同时保存一份 latest
    df.to_csv(paths.LATEST_SIGNAL, index=False, encoding="utf-8-sig")
    print(f"已更新: {paths.LATEST_SIGNAL}")

    print(f"\n前 10 只:")
    for _, row in df.head(10).iterrows():
        print(f"  {row['vt_symbol']:>12s}  "
              f"分数 {row['score']:+.4f}  "
              f"权重 {row['weight']:.2%}  "
              f"{row['target_value']:>10,.0f} 元")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
