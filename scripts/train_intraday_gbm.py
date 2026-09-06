"""全市场日内 GBM 模型训练 —— 从 1m K 线学习哪些票适合日内交易。

## 与 strategies/intraday_t_920368/train_gbm.py 的区别

那个脚本是**单票**（920368.BSE）的日内做T模型，用 9 个基础特征预测
单只股票的短期方向。

本脚本是**全市场**日内模型：
1. 读取全市场清洗后的 1m 数据（data/clean/1m/）
2. 用 24 维日内特征（qmtquant/features/intraday.py）
3. 训练的是"哪些票在当前时刻适合日内操作"的横截面选股模型
4. 输出全市场预测概率，供下游策略（做T、打板、均值回归等）筛选标的

## 训练策略

- **标签**：未来 N 根 bar 的收益率是否超过阈值（默认 10 bar, 5bp）
- **样本**：全市场所有标的的所有分钟 bar，按日期 70/30 分割
- **特征**：24 维日内因子（动量、波动、微观结构、时间位置）
- **模型**：LightGBM 二分类，对全市场统一训练

## 输出

    models/intraday_gbm/
        model.joblib          训练好的模型 + 特征列表 + 超参
        metrics.json          训练集/验证集指标
        feature_importance.csv 特征重要性排名
        predictions_sample.csv 验证集预测样本（用于人工检查）

## 用法

    python scripts/train_intraday_gbm.py
    python scripts/train_intraday_gbm.py --horizon 15 --threshold 0.001
    python scripts/train_intraday_gbm.py --max-symbols 500
    python scripts/train_intraday_gbm.py --min-bars 1000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "intraday_gbm"


def collect_data(clean_dir: Path, max_symbols: int, min_bars: int,
                 exchanges: list[str] | None = None) -> pd.DataFrame:
    """从清洗层收集全市场 1m 数据并计算特征。"""
    from qmtquant.features.intraday import compute_features_for_symbol

    src = clean_dir / "1m"
    if not src.exists():
        # fallback 到原始层
        src = clean_dir.parent / "1m"
    if not src.exists():
        raise FileNotFoundError(f"找不到 1m 数据：{src}")

    ex_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    if exchanges:
        ex_dirs = [d for d in ex_dirs if d.name in exchanges]

    files = []
    for ex in ex_dirs:
        for f in sorted(ex.glob("*.parquet")):
            files.append((f, f"{f.stem}.{ex.name}"))

    if max_symbols and len(files) > max_symbols:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(files), max_symbols, replace=False)
        files = [files[i] for i in sorted(idx)]

    print(f"  收集 {len(files)} 只标的的特征...")
    all_feat = []
    skipped = 0
    t0 = time.time()

    for i, (path, vt) in enumerate(files, 1):
        feat = compute_features_for_symbol(path, vt)
        if feat is None or len(feat) < min_bars:
            skipped += 1
            continue
        all_feat.append(feat)
        if i % 200 == 0:
            print(f"    {i}/{len(files)} ({time.time()-t0:.0f}s)")

    if not all_feat:
        raise ValueError("无足够数据，请先下载并清洗 1m 行情")

    df = pd.concat(all_feat, axis=0)
    print(f"  完成：{len(all_feat)} 只标的，{len(df):,} 条样本，"
          f"跳过 {skipped} 只（数据不足），耗时 {time.time()-t0:.0f}s")
    return df


def train(df: pd.DataFrame, features: list[str],
          horizon: int, threshold: float,
          params: dict) -> dict:
    """训练 GBM 模型并返回结果。"""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    from qmtquant.features.intraday import make_labels

    # 用原始 close 列做标签
    close_df = pd.DataFrame({"close": df["close"]}, index=df.index)
    y = make_labels(close_df, horizon=horizon, threshold=threshold)
    valid = y.index[df.index.isin(y.index)]

    # 按日期分割：前 70% 的天数做训练，后 30% 做验证
    days = pd.Index(df.index.normalize()).unique().sort_values()
    cut = days[max(1, int(len(days) * 0.70)) - 1]
    is_train = df.index.normalize() <= cut

    # 去掉标签为 NaN 的尾部（shift 造成的）
    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    has_label = future_ret.notna()

    mask = has_label
    x = df.loc[mask, features]
    y = y.loc[mask]
    is_train = is_train[mask]

    xt, xv = x[is_train], x[~is_train]
    yt, yv = y[is_train], y[~is_train]

    print(f"\n  训练集 {len(xt):,} 条 ({yt.mean():.1%} 正类)")
    print(f"  验证集 {len(xv):,} 条 ({yv.mean():.1%} 正类)")
    print(f"  日期分割 训练 ≤ {cut.strftime('%Y-%m-%d')}，"
          f"验证 > {cut.strftime('%Y-%m-%d')}")

    model = LGBMClassifier(**params)
    t0 = time.time()
    model.fit(xt, yt)
    train_time = time.time() - t0
    print(f"  训练耗时 {train_time:.0f}s")

    prob = model.predict_proba(xv)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = float(accuracy_score(yv, pred))
    auc = float(roc_auc_score(yv, prob)) if yv.nunique() > 1 else None

    # 训练集指标
    prob_tr = model.predict_proba(xt)[:, 1]
    pred_tr = (prob_tr >= 0.5).astype(int)
    acc_tr = float(accuracy_score(yt, pred_tr))
    auc_tr = float(roc_auc_score(yt, prob_tr)) if yt.nunique() > 1 else None

    metrics = {
        "model": "LightGBMClassifier",
        "horizon_bars": horizon,
        "threshold": threshold,
        "n_features": len(features),
        "features": features,
        "train_samples": int(len(xt)),
        "test_samples": int(len(xv)),
        "train_accuracy": acc_tr,
        "train_auc": auc_tr,
        "test_accuracy": acc,
        "test_auc": auc,
        "train_date_range": [str(xt.index.min()), str(xt.index.max())],
        "test_date_range": [str(xv.index.min()), str(xv.index.max())],
        "train_positive_rate": float(yt.mean()),
        "test_positive_rate": float(yv.mean()),
        "train_time_seconds": round(train_time, 1),
        "params": params,
    }

    # 特征重要性
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # 验证集预测样本
    sample_idx = np.random.RandomState(42).choice(
        len(xv), min(10000, len(xv)), replace=False)
    predictions = pd.DataFrame({
        "datetime": xv.index[sample_idx],
        "symbol": df.loc[xv.index[sample_idx], "symbol"].values
                  if "symbol" in df.columns else "",
        "prob_up": prob[sample_idx],
        "label": yv.values[sample_idx],
    })

    return {
        "model": model,
        "features": features,
        "horizon": horizon,
        "threshold": threshold,
        "params": params,
        "metrics": metrics,
        "importance": importance,
        "predictions": predictions,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="全市场日内 GBM 模型训练")
    p.add_argument("--horizon", type=int, default=10,
                   help="预测窗口（bar 数，默认 10，即 10 分钟）")
    p.add_argument("--threshold", type=float, default=0.0005,
                   help="正类阈值（默认 5bp）")
    p.add_argument("--max-symbols", type=int, default=0,
                   help="最多使用多少只标的（0=全部，用于快速实验）")
    p.add_argument("--min-bars", type=int, default=1000,
                   help="标的至少有多少根 bar 才纳入训练")
    p.add_argument("--output", default=str(MODEL_DIR),
                   help="模型输出目录")
    args = p.parse_args()

    from qmtquant.config import get_config
    from qmtquant.features.intraday import INTRADAY_FEATURES

    cfg = get_config()
    store = Path(cfg.data.store_dir)

    print("=" * 62)
    print("  全市场日内 GBM 模型训练")
    print(f"  预测窗口 {args.horizon} bar，正类阈值 {args.threshold:.4f}")
    print("=" * 62)

    # 收集数据
    df = collect_data(
        store / "clean" if (store / "clean" / "1m").exists() else store,
        max_symbols=args.max_symbols,
        min_bars=args.min_bars,
    )

    # GBM 超参
    params = {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "objective": "binary",
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    result = train(df, INTRADAY_FEATURES, args.horizon, args.threshold, params)

    # 保存
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": result["model"],
        "features": result["features"],
        "horizon": result["horizon"],
        "threshold": result["threshold"],
        "params": result["params"],
    }, out / "model.joblib")

    (out / "metrics.json").write_text(
        json.dumps(result["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8")
    result["importance"].to_csv(out / "feature_importance.csv", index=False)
    result["predictions"].to_csv(out / "predictions_sample.csv", index=False)

    m = result["metrics"]
    print(f"\n{'=' * 62}")
    print(f"  训练完成")
    print(f"  验证集 accuracy {m['test_accuracy']:.4f}  AUC {m['test_auc']:.4f}")
    print(f"  训练集 accuracy {m['train_accuracy']:.4f}  AUC {m['train_auc']:.4f}")
    print(f"\n  模型保存在: {out}")
    print(f"  特征重要性 Top 5:")
    for _, row in result["importance"].head(5).iterrows():
        print(f"    {row['feature']:<20s} {int(row['importance']):>6d}")

    print(f"\n下一步：")
    print(f"  python scripts/predict_intraday.py   # 盘中全市场预测")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
