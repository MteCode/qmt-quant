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
                 exchanges: list[str] | None = None,
                 feature_cols: list[str] | None = None) -> pd.DataFrame:
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

    # === 全市场过滤规则 ===
    # 1. 排除北交所（BSE）—— 流动性差、T+1 限制不同
    # 2. 排除科创板（688xxx.SH）—— 门槛 50 万，散户无法交易
    # 高股价过滤（1 手 > 5 万）在读取数据后按最新收盘价过滤
    ex_dirs = [d for d in ex_dirs if d.name != "BSE"]

    files = []
    for ex in ex_dirs:
        for f in sorted(ex.glob("*.parquet")):
            code = f.stem
            # 科创板：688xxx
            if code.startswith("688"):
                continue
            files.append((f, f"{code}.{ex.name}"))

    if max_symbols and len(files) > max_symbols:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(files), max_symbols, replace=False)
        files = [files[i] for i in sorted(idx)]

    print(f"  收集 {len(files)} 只标的的特征...")
    chunks = []
    skipped = 0
    n_symbols = 0
    batch = []
    t0 = time.time()
    BATCH = 200

    import gc
    # 每个标的：计算特征→转 float32→只保留 features+symbol+label 列
    keep_cols = list(feature_cols or []) + ["close"]

    def _process_batch(batch):
        # 逐个：转 float32 + 只保留需要的列（在 concat 前处理省内存）
        processed = []
        for b in batch:
            cols = [c for c in keep_cols if c in b.columns]
            sub = b[cols].copy()
            for c in sub.columns:
                if sub[c].dtype == np.float64:
                    sub[c] = sub[c].astype(np.float32)
            processed.append(sub)
        return pd.concat(processed, axis=0)

    MAX_PRICE = 500.0  # 1 手 > 5 万的过滤掉

    for i, (path, vt) in enumerate(files, 1):
        feat = compute_features_for_symbol(path, vt)
        if feat is None or len(feat) < min_bars:
            skipped += 1
            continue
        # 过滤高股价：最新收盘价 > 500 元（1 手 > 5 万）
        if "close" in feat.columns and feat["close"].iloc[-1] > MAX_PRICE:
            skipped += 1
            continue
        batch.append(feat)
        n_symbols += 1
        if len(batch) >= BATCH:
            chunks.append(_process_batch(batch))
            batch.clear()
            gc.collect()
        if i % 200 == 0:
            print(f"    {i}/{len(files)} ({time.time()-t0:.0f}s)")

    if batch:
        chunks.append(_process_batch(batch))
        batch.clear()
        gc.collect()

    if not chunks:
        raise ValueError("无足够数据，请先下载并清洗 1m 行情")

    df = pd.concat(chunks, axis=0)
    del chunks; gc.collect()
    print(f"  完成：{n_symbols} 只标的，{len(df):,} 条样本，"
          f"跳过 {skipped} 只（数据不足），耗时 {time.time()-t0:.0f}s")
    return df


def train(df: pd.DataFrame, features: list[str],
          horizon: int, threshold: float,
          params: dict) -> dict:
    """训练 GBM 模型并返回结果。"""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    # 标签：未来 horizon bar 收益率是否超过阈值
    close = df["close"].values.astype(np.float64)
    future_ret = np.empty(len(close), dtype=np.float64)
    future_ret[:] = np.nan
    future_ret[:-horizon] = close[horizon:] / close[:-horizon] - 1
    y_all = (future_ret > threshold).astype(np.int8)
    has_label = ~np.isnan(future_ret)
    del close, future_ret

    # 按日期分割：前 70% 的天数做训练
    day_vals = df.index.normalize()
    days = day_vals.unique().sort_values()
    cut = days[max(1, int(len(days) * 0.70)) - 1]
    is_train_all = np.asarray(day_vals <= cut)

    # 合并 mask：有标签 & (训练 or 验证)
    mask = has_label
    idx_mask = np.where(mask)[0]
    is_train = is_train_all[mask]

    train_idx = idx_mask[is_train]
    val_idx = idx_mask[~is_train]

    sym_arr = None
    dt_idx = df.index.copy()
    yt = y_all[train_idx]
    yv = y_all[val_idx]

    # 训练集过大时随机采样（16GB RAM 下 ~20M 行是上限）
    MAX_TRAIN = 20_000_000
    if len(train_idx) > MAX_TRAIN:
        rng = np.random.RandomState(42)
        keep = rng.choice(len(train_idx), MAX_TRAIN, replace=False)
        keep.sort()
        train_idx = train_idx[keep]
        yt = yt[keep]
        print(f"  训练集过大，随机采样 {MAX_TRAIN:,} 条")

    # 验证集也采样（省内存）
    MAX_VAL = 5_000_000
    if len(val_idx) > MAX_VAL:
        rng_v = np.random.RandomState(99)
        keep_v = rng_v.choice(len(val_idx), MAX_VAL, replace=False)
        keep_v.sort()
        val_idx = val_idx[keep_v]
        yv = yv[keep_v]

    print(f"\n  训练集 {len(train_idx):,} 条 ({yt.mean():.1%} 正类)")
    print(f"  验证集 {len(val_idx):,} 条 ({yv.mean():.1%} 正类)")
    print(f"  日期分割 训练 ≤ {cut.strftime('%Y-%m-%d')}，"
          f"验证 > {cut.strftime('%Y-%m-%d')}")

    # 直接从 DataFrame 逐列构建 Xt/Xv，不创建完整 X 矩阵
    import gc
    nf = len(features)
    Xt = np.empty((len(train_idx), nf), dtype=np.float32)
    Xv = np.empty((len(val_idx), nf), dtype=np.float32)
    for j, f in enumerate(features):
        col = df[f].values
        Xt[:, j] = col[train_idx]
        Xv[:, j] = col[val_idx]
    del df; gc.collect()

    model = LGBMClassifier(**params)
    t0 = time.time()
    model.fit(Xt, yt)
    train_time = time.time() - t0
    print(f"  训练耗时 {train_time:.0f}s")

    prob = model.predict_proba(Xv)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = float(accuracy_score(yv, pred))
    auc = float(roc_auc_score(yv, prob)) if len(np.unique(yv)) > 1 else None

    # 训练集指标（子采样）
    tr_sample = min(200000, len(train_idx))
    tr_si = np.random.RandomState(0).choice(len(train_idx), tr_sample, replace=False)
    prob_tr = model.predict_proba(Xt[tr_si])[:, 1]
    pred_tr = (prob_tr >= 0.5).astype(int)
    acc_tr = float(accuracy_score(yt[tr_si], pred_tr))
    auc_tr = float(roc_auc_score(yt[tr_si], prob_tr)) if len(np.unique(yt[tr_si])) > 1 else None

    metrics = {
        "model": "LightGBMClassifier",
        "horizon_bars": horizon,
        "threshold": threshold,
        "n_features": len(features),
        "features": features,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(val_idx)),
        "train_accuracy": acc_tr,
        "train_auc": auc_tr,
        "test_accuracy": acc,
        "test_auc": auc,
        "train_date_range": [str(dt_idx[train_idx[0]]),
                             str(dt_idx[train_idx[-1]])],
        "test_date_range": [str(dt_idx[val_idx[0]]),
                            str(dt_idx[val_idx[-1]])],
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
    n_sample = min(10000, len(val_idx))
    si = np.random.RandomState(42).choice(len(val_idx), n_sample, replace=False)
    si.sort()
    sym_vals = (sym_arr[val_idx[si]]
                if sym_arr is not None
                else np.full(n_sample, ""))
    predictions = pd.DataFrame({
        "datetime": dt_idx[val_idx[si]],
        "symbol": sym_vals,
        "prob_up": prob[si],
        "label": yv[si],
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
    p.add_argument("--max-symbols", type=int, default=3500,
                   help="最多使用多少只标的（默认 3500，32GB RAM 上限）")
    p.add_argument("--min-bars", type=int, default=1000,
                   help="标的至少有多少根 bar 才纳入训练")
    p.add_argument("--output", default=str(MODEL_DIR),
                   help="模型输出目录")
    p.add_argument("--grid-search", action="store_true",
                   help="网格搜索多组参数，选最优模型")
    args = p.parse_args()

    from qmtquant.config import get_config
    from qmtquant.features.intraday import INTRADAY_FEATURES

    cfg = get_config()
    store = Path(cfg.data.store_dir)

    print("=" * 62)
    if args.grid_search:
        print("  全市场日内 GBM —— 参数网格搜索")
    else:
        print("  全市场日内 GBM 模型训练")
        print(f"  预测窗口 {args.horizon} bar，正类阈值 {args.threshold:.4f}")
    print("=" * 62)

    # 收集数据（只加载一次）
    df = collect_data(
        store / "clean" if (store / "clean" / "1m").exists() else store,
        max_symbols=args.max_symbols,
        min_bars=args.min_bars,
        feature_cols=INTRADAY_FEATURES,
    )

    if args.grid_search:
        # 参数网格
        horizons = [5, 10, 15, 20, 30]
        thresholds = [0.0003, 0.0005, 0.001, 0.002]
        gbm_configs = [
            {"label": "base",
             "n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31,
             "max_depth": 6, "min_child_samples": 100},
            {"label": "deep",
             "n_estimators": 600, "learning_rate": 0.02, "num_leaves": 63,
             "max_depth": 8, "min_child_samples": 200},
            {"label": "wide",
             "n_estimators": 800, "learning_rate": 0.01, "num_leaves": 127,
             "max_depth": 6, "min_child_samples": 50},
        ]
        common = {
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.5, "reg_lambda": 2.0,
            "objective": "binary", "random_state": 42,
            "verbosity": -1, "n_jobs": -1,
        }

        import gc

        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        progress_path = out / "grid_search_progress.csv"

        results = []
        total = len(horizons) * len(thresholds) * len(gbm_configs)
        run = 0
        for h in horizons:
            for th in thresholds:
                for gcfg in gbm_configs:
                    run += 1
                    label = gcfg.get("label", "")
                    params = {k: v for k, v in gcfg.items() if k != "label"}
                    params.update(common)
                    print(f"\n--- [{run}/{total}] horizon={h} threshold={th} "
                          f"config={label} ---", flush=True)
                    try:
                        r = train(df, INTRADAY_FEATURES, h, th, params)
                        m = r["metrics"]
                        results.append({
                            "horizon": h, "threshold": th, "config": label,
                            "test_acc": m["test_accuracy"],
                            "test_auc": m["test_auc"],
                            "train_acc": m["train_accuracy"],
                            "train_auc": m["train_auc"],
                            "result": r,
                        })
                        print(f"    AUC={m['test_auc']:.4f}  "
                              f"Acc={m['test_accuracy']:.4f}", flush=True)
                        # 每轮落盘，崩溃也不丢已完成的结果
                        pd.DataFrame([
                            {k: v for k, v in x.items() if k != "result"}
                            for x in results
                        ]).to_csv(progress_path, index=False)
                    except Exception as e:
                        print(f"    失败: {e}", flush=True)
                    gc.collect()

        if not results:
            print("所有组合都失败了")
            return 1

        # 按验证集 AUC 排序
        results.sort(key=lambda x: x["test_auc"] or 0, reverse=True)

        print(f"\n{'=' * 72}")
        print("  参数搜索结果（按验证集 AUC 降序）")
        print(f"{'=' * 72}")
        print(f"  {'Rank':<5} {'Horizon':<8} {'Thresh':<8} {'Config':<6} "
              f"{'Test AUC':<10} {'Test Acc':<10} {'Train AUC':<10}")
        print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
        for i, r in enumerate(results[:15], 1):
            print(f"  {i:<5} {r['horizon']:<8} {r['threshold']:<8.4f} "
                  f"{r['config']:<6} {r['test_auc']:<10.4f} "
                  f"{r['test_acc']:<10.4f} {r['train_auc']:<10.4f}")

        # 保存最优模型
        best = results[0]
        br = best["result"]
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": br["model"],
            "features": br["features"],
            "horizon": br["horizon"],
            "threshold": br["threshold"],
            "params": br["params"],
        }, out / "model.joblib")
        (out / "metrics.json").write_text(
            json.dumps(br["metrics"], ensure_ascii=False, indent=2),
            encoding="utf-8")
        br["importance"].to_csv(out / "feature_importance.csv", index=False)
        br["predictions"].to_csv(out / "predictions_sample.csv", index=False)

        # 保存全部搜索结果
        grid_df = pd.DataFrame([{k: v for k, v in r.items() if k != "result"}
                                for r in results])
        grid_df.to_csv(out / "grid_search_results.csv", index=False)

        print(f"\n  最优模型: horizon={best['horizon']} "
              f"threshold={best['threshold']} config={best['config']}")
        print(f"  验证集 AUC={best['test_auc']:.4f} Acc={best['test_acc']:.4f}")
        print(f"  已保存到: {out}")
    else:
        params = {
            "n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31,
            "max_depth": 6, "min_child_samples": 100,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.5, "reg_lambda": 2.0,
            "objective": "binary", "random_state": 42,
            "verbosity": -1, "n_jobs": -1,
        }
        result = train(df, INTRADAY_FEATURES, args.horizon, args.threshold,
                       params)

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
        result["predictions"].to_csv(
            out / "predictions_sample.csv", index=False)

        m = result["metrics"]
        print(f"\n{'=' * 62}")
        print(f"  训练完成")
        print(f"  验证集 accuracy {m['test_accuracy']:.4f}  "
              f"AUC {m['test_auc']:.4f}")
        print(f"  训练集 accuracy {m['train_accuracy']:.4f}  "
              f"AUC {m['train_auc']:.4f}")
        print(f"\n  模型保存在: {out}")
        print(f"  特征重要性 Top 5:")
        for _, row in result["importance"].head(5).iterrows():
            print(f"    {row['feature']:<20s} {int(row['importance']):>6d}")

    print(f"\n下一步：")
    print(f"  python scripts/predict_intraday.py   # 盘中全市场预测")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
