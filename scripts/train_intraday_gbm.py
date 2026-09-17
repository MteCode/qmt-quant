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
RUNS_DIR = ROOT / "strategies" / "intraday_gbm" / "runs"


def collect_data(clean_dir: Path, max_symbols: int, min_bars: int,
                 exchanges: list[str] | None = None,
                 feature_cols: list[str] | None = None) -> pd.DataFrame:
    """从清洗层收集全市场 1m 数据并计算特征。"""
    from qmtquant.datafeed.adjust import real_price  # noqa: E402
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
    # 保留 symbol：标签要靠它识别跨标的边界（见 train() 里的屏蔽逻辑）。
    # 存成 category，270M 行下只多几十 MB，而没有它就只能漏掉那部分
    # 垃圾标签 —— 量小，但没有理由留着。
    keep_cols = list(feature_cols or []) + ["close", "symbol"]

    def _process_batch(batch):
        # 逐个：转 float32 + 只保留需要的列（在 concat 前处理省内存）
        processed = []
        for b in batch:
            cols = [c for c in keep_cols if c in b.columns]
            sub = b[cols].copy()
            for c in sub.columns:
                if sub[c].dtype == np.float64:
                    sub[c] = sub[c].astype(np.float32)
            if "symbol" in sub.columns:
                sub["symbol"] = sub["symbol"].astype("category")
            processed.append(sub)
        return pd.concat(processed, axis=0)

    MAX_PRICE = 500.0  # 真实价 > 500 即 1 手 > 5 万
    #: 反推不出真实价、只能按后复权价过滤的标的数。
    #: 不为零说明这批标的的过滤仍可能误判。
    n_no_factor = 0

    for i, (path, vt) in enumerate(files, 1):
        feat = compute_features_for_symbol(path, vt)
        if feat is None or len(feat) < min_bars:
            skipped += 1
            continue
        # 过滤高股价：1 手 > 5 万，即**真实价** > 500 元。
        #
        # 原先写的是 `feat["close"].iloc[-1] > MAX_PRICE`，而 close 是
        # **后复权价**。复权因子从 1.6 到 280 不等，于是这条过滤把
        # 海尔智家（21 元）、新和成（26 元）、北方稀土（40 元）这些
        # 便宜的老蓝筹排除了 —— 它们只是分红送转多、因子高。
        # 实测 500 只里误排 11 只，而真正该排的只有 1 只。
        #
        # 模型因此从一开始就没见过这批标的。
        px = real_price(feat)
        if px is None:
            px = float(feat["close"].iloc[-1])   # 退化：仍按后复权价
            n_no_factor += 1
        if px > MAX_PRICE:
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


def make_labels(close: np.ndarray, day_codes: np.ndarray,
                sym_codes: np.ndarray | None, horizon: int,
                threshold: float) -> tuple:
    """算标签，并屏蔽跨日/跨标的的越界取值。

    抽成独立函数是为了能测 —— train() 要 GB 级数据和一个模型才能跑，
    边界情况在那个粒度上覆盖不到，而边界正是这里唯一会出错的地方。

    ## 屏蔽什么

    df 是全市场拼成的长表，按标的分块（concat 不排序），块内按时间。
    所以 close[i + horizon] 有两种取错：

    1. **跨标的**：每只最后 horizon 行取到下一只股票的价格。占比 0.004%，
       但标的间价格量级差很多，算出的「收益率」是几百倍的异常值，
       不是噪声。

    2. **跨日**：每日最后 horizon 根取到次日价格，「未来收益」跨越隔夜
       跳空。horizon=15 时占 6.20%（241 根/日）。而策略是尾盘强平不留
       隔夜的 —— 这些样本教模型去学一段它永远不可能交易的收益，
       且系统性地集中在尾盘，正是该平仓的时候。

    :returns: (y, has_label, n_dropped)
    """
    n = len(close)
    fut = np.empty(n, dtype=np.float64)
    fut[:] = np.nan
    fut[:-horizon] = close[horizon:] / close[:-horizon] - 1

    same_day = np.zeros(n, dtype=bool)
    same_day[:-horizon] = day_codes[horizon:] == day_codes[:-horizon]

    if sym_codes is None:
        same_sym = np.ones(n, dtype=bool)
    else:
        same_sym = np.zeros(n, dtype=bool)
        same_sym[:-horizon] = sym_codes[horizon:] == sym_codes[:-horizon]

    labelled = ~np.isnan(fut)
    has_label = labelled & same_day & same_sym
    y = (fut > threshold).astype(np.int8)
    return y, has_label, int(labelled.sum() - has_label.sum())


def make_labels_t1(close: np.ndarray, day_codes: np.ndarray,
                   sym_codes: np.ndarray | None, threshold: float,
                   exit_at: str = "open") -> tuple:
    """T+1 标签：当日买入，次一交易日卖出。

    ## 为什么要有这个函数

    A 股 T+1，当天买入的股票当天卖不掉。日内标签（make_labels）刻意屏蔽了
    跨日样本，因为那种策略尾盘强平；而 T+1 策略的持有期**必然跨越隔夜**，
    所以这里的取值方向和它正好相反 —— 跨日不是要屏蔽的越界，而是标签本身。

    实盘印证过这件事：日内版每分钟都在发卖单，每分钟都被
    「可卖数量不足」拒掉，连止损单也执行不了。

    ## 出场取在哪一根

    - ``open``：次日第一根 bar。T+1 一开盘就能卖，是最早的合法出场，
      持有隔夜风险最短，也最容易在实盘复现
    - ``close``：次日最后一根 bar，等于多持有一整天

    「次一交易日」用数据里该标的的下一个日期块判定，不按自然日加一 ——
    停牌、节假日都会让自然日算法取错。

    ## 屏蔽什么

    仍然要屏蔽跨标的：长表按标的分块拼接，每只最后一天的「次日」会落到
    下一只股票头上。两只股票价格量级不同，算出的收益率是异常值不是噪声。

    :returns: (y, has_label, n_dropped)
    """
    n = len(close)
    if n == 0:
        return (np.zeros(0, dtype=np.int8), np.zeros(0, dtype=bool), 0)

    sym = (np.zeros(n, dtype=np.int64) if sym_codes is None
           else np.asarray(sym_codes))

    # (标的, 日期) 变化处即新块的起点
    new_block = np.empty(n, dtype=bool)
    new_block[0] = True
    new_block[1:] = (day_codes[1:] != day_codes[:-1]) | (sym[1:] != sym[:-1])

    block_id = np.cumsum(new_block) - 1
    block_start = np.flatnonzero(new_block)
    block_end = np.r_[block_start[1:], n] - 1
    block_sym = sym[block_start]
    nb = len(block_start)

    # 下一块必须是同一标的，否则「次日」跨到了别的股票
    exit_row = np.full(nb, -1, dtype=np.int64)
    if nb > 1:
        nxt = block_start[1:] if exit_at == "open" else block_end[1:]
        same = block_sym[1:] == block_sym[:-1]
        exit_row[:-1] = np.where(same, nxt, -1)

    row_exit = exit_row[block_id]
    has = row_exit >= 0

    fut = np.full(n, np.nan, dtype=np.float64)
    fut[has] = close[row_exit[has]] / close[has] - 1

    labelled = ~np.isnan(fut)
    has_label = labelled & has
    y = (fut > threshold).astype(np.int8)
    return y, has_label, int(n - has_label.sum())


def train(df: pd.DataFrame, features: list[str],
          horizon: int, threshold: float,
          params: dict, label_mode: str = "intraday",
          exit_at: str = "open") -> dict:
    """训练 GBM 模型并返回结果。

    label_mode:
      - ``intraday``：未来 horizon 根 bar 的收益，屏蔽跨日（尾盘强平的策略）
      - ``t1``：当日买入、次一交易日卖出（A 股 T+1，持有期必然跨夜）
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    # 标签与边界屏蔽见 make_labels() / make_labels_t1()
    close = df["close"].values.astype(np.float64)
    day_codes = pd.factorize(df.index.normalize())[0]
    sym_codes = (pd.factorize(df["symbol"].values)[0]
                 if "symbol" in df.columns else None)
    if sym_codes is None:
        print("  [!] 无 symbol 列，跨标的边界无法屏蔽（约 0.004% 的行）")

    if label_mode == "t1":
        y_all, has_label, n_drop = make_labels_t1(
            close, day_codes, sym_codes, threshold, exit_at)
        print(f"  标签：T+1（次日{'开盘' if exit_at == 'open' else '收盘'}卖出）"
              f"，无次日可用 {n_drop:,} 行"
              f"（{n_drop / max(1, len(close)):.2%}）")
    else:
        y_all, has_label, n_drop = make_labels(
            close, day_codes, sym_codes, horizon, threshold)
        print(f"  标签：屏蔽跨日/跨标的越界 {n_drop:,} 行 "
              f"（{n_drop / max(1, len(close)):.2%}）")
    del close, day_codes, sym_codes

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
        # 口径必须跟着模型走 —— 日内模型和 T+1 模型的输出含义完全不同，
        # 装错了策略照样能跑、照样出信号，只是全错，且没有任何报错
        "label_mode": label_mode,
        "exit_at": exit_at if label_mode == "t1" else None,
        "horizon_bars": horizon if label_mode != "t1" else None,
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
        "label_mode": label_mode,
        "exit_at": exit_at,
        "params": params,
        "metrics": metrics,
        "importance": importance,
        "predictions": predictions,
    }


def _save_run(result: dict, out: Path, label: str = "") -> str:
    """保存训练结果到 models/ 目录，同时注册到 runs/ 供模型管理发现。"""
    from webui.model_registry import create_manifest, save_manifest, generate_run_id

    m = result["metrics"]
    run_id = generate_run_id("lgbm")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": result["model"],
        "features": result["features"],
        "horizon": result["horizon"],
        "threshold": result["threshold"],
        # 策略加载时据此校验口径是否匹配 —— 缺这个字段的是旧的日内模型
        "label_mode": result.get("label_mode", "intraday"),
        "exit_at": result.get("exit_at"),
        "params": result["params"],
    }, run_dir / "model.joblib")

    (run_dir / "metrics.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    result["importance"].to_csv(run_dir / "feature_importance.csv", index=False)
    result["predictions"].to_csv(run_dir / "predictions_sample.csv", index=False)

    manifest = create_manifest(
        run_id=run_id,
        model_type="LightGBMClassifier",
        strategy_id="intraday_gbm",
        params={**result["params"],
                # 口径进 params，管理台列表页才看得到这个模型是哪种标签训的。
                # 日内模型和 T+1 模型混在一个列表里，不标出来就会装错
                "label_mode": result.get("label_mode", "intraday"),
                "exit_at": result.get("exit_at"),
                "horizon_bars": result["horizon"],
                "threshold": result["threshold"]},
        metrics={
            "label_mode": result.get("label_mode", "intraday"),
            "test_accuracy": m["test_accuracy"],
            "test_auc": m["test_auc"],
            "train_accuracy": m["train_accuracy"],
            "train_auc": m["train_auc"],
            "n_features": m["n_features"],
            "train_samples": m["train_samples"],
            "test_samples": m["test_samples"],
        },
        artifacts={
            "model": "model.joblib",
            "metrics_json": "metrics.json",
            "feature_importance": "feature_importance.csv",
            "predictions_sample": "predictions_sample.csv",
        },
        train_period=[m["train_date_range"][0][:10], m["train_date_range"][1][:10]],
        test_period=[m["test_date_range"][0][:10], m["test_date_range"][1][:10]],
        train_time_sec=m.get("train_time_seconds"),
        notes=label,
    )
    save_manifest(run_dir, manifest)

    # 同步到 models/ 目录供策略加载
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in ("model.joblib", "metrics.json", "feature_importance.csv",
              "predictions_sample.csv"):
        shutil.copy2(run_dir / f, out / f)

    return run_id


def main() -> int:
    p = argparse.ArgumentParser(description="全市场日内 GBM 模型训练")
    p.add_argument("--label-mode", default="intraday",
                   choices=["intraday", "t1"],
                   help="intraday=未来N根bar收益（尾盘强平的策略）；"
                        "t1=当日买入次日卖出（A股T+1，默认策略该用这个）")
    p.add_argument("--exit-at", default="open", choices=["open", "close"],
                   help="t1 模式的出场口径：次日开盘或次日收盘")
    p.add_argument("--horizon", type=int, default=10,
                   help="预测窗口（bar 数，默认 10，即 10 分钟）。"
                        "t1 模式下不使用")
    p.add_argument("--threshold", type=float, default=0.0005,
                   help="正类阈值（默认 5bp）。t1 模式的收益量级远大于日内，"
                        "阈值应相应调高")
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

        label = (f"grid best: horizon={best['horizon']} "
                 f"threshold={best['threshold']} config={best['config']}")
        run_id = _save_run(br, out, label)

        # 保存全部搜索结果
        grid_df = pd.DataFrame([{k: v for k, v in r.items() if k != "result"}
                                for r in results])
        grid_df.to_csv(out / "grid_search_results.csv", index=False)

        print(f"\n  最优模型: horizon={best['horizon']} "
              f"threshold={best['threshold']} config={best['config']}")
        print(f"  验证集 AUC={best['test_auc']:.4f} Acc={best['test_acc']:.4f}")
        print(f"  已保存到: {out}")
        print(f"  模型管理 run_id: {run_id}")
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
                       params, args.label_mode, args.exit_at)

        out = Path(args.output)
        if args.label_mode == "t1":
            label = (f"T+1 exit={args.exit_at} threshold={args.threshold}")
        else:
            label = f"horizon={args.horizon} threshold={args.threshold}"
        run_id = _save_run(result, out, label)

        m = result["metrics"]
        print(f"\n{'=' * 62}")
        print(f"  训练完成")
        print(f"  验证集 accuracy {m['test_accuracy']:.4f}  "
              f"AUC {m['test_auc']:.4f}")
        print(f"  训练集 accuracy {m['train_accuracy']:.4f}  "
              f"AUC {m['train_auc']:.4f}")
        print(f"\n  模型保存在: {out}")
        print(f"  模型管理 run_id: {run_id}")
        print(f"  特征重要性 Top 5:")
        for _, row in result["importance"].head(5).iterrows():
            print(f"    {row['feature']:<20s} {int(row['importance']):>6d}")

    print(f"\n下一步：")
    print(f"  python scripts/predict_intraday.py   # 盘中全市场预测")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
