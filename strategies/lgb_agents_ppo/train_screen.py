"""第 1 层：初筛 —— DoubleEnsemble 低过拟合设计。

## 为什么不是普通的 LightGBM

8 种子实验已经证明：同架构模型高度相关（方差比理论值差 2.1 倍），
集成因此压不下方差 —— 八个几乎相同的模型取平均，等于还是一个模型。

DoubleEnsemble 的两个机制正对这个问题：

**样本重加权（SR）** 按训练过程中的 loss 曲线给样本加权：已经拟合好的降权，
一直学不会的升权。A 股大部分样本是「涨跌都在 ±1% 以内」的噪声，
不加权时模型会被它们主导，学到的是噪声里的均值回复结构。

**特征筛选（FS）** 打乱某个特征列看表现掉多少，掉得少就剔除。
Alpha158 的 158 个因子大量互相冗余。更关键的是：不同子模型筛出不同的特征，
**造出的才是真正有差异的模型**，集成才有意义。

## 低过拟合不是一个参数，是一套准入检验

模型训练完不算数，必须过三关才能进下一层：

1. 多种子 Sharpe 分布收窄（不再横跨零轴）
2. 参数网格连片而非孤立尖刺
3. 分段检验符号一致

这三条是 ALSTM 全部没过的。达不到就说明这条路同样走不通，应及早止损。

用法::

    python strategies/lgb_agents_ppo/train_screen.py
    python strategies/lgb_agents_ppo/train_screen.py --no-sr --no-fs   # 对照组
    python strategies/lgb_agents_ppo/train_screen.py --seeds 5         # 多种子
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

TRAIN = ("2016-01-01", "2019-12-31")
VALID = ("2020-01-01", "2021-12-31")
TEST = ("2022-01-01", "2026-08-27")

#: 额外因子及其可获得滞后（交易日）。与 eval_factors.FACTOR_LAG 一致 ——
#: 因子按 trade_date 索引，那是数据「所属」日期而非「拿得到」日期，
#: 不滞后即为前视偏差
EXTRA_FACTORS = {
    "dragon_count_20": 1,
}


def build_dataset(market: str):
    """Alpha158 + 已验证的额外因子。"""
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    handler = Alpha158(
        instruments=market,
        start_time=TRAIN[0], end_time=TEST[1],
        fit_start_time=TRAIN[0], fit_end_time=TRAIN[1],
    )
    return DatasetH(handler, segments={
        "train": TRAIN, "valid": VALID, "test": TEST})


def attach_extra_factors(dataset, store_dir: str):
    """把额外因子拼进特征矩阵。

    返回拼接后的三段数据。Qlib 的 handler 不便直接扩展，
    因此在 prepare 之后手工对齐 —— 代价是要自己处理索引与缺失。
    """
    import pandas as pd

    parts = {}
    for seg in ("train", "valid", "test"):
        x = dataset.prepare(seg, col_set="feature")
        y = dataset.prepare(seg, col_set="label", data_key="raw")
        parts[seg] = (x, y)

    factor_dir = Path(store_dir) / "qlib_data" / "money_factors"
    for name, lag in EXTRA_FACTORS.items():
        p = factor_dir / f"{name}.parquet"
        if not p.exists():
            print(f"  [跳过] {name} 不存在")
            continue
        s = pd.read_parquet(p).iloc[:, 0]
        panel = s.unstack(level=1).sort_index().shift(lag)
        stacked = panel.stack()
        stacked.index.names = ["datetime", "instrument"]

        for seg, (x, y) in parts.items():
            aligned = stacked.reindex(x.index)
            # 计数型因子缺失即 0 次（该股当期未上榜），不是未知
            x[name] = aligned.fillna(0.0).values
        print(f"  [加入] {name}（滞后 {lag} 日）")

    return parts


def _cs_normalize(y):
    """标签截面标准化 —— 每个交易日内减均值除标准差。

    目的有二：一是让量纲与正则参数匹配（见 train_one 的说明），
    二是消除市场整体涨跌。模型该学的是「同一天里哪些股票更强」，
    而不是「哪天大盘涨」—— 后者是择时问题，由第 3 层的 PPO 负责。
    """
    col = y.columns[0]
    s = y[col]
    z = s.groupby(level="datetime").transform(
        lambda v: (v - v.mean()) / (v.std() + 1e-12))
    return z.to_frame(col)


def train_one(parts, seed: int, enable_sr: bool, enable_fs: bool,
              num_models: int, n_jobs: int) -> tuple:
    """训练一个 DoubleEnsemble，返回 (模型, 测试段预测)。"""
    import numpy as np
    import pandas as pd
    from qlib.contrib.model.double_ensemble import DEnsembleModel

    x_tr, y_tr = parts["train"]
    x_va, y_va = parts["valid"]
    x_te, _ = parts["test"]

    # 全空的特征列要剔除。Alpha158 的 VWAP0 依赖 $vwap 字段，
    # 而本项目导出的 qlib 数据只有 OHLCV+amount+factor，该列 100% 为空
    dead = [c for c in x_tr.columns if x_tr[c].isna().all()]
    if dead:
        x_tr = x_tr.drop(columns=dead)
        x_va = x_va.drop(columns=dead)
        x_te = x_te.drop(columns=dead)

    # 标签必须做截面标准化。下面的 lambda_l1/l2 是 Qlib 官方 Alpha158
    # 基准配置，配的是标准差约 1 的归一化标签；直接喂原始收益率
    # （标准差仅 0.03）会让正则项完全压垮信号 —— 实测 IC +0.00205、
    # t +0.54、特征重要性 gain 上限只有 6，模型 33 轮即早停等于没学到东西。
    # 标准化后同一套参数得到 IC +0.02820、t +7.43、gain 上限 2897。
    y_tr = _cs_normalize(y_tr)
    y_va = _cs_normalize(y_va)

    model = DEnsembleModel(
        base_model="gbm",
        num_models=num_models,
        enable_sr=enable_sr,
        enable_fs=enable_fs,
        # sample_ratios 控制每轮特征筛选后保留的比例，长度须等于 bins_fs
        sample_ratios=[0.8, 0.7, 0.6, 0.5, 0.4],
        sub_weights=[1.0] * num_models,
        # decay 必须显式给，Qlib 默认 None 会在 decay**k_th 处崩。
        # 它控制重加权强度随子模型序号衰减：越靠后的子模型权重越平，
        # 避免所有子模型都盯着同一批难样本 —— 那样又会退化成高度相关的集成
        decay=0.5,
        epochs=28,
        seed=seed,
        # LightGBM 超参：深度与叶子数压得较紧，这本身也是抗过拟合手段
        num_leaves=48, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        lambda_l1=205, lambda_l2=580,
        num_threads=n_jobs, verbosity=-1,
    )

    # DEnsembleModel.fit 要的是带 feature/label 两级列的 DataFrame
    def pack(x, y):
        df = pd.concat({"feature": x, "label": y}, axis=1)
        return df.dropna(subset=[("label", y.columns[0])])

    class _DS:
        """最小 dataset 适配器 —— DEnsembleModel 只调用 prepare()。"""

        def __init__(self, tr, va):
            self._d = {"train": tr, "valid": va}

        def prepare(self, segments, col_set="__all", data_key=None):
            if isinstance(segments, list):
                return [self._d[s] for s in segments]
            return self._d[segments]

    ds = _DS(pack(x_tr, y_tr), pack(x_va, y_va))
    model.fit(ds)

    pred = model.predict(_PredDS(x_te))
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    return model, pred


class _PredDS:
    """预测用的最小 dataset 适配器。

    注意与训练侧的差别：``fit`` 取的是 ``col_set="__all"``，要 feature/label
    两级列；而 ``predict`` 用 ``col_set="feature"``，拿到的应是**单层列**
    并直接按特征名索引（``x_test.loc[:, feat_sub]``）。
    包成两级会让子模型的特征名匹配不上，抛 KeyError。
    """

    def __init__(self, x):
        self._x = x

    def prepare(self, segments, col_set="__all", data_key=None):
        return self._x


def evaluate_ic(pred, label, fwd_days: int = 1) -> dict:
    """测试段 IC。t 值用 Newey-West 修正 —— 标签是未来收益，
    相邻观测重叠，直接算会虚高。"""
    import numpy as np
    import pandas as pd

    df = pd.concat([pred.rename("score"),
                    label.iloc[:, 0].rename("label")], axis=1).dropna()
    if df.empty:
        return {}
    ic = df.groupby(level="datetime").apply(
        lambda g: g["score"].corr(g["label"], method="spearman")
        if len(g) >= 30 else np.nan).dropna()
    if len(ic) < 10:
        return {}

    mean, sd = float(ic.mean()), float(ic.std(ddof=1))
    lag = max(fwd_days - 1, 1)
    x = ic.values - mean
    n = len(x)
    var = float(x @ x) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1 - k / (lag + 1.0)) * float(x[k:] @ x[:-k]) / n
    t = mean / (var / n) ** 0.5 if var > 0 else 0.0

    return {"ic_mean": round(mean, 5), "ic_std": round(sd, 5),
            "icir": round(mean / sd, 4) if sd else 0.0,
            "t": round(float(t), 3),
            "t_raw": round(mean / sd * n ** 0.5, 3) if sd else 0.0,
            "ic_positive": round(float((ic > 0).mean()), 4),
            "n_periods": int(len(ic))}


def to_panel(pred):
    """预测 -> 日期 x 标的 的宽表，列名转 vt 格式供回测使用。"""
    from qmtquant.datafeed.qlib_export import from_qlib_code

    panel = pred.unstack(level="instrument")
    if hasattr(panel, "columns") and panel.columns.nlevels > 1:
        panel.columns = panel.columns.droplevel(0)
    panel.columns = [from_qlib_code(str(c)) for c in panel.columns]
    return panel.sort_index()


def main() -> int:
    p = argparse.ArgumentParser(description="第 1 层：初筛训练")
    p.add_argument("--seeds", type=int, default=1,
                    help="训练几个种子。>1 时自动做稳健性检验")
    p.add_argument("--num-models", type=int, default=6,
                    help="DoubleEnsemble 的子模型数")
    p.add_argument("--no-sr", action="store_true", help="关闭样本重加权")
    p.add_argument("--no-fs", action="store_true", help="关闭特征筛选")
    p.add_argument("--n-jobs", type=int, default=0,
                    help="LightGBM 线程数，0 表示自动")
    p.add_argument("--rebuild-features", action="store_true",
                    help="强制重建特征缓存（改了因子定义后必须加）")
    args = p.parse_args()

    import numpy as np
    import pandas as pd

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.datafeed.qlib_init import init_qlib
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    params = paths.load_params()
    uri = str(Path(cfg.data.store_dir) / "qlib_data")
    # Alpha158 有 158 个表达式，缓存上限按并发峰值估算
    init_qlib(uri, n_expressions=158)

    n_jobs = args.n_jobs or max(1, (os.cpu_count() or 8) - 2)

    print("=" * 70)
    print("第 1 层：初筛（DoubleEnsemble 低过拟合设计）")
    print("=" * 70)
    print(f"市场      : {params['market']}")
    print(f"训练/验证 : {TRAIN[0]}~{TRAIN[1]} / {VALID[0]}~{VALID[1]}")
    print(f"测试      : {TEST[0]}~{TEST[1]}（样本外）")
    print(f"子模型数  : {args.num_models}")
    print(f"样本重加权: {'关闭' if args.no_sr else '开启'}")
    print(f"特征筛选  : {'关闭' if args.no_fs else '开启'}")
    print(f"种子数    : {args.seeds}")

    t0 = time.time()
    # 特征构建约 16 分钟且与种子无关，缓存后重跑只要几秒。
    # 缓存键含市场与额外因子 —— 改了因子必须重建，否则会用到旧特征
    cache = paths.MODELS_DIR / f"features_{params['market']}.pkl"
    key = {"market": params["market"], "extra": EXTRA_FACTORS,
           "segments": [list(TRAIN), list(VALID), list(TEST)]}
    parts = label_te = None
    if cache.exists() and not args.rebuild_features:
        import pickle
        try:
            with open(cache, "rb") as f:
                blob = pickle.load(f)
            if blob.get("key") == key:
                parts, label_te = blob["parts"], blob["label_te"]
                print(f"\n复用特征缓存: {cache.name}")
        except (OSError, pickle.PickleError, KeyError):
            parts = None

    if parts is None:
        print("\n构建 Alpha158 特征...")
        dataset = build_dataset(params["market"])
        parts = attach_extra_factors(dataset, cfg.data.store_dir)
        label_te = dataset.prepare("test", col_set="label", data_key="raw")
        paths.ensure_dirs()
        import pickle
        with open(cache, "wb") as f:
            pickle.dump({"key": key, "parts": parts, "label_te": label_te}, f)
        print(f"  已缓存: {cache}")

    x_tr = parts["train"][0]
    print(f"  train {x_tr.shape}  valid {parts['valid'][0].shape}"
          f"  test {parts['test'][0].shape}")
    print(f"  完成，耗时 {time.time() - t0:.0f}s")

    runs, panels = [], []
    for seed in range(args.seeds):
        print(f"\n--- 种子 {seed} ({seed + 1}/{args.seeds}) ---")
        t1 = time.time()
        model, pred = train_one(parts, seed, not args.no_sr, not args.no_fs,
                                args.num_models, n_jobs)
        ic = evaluate_ic(pred, label_te)
        elapsed = time.time() - t1
        runs.append({"seed": seed, **ic, "train_sec": round(elapsed)})
        panels.append(to_panel(pred))
        print(f"  IC {ic.get('ic_mean', 0):+.4f}  "
              f"t(NW) {ic.get('t', 0):+.2f}  "
              f"ICIR {ic.get('icir', 0):+.3f}  "
              f"[{elapsed:.0f}s]")

        if seed == 0:
            paths.ensure_dirs()
            import pickle
            with open(paths.SCREEN_MODEL, "wb") as f:
                pickle.dump(model, f)
            paths.SCREEN_META.write_text(json.dumps({
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "market": params["market"], "num_models": args.num_models,
                "enable_sr": not args.no_sr, "enable_fs": not args.no_fs,
                "train": list(TRAIN), "valid": list(VALID), "test": list(TEST),
                "n_features": int(x_tr.shape[1]),
                "extra_factors": EXTRA_FACTORS,
                "ic": ic,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 多种子稳健性
    if args.seeds > 1:
        ics = np.array([r.get("ic_mean", 0) for r in runs])
        ts = np.array([abs(r.get("t", 0)) for r in runs])
        print("\n" + "-" * 70)
        print("多种子稳健性")
        print("-" * 70)
        print(f"  IC     {ics.min():+.4f} ~ {ics.max():+.4f}   "
              f"中位数 {np.median(ics):+.4f}   标准差 {ics.std(ddof=1):.4f}")
        print(f"  |t|>=2 的次数: {int((ts >= 2).sum())}/{len(ts)}")
        if ics.min() > 0:
            print(f"  IC 全部为正 —— 信号方向稳定")
        else:
            print(f"  !! IC 有 {int((ics <= 0).sum())}/{len(ics)} 次非正")

    # ---- 集成分数：多种子截面排名平均
    print("\n合成分数面板...")
    if len(panels) == 1:
        scores = panels[0]
    else:
        ranked = [p.rank(axis=1, pct=True) for p in panels]
        stacked = pd.concat(ranked)
        scores = stacked.groupby(stacked.index).mean().sort_index()
    print(f"  {scores.shape[0]} 期 x {scores.shape[1]} 只")

    paths.ensure_dirs()
    scores.to_parquet(paths.SCREEN_SCORES)
    print(f"  已保存: {paths.SCREEN_SCORES}")

    # ---- 候选池
    k = params["candidate_k"]
    cand = {}
    for d, row in scores.iterrows():
        top = row.dropna().nlargest(k)
        cand[d] = list(top.index)
    cand_df = pd.DataFrame(
        [(d, i + 1, s) for d, syms in cand.items()
         for i, s in enumerate(syms)],
        columns=["date", "rank", "vt_symbol"])
    cand_df.to_parquet(paths.CANDIDATES, index=False)
    print(f"  候选池: 每期 Top-{k}，共 {len(cand_df):,} 行"
          f" -> {paths.CANDIDATES}")

    paths.ROBUSTNESS.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"seeds": args.seeds, "num_models": args.num_models,
                   "enable_sr": not args.no_sr, "enable_fs": not args.no_fs,
                   "market": params["market"], "test": list(TEST)},
        "runs": runs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"明细: {paths.ROBUSTNESS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
