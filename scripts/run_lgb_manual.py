"""手动构建特征矩阵 + LightGBM —— 完全控制输入因子。

Alpha158（158 列）+ 大资金因子（龙虎榜、融资融券）直接拼接，
用 lightgbm 原生 API 训练，不走 Qlib Model 封装。

用法::

    python scripts/run_lgb_manual.py
    python scripts/run_lgb_manual.py --capital 500000 --holdings 30
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

TRAIN = ("2016-01-01", "2019-12-31")
VALID = ("2020-01-01", "2021-12-31")
TEST = ("2022-01-01", "2026-08-27")


def get_alpha158_exprs():
    """Alpha158 全部因子表达式"""
    from qlib.contrib.data.handler import Alpha158DL

    conf = {
        "kbar": {},
        "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
        "rolling": {},
    }
    fields, names = Alpha158DL.get_feature_config(conf)
    return names, fields


def build_features(market: str) -> tuple:
    """构建完整特征矩阵：Alpha158 + 大资金因子"""
    import numpy as np
    import pandas as pd
    from qlib.data import D

    from qmtquant.config import get_config
    cfg = get_config()
    qlib_dir = Path(cfg.data.store_dir) / "qlib_data"

    instruments = D.instruments(market)

    # ---- Alpha158 ----
    print("提取 Alpha158 特征...")
    names, fields = get_alpha158_exprs()
    t0 = time.time()
    feat_df = D.features(
        instruments, fields,
        start_time=TRAIN[0], end_time=TEST[1],
    )
    feat_df.columns = names
    print(f"  {feat_df.shape[0]:,} 行 x {feat_df.shape[1]} 列, "
          f"耗时 {time.time() - t0:.0f}s")

    # ---- 标签：次日收益率 ----
    print("提取标签（次日收益率）...")
    label_df = D.features(
        instruments, ["Ref($close,-1)/$close-1"],
        start_time=TRAIN[0], end_time=TEST[1],
    )
    label = label_df.iloc[:, 0]
    label.name = "label"

    # ---- 大资金因子 ----
    print("加载大资金因子...")
    factor_dir = qlib_dir / "money_factors"
    money_cols = []

    for name in ["dragon_count_20", "margin_bal_chg"]:
        path = factor_dir / f"{name}.parquet"
        if not path.exists():
            print(f"  {name}: 文件不存在，跳过")
            continue
        s = pd.read_parquet(path).iloc[:, 0]
        # (datetime, instrument) -> swap to (instrument, datetime) 匹配 Qlib
        s = s.swaplevel().sort_index()
        s.name = name
        # 对齐到 feat_df 的 index
        aligned = s.reindex(feat_df.index)
        feat_df[name] = aligned.values
        money_cols.append(name)
        n_valid = aligned.notna().sum()
        print(f"  {name}: {n_valid:,} 有效值 "
              f"({n_valid / len(feat_df):.1%} 覆盖)")

    print(f"\n最终特征矩阵: {feat_df.shape[0]:,} 行 x {feat_df.shape[1]} 列")
    print(f"  Alpha158: {len(names)} 列")
    print(f"  大资金:   {len(money_cols)} 列 ({', '.join(money_cols)})")

    return feat_df, label


def train_lgb(feat_df, label):
    """用 LightGBM 原生 API 训练"""
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    # 按时间切分
    idx = feat_df.index.get_level_values("datetime")
    train_mask = (idx >= TRAIN[0]) & (idx <= TRAIN[1])
    valid_mask = (idx >= VALID[0]) & (idx <= VALID[1])
    test_mask = (idx >= TEST[0]) & (idx <= TEST[1])

    X_train = feat_df[train_mask].copy()
    y_train = label[train_mask].reindex(X_train.index)
    X_valid = feat_df[valid_mask].copy()
    y_valid = label[valid_mask].reindex(X_valid.index)
    X_test = feat_df[test_mask].copy()
    y_test = label[test_mask].reindex(X_test.index)

    # 去掉标签为 NaN / inf / 极端值的行
    def clean_label(X, y):
        mask = y.notna() & np.isfinite(y) & (y.abs() < 0.2)
        return X[mask], y[mask]

    X_train, y_train = clean_label(X_train, y_train)
    X_valid, y_valid = clean_label(X_valid, y_valid)

    # 替换 inf 为 NaN，再标准化
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_valid = X_valid.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # ZScore 标准化（只用训练集统计量）
    print("ZScore 标准化（训练集 fit）...")
    train_mean = X_train.mean()
    train_std = X_train.std()
    train_std[train_std < 1e-8] = 1.0
    X_train = (X_train - train_mean) / train_std
    X_valid = (X_valid - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # clip 极端值 + 填充 NaN
    X_train = X_train.clip(-5, 5).fillna(0)
    X_valid = X_valid.clip(-5, 5).fillna(0)
    X_test = X_test.clip(-5, 5).fillna(0)

    print(f"\n训练集: {len(X_train):,}  验证集: {len(X_valid):,}  "
          f"测试集: {len(X_test):,}")

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "mse",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
    }

    print("训练 LightGBM ...")
    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100),
    ]
    t0 = time.time()
    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=callbacks,
    )
    print(f"  完成，{model.best_iteration} 轮, "
          f"耗时 {time.time() - t0:.0f}s")

    # 特征重要度 top 20
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=X_train.columns,
    ).sort_values(ascending=False)
    print("\nTop 20 特征重要度:")
    for i, (name, val) in enumerate(importance.head(20).items(), 1):
        print(f"  {i:2d}. {name:<20} {val:>10.0f}")

    # 测试集预测
    pred = pd.Series(model.predict(X_test), index=X_test.index, name="score")

    return pred, y_test, model, importance


def evaluate_ic(pred, label) -> dict:
    import numpy as np
    import pandas as pd

    df = pd.concat([pred.rename("score"), label.rename("label")],
                   axis=1).dropna()
    if df.empty:
        return {}
    ic = df.groupby(level="datetime").apply(
        lambda g: g["score"].corr(g["label"], method="spearman")
        if len(g) >= 30 else np.nan).dropna()
    if len(ic) < 3:
        return {}
    mean, sd = float(ic.mean()), float(ic.std(ddof=1))
    return {
        "IC均值": mean,
        "IC标准差": sd,
        "ICIR": mean / sd if sd else 0.0,
        "t值": mean / sd * np.sqrt(len(ic)) if sd else 0.0,
        "IC>0占比": float((ic > 0).mean()),
        "横截面数": len(ic),
    }


def to_score_panel(pred):
    from qmtquant.datafeed.qlib_export import from_qlib_code
    panel = pred.unstack(level="instrument")
    if hasattr(panel, "columns") and panel.columns.nlevels > 1:
        panel.columns = panel.columns.droplevel(0)
    panel.columns = [from_qlib_code(str(c)) for c in panel.columns]
    return panel.sort_index()


def main() -> int:
    p = argparse.ArgumentParser(
        description="手动特征矩阵 + LightGBM")
    p.add_argument("--uri", default=None)
    p.add_argument("--market", default="csi1000")
    p.add_argument("--index", default="000852.SH")
    p.add_argument("--holdings", type=int, default=30)
    p.add_argument("--rebalance", type=int, default=20)
    p.add_argument("--capital", type=float, default=500_000)
    p.add_argument("--report", default="reports/lgb_manual")
    p.add_argument("--no-drawdown", action="store_true")
    args = p.parse_args()

    import pandas as pd

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    uri = args.uri or str(Path(cfg.data.store_dir) / "qlib_data")

    import qlib
    qlib.init(provider_uri=uri, region="cn", joblib_backend="threading")

    print("=" * 62)
    print("手动特征矩阵 + LightGBM（Alpha158 + 大资金因子）")
    print("=" * 62)
    print(f"市场: {args.market}  本金: {args.capital:,.0f}")
    print(f"持仓: {args.holdings} 只  调仓: 每 {args.rebalance} 日")
    print(f"训练: {TRAIN[0]}~{TRAIN[1]}  验证: {VALID[0]}~{VALID[1]}")
    print(f"测试: {TEST[0]}~{TEST[1]} (样本外)\n")

    t0 = time.time()

    feat_df, label = build_features(args.market)
    pred, y_test, model, importance = train_lgb(feat_df, label)

    print(f"\n  {len(pred):,} 条预测")

    print("\n" + "-" * 62)
    print("测试段 IC（样本外）")
    print("-" * 62)
    ic = evaluate_ic(pred, y_test)
    if not ic:
        print("  横截面不足")
        return 1
    for k, v in ic.items():
        print(f"  {k:<10} {v:>10.4f}" if isinstance(v, float)
              else f"  {k:<10} {v:>10}")
    if abs(ic["t值"]) < 2:
        print("  !! |t| < 2，信号不显著")

    # ---------- 组合回测 ----------
    print("\n" + "-" * 62)
    print("组合回测（T+1 / 涨跌停 / 整手 / 成本 / 回撤控制）")
    print("-" * 62)

    scores = to_score_panel(pred)
    print(f"分数面板: {scores.shape[0]} 期 x {scores.shape[1]} 只")

    from qmtquant.core.constants import Interval
    from qmtquant.datafeed.xt_feed import IndexFeed, XtDataFeed
    from qmtquant.engine.backtest_engine import BacktestEngine
    from qmtquant.report.html_report import build_report
    from qmtquant.risk.drawdown import DrawdownConfig, DrawdownController
    from qmtquant.strategy.signal_rank import SignalRankStrategy
    from qmtquant.universe.providers import HistoricalUniverse

    weight_csv = (Path(cfg.data.store_dir) / "universe"
                  / f"index_weight_{args.index}.csv")
    universe = HistoricalUniverse(str(weight_csv))
    symbols = universe.all_symbols()

    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    bars = feed.load_bars(symbols, TEST[0], TEST[1], Interval.DAILY)
    if not bars:
        print("测试段没有行情数据")
        return 1
    print(f"装载 {len(bars):,} 根 K 线")

    drawdown = None
    if not args.no_drawdown:
        r = cfg.risk
        drawdown = DrawdownController(DrawdownConfig(
            close_only_threshold=r.drawdown_close_only,
            reduce_threshold=r.drawdown_reduce,
            reduce_keep_ratio=r.drawdown_reduce_keep,
            flat_threshold=r.drawdown_flat,
            recovery_ratio=r.drawdown_recovery_ratio,
            min_observations=r.drawdown_min_observations,
            max_freeze_observations=r.drawdown_max_freeze))

    engine = BacktestEngine(initial_capital=args.capital, cost=cfg.cost,
                            drawdown=drawdown)
    engine.load_data(bars)
    engine.set_universe(universe)
    engine.add_strategy(SignalRankStrategy, symbols, {
        "max_holdings": args.holdings, "rebalance_days": args.rebalance})
    engine.strategy.scores = scores.sort_index()
    engine.strategy._score_dates = engine.strategy.scores.index.to_numpy()

    stats = engine.run()
    print()
    print(stats.summary())
    print()
    print(universe.describe_bias().summary())

    # ---- 基准
    tr = Path(cfg.data.store_dir) / "index" / f"{args.index}_tr.parquet"
    if tr.exists():
        d = pd.read_parquet(tr)
        bench = (d.assign(trade_date=pd.to_datetime(d["trade_date"]))
                  .set_index("trade_date")["close"].sort_index())
        bench = bench[(bench.index >= TEST[0]) & (bench.index <= TEST[1])]
        kind = "全收益"
    else:
        bench = IndexFeed(cfg.data.store_dir).load_close(
            args.index, TEST[0], TEST[1])
        kind = "!! 价格指数"

    if not bench.empty:
        bench_ret = bench.iloc[-1] / bench.iloc[0] - 1
        print(f"\n基准 {args.index} {kind}: {bench_ret:+.2%}")
        print(f"策略超额: {stats.total_return - bench_ret:+.2%}")

    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    engine.get_equity_df().to_csv(out / "equity.csv", encoding="utf-8-sig")
    tdf = engine.get_trades_df()
    if not tdf.empty:
        tdf.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    scores.to_parquet(out / "scores.parquet")
    pd.Series(ic).to_csv(out / "test_ic.csv", encoding="utf-8-sig")
    importance.to_csv(out / "feature_importance.csv", encoding="utf-8-sig")

    report = build_report(
        engine, stats, out / "lgb_manual_report.html",
        title="Alpha158+大资金 手动特征 LightGBM",
        subtitle=(f"{args.market} · 本金{args.capital/1e4:.0f}万 · "
                  f"持仓{args.holdings}只 · 每{args.rebalance}日调仓 · "
                  f"测试{TEST[0]}~{TEST[1]}(样本外) · "
                  f"IC {ic['IC均值']:.4f} t={ic['t值']:.2f}"),
        benchmark=bench if not bench.empty else None)

    print(f"\n明细: {out.resolve()}")
    print(f"报告: {report.resolve()}")
    print(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
