"""Alpha158 + 大资金因子 + LightGBM 增强版。

在 Alpha158（158 个量价因子）基础上加入：
- dragon_count_20 : 龙虎榜20日累计上榜次数（IC=-0.028, t=-18.13）
- margin_bal_chg  : 融资余额日变化率（IC=-0.024, t=-6.41）

用法::

    python scripts/run_lgb_enhanced.py --market csi1000 --index 000852.SH
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


def load_money_factors(qlib_dir: Path, market: str):
    """加载大资金因子并对齐到 Qlib 格式"""
    import pandas as pd
    from qlib.data import D

    instruments = D.instruments(market)

    factor_dir = qlib_dir / "money_factors"
    factors = {}

    for name in ["dragon_count_20", "margin_bal_chg"]:
        path = factor_dir / f"{name}.parquet"
        if not path.exists():
            print(f"  跳过 {name}: 文件不存在")
            continue
        df = pd.read_parquet(path)
        s = df.iloc[:, 0]
        # index 已经是 (datetime, instrument) qlib code 格式
        # 需要 swaplevel 成 (instrument, datetime) 匹配 Qlib
        s = s.swaplevel().sort_index()
        factors[name] = s
        print(f"  {name}: {len(s):,} 条")

    return factors


def build_enhanced_dataset(uri: str, market: str, money_factors: dict):
    """Alpha158 + 大资金因子"""
    import pandas as pd
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    handler = Alpha158(
        instruments=market,
        start_time=TRAIN[0], end_time=TEST[1],
        fit_start_time=TRAIN[0], fit_end_time=TRAIN[1],
    )

    dataset = DatasetH(handler, segments={
        "train": TRAIN, "valid": VALID, "test": TEST,
    })

    # 获取原始特征 DataFrame 并追加大资金因子
    for seg in ["train", "valid", "test"]:
        df = dataset.prepare(seg, col_set="feature", data_key="raw")
        if df is None or df.empty:
            continue
        for name, series in money_factors.items():
            # 对齐到 df 的 index
            aligned = series.reindex(df.index)
            df[(name,)] = aligned.values if hasattr(aligned, 'values') else aligned

    return dataset


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
    p = argparse.ArgumentParser(description="Alpha158 + 大资金 + LightGBM")
    p.add_argument("--uri", default=None)
    p.add_argument("--market", default="csi1000")
    p.add_argument("--index", default="000852.SH")
    p.add_argument("--holdings", type=int, default=50)
    p.add_argument("--rebalance", type=int, default=20)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--report", default="reports/lgb_enhanced")
    p.add_argument("--no-drawdown", action="store_true")
    args = p.parse_args()

    import pandas as pd
    import numpy as np

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    uri = args.uri or str(Path(cfg.data.store_dir) / "qlib_data")
    if not Path(uri).exists():
        print(f"数据目录不存在: {uri}")
        return 1

    import qlib
    qlib.init(provider_uri=uri, region="cn", joblib_backend="threading")

    print("=" * 62)
    print("Alpha158 + 大资金因子 + LightGBM（增强版）")
    print("=" * 62)
    print(f"数据      : {uri}  市场 {args.market}")
    print(f"训练      : {TRAIN[0]} ~ {TRAIN[1]}")
    print(f"验证      : {VALID[0]} ~ {VALID[1]}（早停）")
    print(f"测试      : {TEST[0]} ~ {TEST[1]}  <- 样本外\n")

    t0 = time.time()

    # 加载大资金因子
    print("加载大资金因子...")
    money_factors = load_money_factors(Path(uri), args.market)

    # 构建增强数据集：Alpha158 作为基础 handler，手动追加大资金因子
    print("\n构建 Alpha158 特征...")
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    handler = Alpha158(
        instruments=args.market,
        start_time=TRAIN[0], end_time=TEST[1],
        fit_start_time=TRAIN[0], fit_end_time=TRAIN[1],
    )
    dataset = DatasetH(handler, segments={
        "train": TRAIN, "valid": VALID, "test": TEST,
    })

    # 在 handler 层面追加大资金因子
    # 获取处理后的完整数据，手动追加列后重建 dataset
    print("合并大资金因子到特征矩阵...")

    for seg in ["train", "valid", "test"]:
        df_feat = dataset.prepare(seg, col_set="feature",
                                   data_key=DataHandlerLP.DK_L)
        df_label = dataset.prepare(seg, col_set="label",
                                    data_key=DataHandlerLP.DK_L)
        if df_feat is None or df_feat.empty:
            continue

        added = 0
        for name, series in money_factors.items():
            aligned = series.reindex(df_feat.index).fillna(0)
            df_feat[name] = aligned.values
            added += 1

        print(f"  {seg}: {df_feat.shape[1]} 列 "
              f"（158 原始 + {added} 大资金）")

    print(f"  构建耗时 {time.time() - t0:.0f}s")

    print("\n训练 LightGBM ...")
    from qlib.contrib.model.gbdt import LGBModel
    model = LGBModel()
    t1 = time.time()
    model.fit(dataset)
    print(f"  完成，耗时 {time.time() - t1:.0f}s")

    print("\n在测试段预测 ...")
    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    label = dataset.prepare("test", col_set="label",
                            data_key="raw").iloc[:, 0]
    print(f"  {len(pred):,} 条预测")

    print("\n" + "-" * 62)
    print("测试段 IC（样本外）")
    print("-" * 62)
    ic = evaluate_ic(pred, label)
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

    report = build_report(
        engine, stats, out / "lgb_enhanced_report.html",
        title="Alpha158 + 大资金 + LightGBM 回测",
        subtitle=(f"{args.market} · 持仓 {args.holdings} 只 · "
                  f"每 {args.rebalance} 日调仓 · "
                  f"测试段 {TEST[0]} ~ {TEST[1]}（样本外） · "
                  f"IC {ic['IC均值']:.4f} t={ic['t值']:.2f}"),
        benchmark=bench if not bench.empty else None)

    print(f"\n明细: {out.resolve()}")
    print(f"报告: {report.resolve()}")
    print(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
