"""Alpha360 + ALSTM（Attention LSTM），跑一遍并出回测报告。

ALSTM 用 6 个原始价量特征 × 60 天滑动窗口（Alpha360），
通过注意力机制自动学习哪些时间步更重要。

时间切分与 run_qlib_ml.py 一致，超参数用 Qlib 默认。

用法::

    python scripts/run_alstm.py --market csi1000 --index 000852.SH
    python scripts/run_alstm.py --market csi1000 --index 000852.SH --holdings 50 --rebalance 20
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


def build_dataset(uri: str, market: str):
    from qlib.contrib.data.handler import Alpha360
    from qlib.data.dataset import DatasetH

    handler = Alpha360(
        instruments=market,
        start_time=TRAIN[0], end_time=TEST[1],
        fit_start_time=TRAIN[0], fit_end_time=TRAIN[1],
    )
    return DatasetH(handler, segments={
        "train": TRAIN, "valid": VALID, "test": TEST,
    })


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
    p = argparse.ArgumentParser(description="Alpha360 + ALSTM")
    p.add_argument("--uri", default=None, help="qlib 数据目录")
    p.add_argument("--market", default="csi1000")
    p.add_argument("--index", default="000852.SH")
    p.add_argument("--holdings", type=int, default=50)
    p.add_argument("--rebalance", type=int, default=20)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--report", default="reports/alstm_csi1000")
    p.add_argument("--no-drawdown", action="store_true")
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0, help="-1 表示用 CPU")
    args = p.parse_args()

    import pandas as pd

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    uri = args.uri or str(Path(cfg.data.store_dir) / "qlib_data")
    if not Path(uri).exists():
        print(f"数据目录不存在: {uri}\n请先运行 scripts/export_qlib.py")
        return 1

    import qlib
    qlib.init(provider_uri=uri, region="cn", joblib_backend="threading")

    print("=" * 62)
    print("Alpha360 + ALSTM（Attention LSTM）")
    print("=" * 62)
    print(f"数据      : {uri}  市场 {args.market}")
    print(f"训练      : {TRAIN[0]} ~ {TRAIN[1]}")
    print(f"验证      : {VALID[0]} ~ {VALID[1]}（早停）")
    print(f"测试      : {TEST[0]} ~ {TEST[1]}  <- 样本外")
    print(f"GPU       : {args.gpu}")
    print(f"Epochs    : {args.n_epochs}\n")

    t0 = time.time()
    print("构建 Alpha360 特征（6 特征 x 60 天）...")
    dataset = build_dataset(uri, args.market)
    print(f"  完成，耗时 {time.time() - t0:.0f}s")

    print("\n训练 ALSTM ...")
    from qlib.contrib.model.pytorch_alstm import ALSTM
    model = ALSTM(
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=args.n_epochs,
        lr=0.001,
        early_stop=20,
        batch_size=2000,
        loss="mse",
        GPU=args.gpu,
    )
    t1 = time.time()
    model.fit(dataset)
    print(f"  完成，耗时 {(time.time() - t1) / 60:.1f} 分钟")

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
        print("  横截面不足，无法计算")
        return 1
    for k, v in ic.items():
        print(f"  {k:<10} {v:>10.4f}" if isinstance(v, float)
              else f"  {k:<10} {v:>10}")
    if abs(ic["t值"]) < 2:
        print("  !! |t| < 2，信号在样本外不显著")

    # ---------------- 组合回测 ----------------
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
        kind = "!! 价格指数（超额被高估约 2%/年）"

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
        engine, stats, out / "alstm_report.html",
        title="Alpha360 + ALSTM 回测",
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
