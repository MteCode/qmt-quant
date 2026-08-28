"""Alpha158 + LightGBM，跑一遍并出回测报告。

⚠ **必须作为真实文件运行，不能 ``python - <<EOF``。**
Qlib 内部用 joblib 并行，Windows 上 multiprocessing 是 spawn 模式，
子进程要重新 import ``__main__`` —— 从 stdin 运行时没有可导入的主模块，
表现为永久挂起而不是报错（实测卡满 10 分钟无输出）。

## 时间切分（先声明，跑完不改）

======  ======================  =========================================
用途     区间                     说明
======  ======================  =========================================
训练     2016-01-01 ~ 2019-12-31  模型拟合
验证     2020-01-01 ~ 2021-12-31  早停，选迭代轮数
**测试** 2022-01-01 ~ 2026-08-27  **样本外，唯一判据，不回头调参**
======  ======================  =========================================

## ⚠ 这条路的过拟合风险高于本项目此前所有尝试

Alpha158 是 158 个因子，LightGBM 有几十个超参数。
本项目的自由度预算是 **4.3 个参数**（10 年日线折算 43 个独立市场状态，
每参数 10 个），这里直接爆表几十倍。

因此本脚本刻意：

- **用 Qlib 的默认超参数**，一个都不调
- 只看测试段结果，训练/验证段的漂亮数字不作为结论
- 不做任何「换个参数再试试」

## 分工

Qlib 负责找信号（Alpha158 + LightGBM），本项目引擎负责按 A 股规则执行 ——
T+1、涨跌停、整手、停牌不可交易、交易成本、回撤控制。
Qlib 自带的回测对这些处理得很粗。

用法::

    python scripts/run_qlib_ml.py
    python scripts/run_qlib_ml.py --holdings 50 --rebalance 20
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Qlib 训练时会往 mlflow 写指标。新版 mlflow 默认禁用文件后端并抛异常，
# 而我们只是要跑一次实验、不需要实验管理。放开文件后端即可，
# 不放开的话 model.fit 会在训练完成后才炸（白等十分钟）
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

#: 时间切分。**改这里等于换一次实验，改之前想清楚**
TRAIN = ("2016-01-01", "2019-12-31")
VALID = ("2020-01-01", "2021-12-31")
TEST = ("2022-01-01", "2026-08-27")


def build_dataset(uri: str, market: str):
    """Alpha158 数据集。

    ``fit_start_time`` 必须是训练段起点 —— 数据预处理（标准化、
    去极值）的统计量只能用训练段算，用全样本算等于把测试段的
    分布信息泄漏进训练。这是 Qlib 配置里最容易写错的一处。
    """
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    handler = Alpha158(
        instruments=market,
        start_time=TRAIN[0], end_time=TEST[1],
        fit_start_time=TRAIN[0], fit_end_time=TRAIN[1],
    )
    return DatasetH(handler, segments={
        "train": TRAIN, "valid": VALID, "test": TEST,
    })


def evaluate_ic(pred, label) -> dict:
    """测试段的 IC。

    与本项目 factor_ic 模块口径一致：秩相关（抗极值），
    逐日横截面算一个值，再看均值与 t 统计量。
    """
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
    """Qlib 的 MultiIndex 预测 -> 本项目的 日期 × vt_symbol 面板"""
    from qmtquant.datafeed.qlib_export import from_qlib_code

    panel = pred.unstack(level="instrument")
    if hasattr(panel, "columns") and panel.columns.nlevels > 1:
        panel.columns = panel.columns.droplevel(0)
    panel.columns = [from_qlib_code(str(c)) for c in panel.columns]
    return panel.sort_index()


def main() -> int:
    p = argparse.ArgumentParser(description="Alpha158 + LightGBM")
    p.add_argument("--uri", default=None, help="qlib 数据目录")
    p.add_argument("--market", default="csi300")
    p.add_argument("--index", default="000300.SH")
    p.add_argument("--holdings", type=int, default=30, help="持仓只数")
    p.add_argument("--rebalance", type=int, default=20, help="调仓间隔（交易日）")
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--report", default="reports/qlib_ml")
    p.add_argument("--no-drawdown", action="store_true",
                   help="关闭回撤控制（默认开启）")
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
    print("Alpha158 + LightGBM")
    print("=" * 62)
    print(f"数据      : {uri}  市场 {args.market}")
    print(f"训练      : {TRAIN[0]} ~ {TRAIN[1]}")
    print(f"验证      : {VALID[0]} ~ {VALID[1]}（早停）")
    print(f"测试      : {TEST[0]} ~ {TEST[1]}  ← 样本外，唯一判据")
    print("超参数    : Qlib 默认，一个都不调\n")

    t0 = time.time()
    print("构建 Alpha158 特征（158 个因子）...")
    dataset = build_dataset(uri, args.market)
    print(f"  完成，耗时 {time.time() - t0:.0f}s")

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
        print("  横截面不足，无法计算")
        return 1
    for k, v in ic.items():
        print(f"  {k:<10} {v:>10.4f}" if isinstance(v, float)
              else f"  {k:<10} {v:>10}")
    if abs(ic["t值"]) < 2:
        print("  ⚠ |t| < 2，信号在样本外不显著")

    # ---------------- 组合回测：本项目引擎，A 股规则 ----------------
    print("\n" + "-" * 62)
    print("组合回测（本项目引擎：T+1 / 涨跌停 / 整手 / 成本 / 回撤控制）")
    print("-" * 62)

    scores = to_score_panel(pred)
    print(f"分数面板: {scores.shape[0]} 期 × {scores.shape[1]} 只")

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

    # ---- 基准：全收益指数优先
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
        kind = "⚠ 价格指数（超额被高估约 2%/年）"

    if not bench.empty:
        bench_ret = bench.iloc[-1] / bench.iloc[0] - 1
        print(f"\n基准 {args.index} {kind}: {bench_ret:+.2%}")
        print(f"策略超额: {stats.total_return - bench_ret:+.2%}")
        if stats.total_return < bench_ret:
            print("  ⚠ 跑输基准")

    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    engine.get_equity_df().to_csv(out / "equity.csv", encoding="utf-8-sig")
    tdf = engine.get_trades_df()
    if not tdf.empty:
        tdf.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    scores.to_parquet(out / "scores.parquet")
    pd.Series(ic).to_csv(out / "test_ic.csv", encoding="utf-8-sig")

    report = build_report(
        engine, stats, out / "qlib_ml_report.html",
        title="Alpha158 + LightGBM 回测",
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
