"""Alpha360 + ALSTM（Attention LSTM），跑一遍并出回测报告。

ALSTM 用 6 个原始价量特征 × 60 天滑动窗口（Alpha360），
通过注意力机制自动学习哪些时间步更重要。

时间切分与 run_qlib_ml.py 一致，超参数用 Qlib 默认。

用法::

    python strategies/alstm_ppo_csi1000/train_alstm.py
    python strategies/alstm_ppo_csi1000/train_alstm.py --holdings 50 --rebalance 20

⚠ 重训会覆盖 models/alstm_scores.parquet。覆盖前先 git commit，
   否则旧分数无法找回，对应的回测结果永久失去可复现性。
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
PARAMS = paths.load_params()

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


#: 网络结构超参。改这些等于换了模型，旧权重加载会失败（形状对不上），
#: 所以随权重一起存进 meta，加载时校验
HYPER = {
    "d_feat": 6,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.0,
}


def save_model(model, n_epochs: int) -> None:
    """保存 ALSTM 权重 + 超参。

    Qlib 的 ALSTM 包装类本身不可 pickle（含 logger、optimizer 等），
    但内部 `ALSTM_model` 是标准 nn.Module，存 state_dict 即可。
    """
    import json

    import torch

    paths.ensure_dirs()
    torch.save(model.ALSTM_model.state_dict(), paths.ALSTM_WEIGHTS)
    meta = dict(HYPER)
    meta.update({
        "n_epochs": n_epochs,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train": list(TRAIN), "valid": list(VALID), "test": list(TEST),
    })
    paths.ALSTM_META.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"权重已保存: {paths.ALSTM_WEIGHTS}")


def load_model(gpu: int = -1):
    """加载已训练的 ALSTM，返回可直接 predict 的模型。

    权重不存在返回 None，调用方自行决定是重训还是报错。
    """
    import json

    import torch
    from qlib.contrib.model.pytorch_alstm import ALSTM

    if not paths.ALSTM_WEIGHTS.exists():
        return None

    meta = {}
    if paths.ALSTM_META.exists():
        meta = json.loads(paths.ALSTM_META.read_text(encoding="utf-8"))
    for k, v in HYPER.items():
        if k in meta and meta[k] != v:
            raise ValueError(
                f"权重的网络结构与当前代码不符: {k}={meta[k]} vs {v}。"
                f"改过结构就必须重训，不能加载旧权重")

    model = ALSTM(**HYPER, n_epochs=1, lr=0.001, early_stop=20,
                  batch_size=2000, loss="mse", GPU=gpu)
    state = torch.load(paths.ALSTM_WEIGHTS, map_location=model.device)
    model.ALSTM_model.load_state_dict(state)
    model.ALSTM_model.to(model.device)
    # Qlib 的 predict() 会检查这个标志，不置位会拒绝推理
    model.fitted = True
    print(f"已加载权重: {paths.ALSTM_WEIGHTS}"
          + (f"（训练于 {meta['trained_at']}）" if "trained_at" in meta else ""))
    return model


def main() -> int:
    p = argparse.ArgumentParser(description="Alpha360 + ALSTM")
    p.add_argument("--uri", default=None, help="qlib 数据目录")
    p.add_argument("--market", default=PARAMS["market"])
    p.add_argument("--index", default=PARAMS["index"])
    p.add_argument("--holdings", type=int, default=50)
    p.add_argument("--rebalance", type=int, default=20)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--report", default=str(paths.BACKTEST_DIR))
    p.add_argument("--no-drawdown", action="store_true")
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0, help="-1 表示用 CPU")
    p.add_argument("--reuse-weights", action="store_true",
                    help="加载已保存的权重直接推理，不重新训练。"
                         "用于复现回测或每日刷新分数（约 1 分钟 vs 重训 20 分钟）")
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

    # Alpha360 有 360 个表达式，并发下会撑爆 Qlib 默认 500 条的内存缓存，
    # 触发 LRU 淘汰竞争导致随机 KeyError。详见 qlib_init 的模块文档
    from qmtquant.datafeed.qlib_init import init_qlib
    cache_limit = init_qlib(uri, n_expressions=360)

    print("=" * 62)
    print("Alpha360 + ALSTM（Attention LSTM）")
    print("=" * 62)
    print(f"缓存上限  : {cache_limit} 条")
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

    model = None
    if args.reuse_weights:
        model = load_model(args.gpu)
        if model is None:
            print("\n[WARN] 没有已保存的权重，改为重新训练")

    if model is None:
        print("\n训练 ALSTM ...")
        from qlib.contrib.model.pytorch_alstm import ALSTM
        model = ALSTM(
            **HYPER,
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
        save_model(model, args.n_epochs)

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
    engine.get_equity_df().to_csv(out / "alstm_only_equity.csv",
                                  encoding="utf-8-sig")
    tdf = engine.get_trades_df()
    if not tdf.empty:
        tdf.to_csv(out / "alstm_only_trades.csv", index=False,
                   encoding="utf-8-sig")
    pd.Series(ic).to_csv(out / "alstm_test_ic.csv", encoding="utf-8-sig")

    # 分数面板存到 models/（走 LFS）。这是下游 PPO 回测和实盘选股的输入，
    # 覆盖前应先提交旧版本，否则对应回测结果失去可复现性
    paths.ensure_dirs()
    scores.to_parquet(paths.ALSTM_SCORES)
    print(f"分数面板已保存: {paths.ALSTM_SCORES}")

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
