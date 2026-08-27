"""选股（多标的组合）回测入口。

用法：
    # 沪深300 动量轮动，带上市日过滤
    python scripts/run_portfolio_backtest.py --sector 沪深300 --start 2021-01-01

    # 关闭上市日过滤，观察偏差影响有多大
    python scripts/run_portfolio_backtest.py --sector 沪深300 --no-pit

    # 用外部历史成分股 CSV（可完全消除幸存者偏差）
    python scripts/run_portfolio_backtest.py --universe-csv data/hs300_history.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import Interval  # noqa: E402
from qmtquant.datafeed.xt_feed import XtDataFeed  # noqa: E402
from qmtquant.engine.backtest_engine import BacktestEngine  # noqa: E402
from qmtquant.strategy.momentum import (  # noqa: E402
    MomentumRotationStrategy,
)
from qmtquant.universe.providers import (  # noqa: E402
    HistoricalUniverse,
    PointInTimeUniverse,
    StaticUniverse,
)
from qmtquant.utils.logger import setup_logging  # noqa: E402


def build_universe(args, cfg, feed: XtDataFeed):
    """构造标的池，返回 (provider, 需要装载数据的标的列表)"""
    if args.universe_csv:
        provider = HistoricalUniverse(args.universe_csv)
        return provider, provider.all_symbols()

    # 从本地导出的元数据取名单与上市日，不依赖 QMT 客户端
    meta_path = Path(cfg.data.store_dir) / "universe" / f"universe_{args.sector}.parquet"
    if not meta_path.exists():
        print(f"缺少标的元数据: {meta_path}")
        print(f"请先运行: python scripts/build_universe.py --sector {args.sector}")
        return None, []

    meta = pd.read_parquet(meta_path)
    symbols = meta["vt_symbol"].tolist()
    base = StaticUniverse(symbols, source=f"{args.sector} 当前成分快照")

    if args.no_pit:
        return base, symbols

    listing = {r.vt_symbol: r.listing_date for r in meta.itertuples()
               if pd.notna(r.listing_date)}
    delist = {r.vt_symbol: r.delist_date for r in meta.itertuples()
              if pd.notna(r.delist_date)}
    inclusion = ({r.vt_symbol: r.inclusion_date for r in meta.itertuples()
                  if pd.notna(getattr(r, "inclusion_date", None))}
                 if "inclusion_date" in meta.columns else {})
    provider = PointInTimeUniverse(base, listing, delist,
                                   min_days_since_ipo=args.min_ipo_days,
                                   inclusion_dates=inclusion)
    return provider, symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="选股组合回测")
    parser.add_argument("--sector", default="沪深300")
    parser.add_argument("--universe-csv", default=None, help="历史成分股 CSV")
    parser.add_argument("--no-pit", action="store_true",
                        help="关闭上市日过滤（用于对比偏差影响）")
    parser.add_argument("--min-ipo-days", type=int, default=60)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--holdings", type=int, default=10, help="持仓只数")
    parser.add_argument("--rebalance", type=int, default=20, help="调仓间隔（交易日）")
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--skip-recent", type=int, default=20,
                        help="跳过最近 N 日，规避短期反转效应污染动量")
    parser.add_argument("--reverse", action="store_true",
                        help="反向模式：选跌得最惨的。"
                             "实测动量 IC 显著为负（-0.0575, t=-4.6），"
                             "反着做在逻辑上值得一试")
    parser.add_argument("--report", default="reports")
    parser.add_argument("--benchmark", default="000300.SH",
                        help="基准指数代码，空字符串表示不对标")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    start = args.start or cfg.backtest.start
    end = args.end or cfg.backtest.end
    capital = args.capital or cfg.backtest.initial_capital

    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    provider, symbols = build_universe(args, cfg, feed)
    if provider is None:
        return 1

    print(f"装载数据：{len(symbols)} 只标的，{start} ~ {end} ...")
    bars = feed.load_bars(symbols, start, end, Interval.DAILY)
    if not bars:
        print("没有可用数据，请先运行 scripts/download_data.py")
        return 1
    print(f"共 {len(bars):,} 根 K 线")

    engine = BacktestEngine(initial_capital=capital, cost=cfg.cost)
    engine.load_data(bars)
    engine.set_universe(provider)
    engine.add_strategy(MomentumRotationStrategy, symbols, {
        "lookback": args.lookback,
        "skip_recent": args.skip_recent,
        "reverse": args.reverse,
        "max_holdings": args.holdings,
        "rebalance_days": args.rebalance,
    })

    stats = engine.run()

    print()
    print(stats.summary())
    print()
    print(provider.describe_bias().summary())

    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    engine.get_equity_df().to_csv(out / "portfolio_equity.csv", encoding="utf-8-sig")
    trades_df = engine.get_trades_df()
    if not trades_df.empty:
        trades_df.to_csv(out / "portfolio_trades.csv", index=False, encoding="utf-8-sig")

    from qmtquant.datafeed.xt_feed import BENCHMARKS, IndexFeed
    from qmtquant.report.html_report import build_report

    bench = None
    if args.benchmark:
        bench = IndexFeed(cfg.data.store_dir).load_close(args.benchmark, start, end)
        if bench.empty:
            print(f"[!] 无基准数据 {args.benchmark}，"
                  f"请先运行 scripts/download_index.py")
            bench = None

    bench_name = BENCHMARKS.get(args.benchmark, args.benchmark)
    report = build_report(
        engine, stats, out / "portfolio_report.html",
        title="选股组合回测",
        subtitle=(f"{args.sector} · {'反向' if args.reverse else '正向'}动量 · "
                  f"持仓 {args.holdings} 只 · {start} ~ {end}"
                  + (f" · 基准 {bench_name}" if bench is not None else "")),
        benchmark=bench,
    )
    print(f"\n明细已输出到 {out.resolve()}")
    print(f"可视化报告: {report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
