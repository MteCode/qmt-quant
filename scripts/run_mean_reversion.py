"""均值回归策略回测。

用法：
    python scripts/run_mean_reversion.py --sector 沪深300 --start 2021-01-01
    python scripts/run_mean_reversion.py --entry-z -1.2 --holdings 10
    python scripts/run_mean_reversion.py --no-trend-filter   # 观察去掉趋势过滤的后果
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import Interval  # noqa: E402
from qmtquant.datafeed.xt_feed import BENCHMARKS, IndexFeed, XtDataFeed  # noqa: E402
from qmtquant.engine.backtest_engine import BacktestEngine  # noqa: E402
from qmtquant.report.html_report import build_report  # noqa: E402
from qmtquant.risk.drawdown import DrawdownConfig, DrawdownController  # noqa: E402
from qmtquant.strategy.mean_reversion import MeanReversionStrategy  # noqa: E402
from qmtquant.universe.providers import PointInTimeUniverse, StaticUniverse  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402


def build_universe(sector: str, cfg, min_ipo_days: int):
    meta_path = (Path(cfg.data.store_dir) / "universe"
                 / f"universe_{sector}.parquet")
    if not meta_path.exists():
        print(f"缺少标的元数据: {meta_path}")
        print(f"请先运行: python scripts/build_universe.py --sector {sector}")
        return None, []

    meta = pd.read_parquet(meta_path)
    symbols = meta["vt_symbol"].tolist()
    base = StaticUniverse(symbols, source=f"{sector} 当前成分快照")
    listing = {r.vt_symbol: r.listing_date for r in meta.itertuples()
               if pd.notna(r.listing_date)}
    delist = {r.vt_symbol: r.delist_date for r in meta.itertuples()
              if pd.notna(r.delist_date)}
    return PointInTimeUniverse(base, listing, delist,
                               min_days_since_ipo=min_ipo_days), symbols


def main() -> int:
    p = argparse.ArgumentParser(description="均值回归策略回测")
    p.add_argument("--sector", default="沪深300")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=None)
    p.add_argument("--min-ipo-days", type=int, default=60)

    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--entry-z", type=float, default=-2.0)
    p.add_argument("--exit-z", type=float, default=-0.5)
    p.add_argument("--stop-z", type=float, default=-4.0)
    p.add_argument("--max-holding-days", type=int, default=20)
    p.add_argument("--trend-window", type=int, default=120)
    p.add_argument("--no-trend-filter", action="store_true",
                   help="关闭趋势过滤，用于观察「下跌趋势中接飞刀」的后果")
    p.add_argument("--min-turnover", type=float, default=50_000_000)
    p.add_argument("--holdings", type=int, default=10)

    p.add_argument("--drawdown", action="store_true", help="启用回撤控制")
    p.add_argument("--benchmark", default="000300.SH")
    p.add_argument("--report", default="reports")
    args = p.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    start = args.start or cfg.backtest.start
    end = args.end or cfg.backtest.end
    capital = args.capital or cfg.backtest.initial_capital

    provider, symbols = build_universe(args.sector, cfg, args.min_ipo_days)
    if provider is None:
        return 1

    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    print(f"装载数据：{len(symbols)} 只标的，{start} ~ {end} ...")
    bars = feed.load_bars(symbols, start, end, Interval.DAILY)
    if not bars:
        print("没有可用数据，请先运行 scripts/download_data.py")
        return 1
    print(f"共 {len(bars):,} 根 K 线")

    drawdown = None
    if args.drawdown:
        drawdown = DrawdownController(DrawdownConfig(min_observations=20))

    engine = BacktestEngine(initial_capital=capital, cost=cfg.cost,
                            drawdown=drawdown)
    engine.load_data(bars)
    engine.set_universe(provider)
    engine.add_strategy(MeanReversionStrategy, symbols, {
        "lookback": args.lookback,
        "entry_z": args.entry_z,
        "exit_z": args.exit_z,
        "stop_z": args.stop_z,
        "max_holding_days": args.max_holding_days,
        # 0 = 关闭过滤。注意不能用 1：窗口为 1 时均线即价格自身，
        # price > price 恒为 False，会变成永远不许买入
        "trend_filter_window": 0 if args.no_trend_filter else args.trend_window,
        "min_turnover": args.min_turnover,
        "max_holdings": args.holdings,
    })

    stats = engine.run()
    strategy = engine.strategy

    print()
    print(stats.summary())
    print()
    print("--- 退出原因归因 ---")
    if strategy.exit_stats:
        total = sum(strategy.exit_stats.values())
        for reason, n in sorted(strategy.exit_stats.items(),
                                key=lambda kv: -kv[1]):
            print(f"  {reason:<12} {n:>4} 次  ({n / total:.1%})")
    else:
        print("  无退出记录")
    if drawdown:
        print()
        print(drawdown.summary())

    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    engine.get_equity_df().to_csv(out / "mr_equity.csv", encoding="utf-8-sig")
    tdf = engine.get_trades_df()
    if not tdf.empty:
        tdf.to_csv(out / "mr_trades.csv", index=False, encoding="utf-8-sig")

    bench = None
    if args.benchmark:
        bench = IndexFeed(cfg.data.store_dir).load_close(args.benchmark,
                                                         start, end)
        if bench.empty:
            bench = None

    report = build_report(
        engine, stats, out / "mean_reversion_report.html",
        title="均值回归策略回测",
        subtitle=(f"{args.sector} · 持仓 {args.holdings} 只 · "
                  f"z入场{args.entry_z}/出场{args.exit_z}/止损{args.stop_z} · "
                  f"{start} ~ {end}"
                  + (f" · 基准 {BENCHMARKS.get(args.benchmark, args.benchmark)}"
                     if bench is not None else "")),
        benchmark=bench,
    )
    print(f"\n可视化报告: {report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
