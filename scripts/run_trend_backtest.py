"""趋势策略回测入口（日线 / 日内）。

用法：
    # 日线：周线MA5 + 年线MA250，趋势跟随
    python scripts/run_trend_backtest.py --symbols 600519.SH --mode trend

    # 同一段行情对比两种模式
    python scripts/run_trend_backtest.py --symbols 600519.SH --compare

    # 日内 VWAP（需已下载 1 分钟线）
    python scripts/run_trend_backtest.py --symbols 600519.SH --interval 1m \\
        --strategy vwap --mode reversion
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import Interval  # noqa: E402
from qmtquant.datafeed.xt_feed import XtDataFeed  # noqa: E402
from qmtquant.engine.backtest_engine import BacktestEngine  # noqa: E402
from qmtquant.strategy.intraday_vwap import IntradayVwapStrategy  # noqa: E402
from qmtquant.strategy.trend_ma import TrendMaStrategy  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize  # noqa: E402


def run_one(bars, capital, cost, strategy_cls, symbols, setting):
    engine = BacktestEngine(initial_capital=capital, cost=cost)
    engine.load_data(bars)
    engine.add_strategy(strategy_cls, symbols, setting)
    stats = engine.run()
    return engine, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="趋势策略回测")
    parser.add_argument("--symbols", default="600519.SH")
    parser.add_argument("--strategy", default="ma", choices=["ma", "vwap"])
    parser.add_argument("--interval", default="1d", choices=["1d", "1m"])
    parser.add_argument("--mode", default="trend", choices=["trend", "reversion"])
    parser.add_argument("--compare", action="store_true", help="两种模式都跑并对比")
    parser.add_argument("--fast", type=int, default=5, help="周线窗口")
    parser.add_argument("--slow", type=int, default=250, help="年线窗口")
    parser.add_argument("--no-filter", action="store_true", help="关闭年线趋势过滤")
    parser.add_argument("--role", default="entry_timing",
                        choices=["entry_timing", "t0_rotation"])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--report", default="reports")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    symbols = [normalize(s.strip()) for s in args.symbols.split(",") if s.strip()]
    start = args.start or cfg.backtest.start
    end = args.end or cfg.backtest.end
    capital = args.capital or cfg.backtest.initial_capital
    interval = Interval(args.interval)

    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    bars = feed.load_bars(symbols, start, end, interval)
    if not bars:
        print(f"没有 {interval.value} 数据，请先运行 scripts/download_data.py")
        return 1
    print(f"装载 {len(bars):,} 根 {interval.value} K线，"
          f"{bars[0].datetime.date()} ~ {bars[-1].datetime.date()}")

    if args.strategy == "ma":
        cls = TrendMaStrategy
        base = {"fast_window": args.fast, "slow_window": args.slow,
                "use_trend_filter": not args.no_filter}
    else:
        cls = IntradayVwapStrategy
        base = {"role": args.role}

    modes = ["trend", "reversion"] if args.compare else [args.mode]
    results = []

    for mode in modes:
        setting = dict(base, mode=mode)
        engine, stats = run_one(bars, capital, cfg.cost, cls, symbols, setting)
        results.append((mode, engine, stats))
        print()
        print(f"########## mode = {mode} ##########")
        print(stats.summary())

    if len(results) > 1:
        print()
        print("=" * 62)
        print(f"{'指标':<12}{'trend':>16}{'reversion':>16}")
        print("-" * 62)
        for label, attr, fmt in [
            ("总收益率", "total_return", "{:.2%}"),
            ("年化收益率", "annual_return", "{:.2%}"),
            ("最大回撤", "max_drawdown", "{:.2%}"),
            ("Sharpe", "sharpe_ratio", "{:.3f}"),
            ("成交笔数", "total_trades", "{:d}"),
            ("胜率", "win_rate", "{:.2%}"),
        ]:
            vals = [fmt.format(getattr(s, attr)) for _, _, s in results]
            print(f"{label:<12}{vals[0]:>16}{vals[1]:>16}")
        print("=" * 62)

    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    for mode, engine, _ in results:
        tag = f"{args.strategy}_{args.interval}_{mode}"
        engine.get_equity_df().to_csv(out / f"trend_{tag}_equity.csv",
                                      encoding="utf-8-sig")
        tdf = engine.get_trades_df()
        if not tdf.empty:
            tdf.to_csv(out / f"trend_{tag}_trades.csv", index=False,
                       encoding="utf-8-sig")
    print(f"\n明细已输出到 {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
