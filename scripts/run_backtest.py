"""回测入口。

用法：
    # 用模拟数据自测框架（无需 QMT）
    python scripts/run_backtest.py --mock

    # 用本地已下载的真实数据
    python scripts/run_backtest.py --symbols 000001.SZ,600519.SH --start 2021-01-01
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import Interval  # noqa: E402
from qmtquant.datafeed.csv_feed import generate_random_bars  # noqa: E402
from qmtquant.engine.backtest_engine import BacktestEngine  # noqa: E402
from qmtquant.strategy.examples.ma_cross import MaCrossStrategy  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="qmtquant 回测")
    parser.add_argument("--symbols", default="000001.SZ", help="逗号分隔的标的代码")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--fast", type=int, default=5)
    parser.add_argument("--slow", type=int, default=20)
    parser.add_argument("--mock", action="store_true", help="用随机行情自测框架")
    parser.add_argument("--report", default="reports", help="报告输出目录")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    vt_symbols = [normalize(s.strip()) for s in args.symbols.split(",") if s.strip()]
    start = args.start or cfg.backtest.start
    end = args.end or cfg.backtest.end
    capital = args.capital or cfg.backtest.initial_capital

    # ---- 装载数据
    if args.mock:
        print("[!] 使用随机模拟行情，绩效结果无任何参考意义，仅验证框架链路")
        bars = []
        for i, s in enumerate(vt_symbols):
            bars += generate_random_bars(s, days=500, seed=42 + i)
    else:
        from qmtquant.datafeed.xt_feed import XtDataFeed
        feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
        bars = feed.load_bars(vt_symbols, start, end, Interval.DAILY)

    if not bars:
        print("没有可用数据。先运行 python scripts/download_data.py，或加 --mock 自测")
        return 1

    # ---- 运行回测
    engine = BacktestEngine(initial_capital=capital, cost=cfg.cost)
    engine.load_data(bars)
    engine.add_strategy(MaCrossStrategy, vt_symbols, {
        "fast_window": args.fast,
        "slow_window": args.slow,
    })

    stats = engine.run()
    print()
    print(stats.summary())

    # ---- 输出明细
    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    engine.get_equity_df().to_csv(out / "equity.csv", encoding="utf-8-sig")
    trades_df = engine.get_trades_df()
    if not trades_df.empty:
        trades_df.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    print(f"\n明细已输出到 {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
