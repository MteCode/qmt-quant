"""通用回测入口 —— 任何策略都能跑，不用为它单独写脚本。

## 快速上手

    # 看策略有哪些参数
    python scripts/run_strategy.py --strategy mean_reversion --describe

    # 跑内置策略
    python scripts/run_strategy.py --strategy index_timing \\
        --symbols 510300.SH --set ma_window=200 --set band=0.03

    # 跑你自己写的策略（完整路径即可，不用改本脚本）
    python scripts/run_strategy.py --strategy my_pkg.my_mod.MyStrategy \\
        --universe-csv data/universe/index_weight_000300.SH.csv

## 标的池三选一

============================  ==========================================
方式                           偏差情况
============================  ==========================================
``--symbols 600519.SH,...``    自己指定，偏差自负
``--sector 沪深300``           当前成分快照 + 上市日/纳入日过滤
``--universe-csv ...``         **历史成分股，无幸存者偏差与成分股前视**
============================  ==========================================

选股策略请务必用第三种。前两种在实测中让同一组参数的总收益
从 -4.49% 变成 +320.33% —— 320 个百分点全是偏差。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import Interval  # noqa: E402
from qmtquant.datafeed.xt_feed import IndexFeed, XtDataFeed  # noqa: E402
from qmtquant.engine.backtest_engine import BacktestEngine  # noqa: E402
from qmtquant.report.html_report import build_report  # noqa: E402
from qmtquant.research.loader import (  # noqa: E402
    check_params,
    describe,
    load_strategy,
    parse_params,
)
from qmtquant.risk.drawdown import DrawdownConfig, DrawdownController  # noqa: E402
from qmtquant.universe.providers import (  # noqa: E402
    HistoricalUniverse,
    PointInTimeUniverse,
    StaticUniverse,
)
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize  # noqa: E402


def build_universe(args, cfg):
    """按命令行选项构造标的池，返回 (provider, symbols)"""
    if args.universe_csv:
        p = HistoricalUniverse(args.universe_csv)
        return p, p.all_symbols()

    if args.symbols:
        syms = [normalize(s.strip()) for s in args.symbols.split(",") if s.strip()]
        return StaticUniverse(syms, source="命令行指定",
                               from_index_snapshot=False), syms

    meta_path = (Path(cfg.data.store_dir) / "universe"
                 / f"universe_{args.sector}.parquet")
    if not meta_path.exists():
        raise SystemExit(
            f"缺少标的元数据: {meta_path}\n"
            f"请先运行: python scripts/build_universe.py --sector {args.sector}")

    meta = pd.read_parquet(meta_path)
    syms = meta["vt_symbol"].tolist()
    base = StaticUniverse(syms, source=f"{args.sector} 当前成分快照")
    listing = {r.vt_symbol: r.listing_date for r in meta.itertuples()
               if pd.notna(r.listing_date)}
    delist = {r.vt_symbol: r.delist_date for r in meta.itertuples()
              if pd.notna(r.delist_date)}
    inclusion = ({r.vt_symbol: r.inclusion_date for r in meta.itertuples()
                  if pd.notna(getattr(r, "inclusion_date", None))}
                 if "inclusion_date" in meta.columns else {})
    return PointInTimeUniverse(base, listing, delist,
                               min_days_since_ipo=args.min_ipo_days,
                               inclusion_dates=inclusion), syms


def main() -> int:
    p = argparse.ArgumentParser(
        description="通用策略回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--strategy", required=True,
                   help="短名（mean_reversion 等）或完整路径 pkg.mod.Class")
    p.add_argument("--set", action="append", dest="params", metavar="K=V",
                   help="策略参数，可重复。值按 JSON 解析")
    p.add_argument("--describe", action="store_true",
                   help="只打印策略参数与默认值后退出")

    p.add_argument("--symbols", default=None, help="逗号分隔的标的代码")
    p.add_argument("--sector", default="沪深300")
    p.add_argument("--universe-csv", default=None,
                   help="历史成分股 CSV（推荐，无幸存者偏差）")
    p.add_argument("--min-ipo-days", type=int, default=60)

    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=None)
    p.add_argument("--interval", default="1d", choices=["1d", "1w", "1m"])
    p.add_argument("--drawdown", action="store_true", help="启用回撤控制")
    p.add_argument("--benchmark", default="000300.SH",
                   help="基准指数，空字符串表示不对标")
    p.add_argument("--report", default="reports",
                   help="输出目录。**不同策略请用不同目录，否则会互相覆盖**")
    args = p.parse_args()

    cls = load_strategy(args.strategy)
    if args.describe:
        print(describe(cls))
        return 0

    params = parse_params(args.params)
    # 未声明的参数会被静默丢弃，敲错名字不会报错，所以这里主动拦
    check_params(cls, params)

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    start = args.start or cfg.backtest.start
    end = args.end or cfg.backtest.end
    capital = args.capital or cfg.backtest.initial_capital

    provider, symbols = build_universe(args, cfg)
    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    print(f"策略  : {cls.__module__}.{cls.__name__}")
    print(f"参数  : {params or '（全部用默认值）'}")
    print(f"标的  : {len(symbols)} 只   区间: {start} ~ {end}")
    print("装载数据 ...")

    bars = feed.load_bars(symbols, start, end, Interval(args.interval))
    if not bars:
        print("没有可用数据，请先运行 scripts/download_data.py")
        return 1
    print(f"共 {len(bars):,} 根 K 线\n")

    drawdown = (DrawdownController(DrawdownConfig(min_observations=20))
                if args.drawdown else None)
    engine = BacktestEngine(initial_capital=capital, cost=cfg.cost,
                            drawdown=drawdown)
    engine.load_data(bars)
    engine.set_universe(provider)
    engine.add_strategy(cls, symbols, params)

    stats = engine.run()

    print(stats.summary())
    print()
    print(provider.describe_bias().summary())
    if drawdown:
        print()
        print(drawdown.summary())

    # 与买入持有对照。单标的时这是比指数更诚实的基准 ——
    # 择时策略要证明的是「进出场比一直拿着强」
    if len(symbols) == 1:
        closes = pd.Series([b.close_price for b in bars],
                           index=[b.datetime for b in bars]).sort_index()
        bh = closes.iloc[-1] / closes.iloc[0] - 1
        bh_dd = (closes / closes.cummax() - 1).min()
        print(f"\n买入持有      : 收益 {bh:+.2%}  最大回撤 {bh_dd:.2%}")
        print(f"策略相对       : 收益 {stats.total_return - bh:+.2%}  "
              f"回撤 {stats.max_drawdown - bh_dd:+.2%}")
        if stats.total_return < bh:
            print("  ⚠ 跑输买入持有")

    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    tag = cls.__name__
    engine.get_equity_df().to_csv(out / f"{tag}_equity.csv",
                                  encoding="utf-8-sig")
    tdf = engine.get_trades_df()
    if not tdf.empty:
        tdf.to_csv(out / f"{tag}_trades.csv", index=False,
                   encoding="utf-8-sig")

    bench = None
    if args.benchmark:
        bench = IndexFeed(cfg.data.store_dir).load_close(args.benchmark,
                                                         start, end)
        if bench.empty:
            print(f"\n[!] 无基准数据 {args.benchmark}，"
                  "请先运行 scripts/download_index.py")
            bench = None

    report = build_report(
        engine, stats, out / f"{tag}_report.html",
        title=f"{tag} 回测",
        subtitle=(f"{len(symbols)} 只标的 · {start} ~ {end}"
                  + (f" · {params}" if params else "")),
        benchmark=bench)
    print(f"\n明细: {out.resolve()}")
    print(f"报告: {report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
