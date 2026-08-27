"""指数择时（时序趋势跟随）回测。

交易沪深300 ETF，信号来自 ETF 自身价格 —— 不涉及成分股，
因此**不受幸存者偏差与成分股前视影响**，是当前数据条件下最干净的回测。

用法::

    # 单组参数
    python scripts/run_index_timing.py --symbol 510300.SH --ma 200

    # 参数网格扫描（判断是否存在参数平原）
    python scripts/run_index_timing.py --grid

对照基准是**买入持有**而非指数点位 —— 择时策略要证明的是
「进出场比一直拿着强」，跟指数比是偷换问题。
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import Interval  # noqa: E402
from qmtquant.datafeed.xt_feed import XtDataFeed  # noqa: E402
from qmtquant.engine.backtest_engine import BacktestEngine  # noqa: E402
from qmtquant.report.html_report import build_report  # noqa: E402
from qmtquant.strategy.index_timing import IndexTimingStrategy  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize  # noqa: E402

#: 网格扫描的参数空间。窗口跨度取大是有意的 ——
#: 只有相邻窗口表现接近（参数平原）才说明规律真实存在
GRID = {
    "ma_window": [20, 40, 60, 80, 120, 200],
    "band": [0.0, 0.01, 0.02, 0.03],
    "confirm_days": [1, 2, 3],
}


def buy_and_hold(bars) -> tuple[float, float]:
    """买入持有的总收益与最大回撤，作为对照基准"""
    closes = pd.Series([b.close_price for b in bars],
                       index=[b.datetime for b in bars]).sort_index()
    total = closes.iloc[-1] / closes.iloc[0] - 1
    mdd = (closes / closes.cummax() - 1).min()
    return total, mdd


def run_one(bars, vt_symbol: str, setting: dict, capital: float, cost):
    engine = BacktestEngine(initial_capital=capital, cost=cost)
    engine.load_data(bars)
    engine.add_strategy(IndexTimingStrategy, [vt_symbol], setting)
    return engine, engine.run()


def main() -> int:
    p = argparse.ArgumentParser(description="指数择时回测")
    p.add_argument("--symbol", default="510300.SH", help="ETF 代码")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=None)

    p.add_argument("--ma", type=int, default=200, help="均线窗口")
    p.add_argument("--band", type=float, default=0.02, help="缓冲带")
    p.add_argument("--confirm-days", type=int, default=1)
    p.add_argument("--min-holding-days", type=int, default=5)

    p.add_argument("--grid", action="store_true", help="跑参数网格扫描")
    p.add_argument("--report", default="reports")
    args = p.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    vt_symbol = normalize(args.symbol)
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    capital = args.capital or cfg.backtest.initial_capital

    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    bars = feed.load_bars([vt_symbol], args.start, end, Interval.DAILY)
    if not bars:
        print(f"没有 {vt_symbol} 的数据，请先运行 scripts/download_data.py "
              f"--symbols {args.symbol} --intervals 1d")
        return 1

    bh_ret, bh_mdd = buy_and_hold(bars)
    print(f"{vt_symbol}  {len(bars)} 根日线  "
          f"{bars[0].datetime.date()} ~ {bars[-1].datetime.date()}")
    print(f"买入持有: 总收益 {bh_ret:+.2%}  最大回撤 {bh_mdd:.2%}\n")

    if args.grid:
        return run_grid(bars, vt_symbol, capital, cfg, bh_ret, bh_mdd, args)

    setting = {"ma_window": args.ma, "band": args.band,
               "confirm_days": args.confirm_days,
               "min_holding_days": args.min_holding_days}
    engine, stats = run_one(bars, vt_symbol, setting, capital, cfg.cost)

    print(stats.summary())
    print(f"\n信号切换 {engine.strategy.switch_count} 次，"
          f"期末信号={engine.strategy.signal}")
    print(f"\n对照买入持有：收益 {stats.total_return - bh_ret:+.2%}，"
          f"回撤 {stats.max_drawdown - bh_mdd:+.2%}")
    if stats.total_return < bh_ret:
        print("  ⚠ 跑输买入持有 —— 择时的代价没有换来收益，只换来回撤")

    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    report = build_report(
        engine, stats, out / "index_timing_report.html",
        title="指数择时回测",
        subtitle=(f"{vt_symbol} · MA{args.ma} · 缓冲带{args.band:.0%} · "
                  f"确认{args.confirm_days}日 · {args.start} ~ {end}"))
    print(f"\n可视化报告: {report.resolve()}")
    return 0


def run_grid(bars, vt_symbol, capital, cfg, bh_ret, bh_mdd, args) -> int:
    """参数网格扫描。

    看的不是「最好那组多好」，而是**相邻参数是否表现接近** ——
    孤峰意味着过拟合，平原才说明规律真实。
    """
    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    print(f"扫描 {len(combos)} 组参数...\n")

    rows = []
    for i, values in enumerate(combos, 1):
        setting = dict(zip(keys, values))
        setting["min_holding_days"] = args.min_holding_days
        try:
            engine, stats = run_one(bars, vt_symbol, setting, capital, cfg.cost)
        except Exception as e:  # noqa: BLE001 —— 单组失败不应中断整个扫描
            print(f"  [{i}/{len(combos)}] {setting} 失败：{e}")
            continue
        rows.append({**setting,
                     "总收益": stats.total_return,
                     "年化": stats.annual_return,
                     "最大回撤": stats.max_drawdown,
                     "Sharpe": stats.sharpe,
                     "切换": engine.strategy.switch_count})
        if i % 12 == 0:
            print(f"  {i}/{len(combos)} ...")

    if not rows:
        print("全部失败")
        return 1

    df = pd.DataFrame(rows)
    beat_ret = int((df["总收益"] > bh_ret).sum())
    beat_dd = int((df["最大回撤"] > bh_mdd).sum())

    print(f"\n{len(df)} 组结果")
    print(f"  收益为正      : {int((df['总收益'] > 0).sum())}/{len(df)}")
    print(f"  跑赢买入持有  : {beat_ret}/{len(df)}")
    print(f"  回撤优于持有  : {beat_dd}/{len(df)}")

    print("\n按 MA 窗口聚合（中位数）:")
    agg = df.groupby("ma_window")[["总收益", "最大回撤", "Sharpe", "切换"]].median()
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\n最好的 8 组:")
    top = df.nlargest(8, "总收益")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index_timing_grid.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n完整结果: {path.resolve()}")

    if beat_ret == 0:
        print("\n⚠ 没有任何一组跑赢买入持有。"
              "本策略的价值只在降回撤，不在增收益 —— 上实盘前务必想清楚这一点。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
