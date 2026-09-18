"""ETF 动量轮动回测。

## 必须跑遍调仓相位

PortfolioStrategy 的文档记录过一个实测：同一份信号、同一组参数，20 日
调仓下仅改变相位（0/4/8/12/16），4 年半累计收益从 -9.62% 到 +73.02%，
横跨 82 个百分点。

也就是说单相位回测测的不是策略好坏，而是「你碰巧从哪天开始」。所以这个
脚本默认跑全部相位并给出分布 —— 中位数才是对策略的估计，最好的那一次
不是。

## 与实盘共用同一个策略类

回测和实盘都跑 strategies/etf_rotation/strategy.py 里的
EtfRotationStrategy。这个项目已经因为「回测与实盘走两套逻辑」吃过大亏：
回测做同日买卖而 A 股 T+1 根本不允许，标定出的参数拿到实盘完全不适用。
ETF 这条线从一开始就不分叉。

用法::

    python scripts/backtest_etf_rotation.py
    python scripts/backtest_etf_rotation.py --max-holdings 5 --rebalance 10
    python scripts/backtest_etf_rotation.py --phases 0        # 只跑一个相位
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

UNIVERSE_CSV = ROOT / "data" / "universe" / "etf_candidates.csv"
OUT_DIR = ROOT / "strategies" / "etf_rotation" / "backtest"
RUNS_DIR = ROOT / "strategies" / "etf_rotation" / "runs"

#: 不参与轮动的类别 —— 它们是防御腿，不是收益来源
DEFENSIVE_CATS = ("债券", "货币")


def load_universe(max_symbols: int) -> tuple[list[str], str]:
    """读候选池，返回 (轮动标的, 防御腿)。"""
    import pandas as pd

    if not UNIVERSE_CSV.exists():
        raise FileNotFoundError(
            f"找不到候选池 {UNIVERSE_CSV}，先跑 scripts/screen_etf_universe.py")
    df = pd.read_csv(UNIVERSE_CSV, encoding="utf-8-sig")

    defensive = ""
    dfs = df[df["category"].isin(DEFENSIVE_CATS)]
    if not dfs.empty:
        # 防御腿取成交额最大的那只，流动性最好
        defensive = str(dfs.sort_values("avg_amount",
                                        ascending=False).iloc[0]["vt_symbol"])

    rot = df[~df["category"].isin(DEFENSIVE_CATS)]
    rot = rot.sort_values("avg_amount", ascending=False)
    syms = rot["vt_symbol"].astype(str).tolist()[:max_symbols]
    if defensive:
        syms.append(defensive)
    return syms, defensive


def run_one(symbols, defensive, bars, args, phase: int) -> dict | None:
    """跑单个相位，返回绩效。"""
    from qmtquant.config import get_config
    from qmtquant.engine.backtest_engine import BacktestEngine
    from qmtquant.risk.drawdown import DrawdownConfig, DrawdownController
    from strategies.etf_rotation.strategy import EtfRotationStrategy

    cfg = get_config()
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
    engine.add_strategy(EtfRotationStrategy, symbols, {
        "max_holdings": args.max_holdings,
        "rebalance_days": args.rebalance,
        "rebalance_phase": phase,
        "lookbacks": tuple(int(x) for x in args.lookbacks.split(",")),
        "min_momentum": args.min_momentum,
        "defensive": defensive,
        "vol_adjust": not args.no_vol_adjust,
    })
    stats = engine.run()
    return {
        "phase": phase,
        "total_return": float(stats.total_return),
        "annual_return": float(stats.annual_return),
        "max_drawdown": float(stats.max_drawdown),
        "sharpe": float(stats.sharpe_ratio),
        "n_trades": int(stats.total_trades),
        "win_rate": float(stats.win_rate),
        "trading_days": int(stats.trading_days),
        "turnover_rate": float(getattr(stats, "turnover_rate", 0.0)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="ETF 动量轮动回测")
    p.add_argument("--capital", type=float, default=200000,
                   help="本金，硬性 20 万")
    p.add_argument("--max-holdings", type=int, default=5,
                   help="持仓只数。单票 ≤20%% 是硬性风控，最少 5 只")
    p.add_argument("--rebalance", type=int, default=10, help="调仓周期（交易日）")
    p.add_argument("--lookbacks", default="20,60,120", help="动量回看窗口")
    p.add_argument("--min-momentum", type=float, default=0.0,
                   help="绝对动量下限，低于此不持有")
    p.add_argument("--no-vol-adjust", action="store_true",
                   help="不按波动率调整动量")
    p.add_argument("--max-symbols", type=int, default=60,
                   help="参与轮动的 ETF 只数（按成交额取前 N）")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2030-12-31")
    p.add_argument("--phases", default="",
                   help="只跑指定相位（逗号分隔）。留空则跑遍 0..rebalance-1")
    args = p.parse_args()

    from qmtquant.config import get_config
    from qmtquant.core.constants import Interval
    from qmtquant.datafeed.xt_feed import XtDataFeed

    cfg = get_config()
    print("=" * 66)
    print("ETF 动量轮动回测")
    print("=" * 66)
    print(f"  本金 {args.capital:,.0f}   持仓 {args.max_holdings} 只"
          f"（单票 {100/args.max_holdings:.0f}%，限 20%）")
    print(f"  调仓 每 {args.rebalance} 交易日   动量窗口 {args.lookbacks}")
    print(f"  区间 {args.start} ~ {args.end}")

    symbols, defensive = load_universe(args.max_symbols)
    print(f"  轮动池 {len(symbols)-1 if defensive else len(symbols)} 只"
          f"   防御腿 {defensive or '无（空仓）'}")

    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    bars = feed.load_bars(symbols, args.start, args.end, Interval.DAILY)
    if not bars:
        print("  没有行情数据，先跑 download_data.py 下 ETF 日线")
        return 1
    print(f"  装载 {len(bars):,} 根 K 线")

    phases = ([int(x) for x in args.phases.split(",") if x.strip()]
              if args.phases else list(range(args.rebalance)))
    print(f"\n跑 {len(phases)} 个调仓相位 —— 单相位的数字测的是运气，不是策略")
    print("-" * 66)
    print(f"  {'相位':>4} {'总收益':>9} {'年化':>9} {'回撤':>9} "
          f"{'夏普':>7} {'交易':>6} {'胜率':>7}")
    print("-" * 66)

    results = []
    for ph in phases:
        r = run_one(symbols, defensive, bars, args, ph)
        if r is None:
            continue
        results.append(r)
        print(f"  {ph:>4} {r['total_return']:>8.2%} {r['annual_return']:>8.2%} "
              f"{r['max_drawdown']:>8.2%} {r['sharpe']:>7.2f} "
              f"{r['n_trades']:>6} {r['win_rate']:>6.1%}")

    if not results:
        print("  全部相位都没跑出结果")
        return 1

    def _agg(key):
        vals = [r[key] for r in results]
        return (statistics.median(vals), min(vals), max(vals))

    print("-" * 66)
    print("相位分布（中位数才是对策略的估计，最好的那次不是）")
    print("-" * 66)
    for key, label, pct in (("total_return", "总收益", True),
                            ("annual_return", "年化", True),
                            ("max_drawdown", "最大回撤", True),
                            ("sharpe", "夏普", False)):
        med, lo, hi = _agg(key)
        f = (lambda v: f"{v:>8.2%}") if pct else (lambda v: f"{v:>8.2f}")
        print(f"  {label:<8} 中位 {f(med)}   最差 {f(lo)}   最好 {f(hi)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": vars(args),
        "defensive": defensive,
        "n_symbols": len(symbols),
        "phases": results,
        "median": {k: statistics.median([r[k] for r in results])
                   for k in ("total_return", "annual_return",
                             "max_drawdown", "sharpe")},
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {(OUT_DIR / 'summary.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
