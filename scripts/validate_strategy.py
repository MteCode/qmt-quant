"""策略稳健性验证：四道检验。

回答的不是「哪组参数收益最高」，而是「这个策略是否只在某个特定参数点上有效」。
只在一个点有效的策略是拟合噪声的产物，实盘必然失效。

四道检验分别回答：

1. **参数平原** —— 最优点周围的参数是不是也还行？孤峰即过拟合
2. **样本外** —— 样本内选出的参数，拿到没见过的区间还灵吗？
3. **walk-forward** —— 滚动地「用过去选参数、在未来验证」，多个窗口是否稳定
4. **成本敏感性** —— 手续费翻倍还活得下去吗？多少倍成本会打平？

用法：
    # 内置策略
    python scripts/validate_strategy.py --strategy mean_reversion

    # 你自己写的策略 + 自定义参数网格
    python scripts/validate_strategy.py --strategy my_pkg.my_mod.MyStrategy         --grid '{"ma_window": [20, 60, 120], "band": [0.0, 0.02]}'

    # 用历史成分池（推荐，无幸存者偏差）
    python scripts/validate_strategy.py --strategy mean_reversion         --universe-csv data/universe/index_weight_000300.SH.csv

    python scripts/validate_strategy.py --skip walkforward   # 跳过耗时的滚动验证
    python scripts/validate_strategy.py --quick              # 小网格快速跑
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import CostConfig, LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import Interval  # noqa: E402
from qmtquant.datafeed.xt_feed import XtDataFeed  # noqa: E402
from qmtquant.engine.backtest_engine import BacktestEngine  # noqa: E402
from qmtquant.risk.drawdown import DrawdownConfig, DrawdownController  # noqa: E402
from qmtquant.research.validation import (  # noqa: E402
    SplitReport,
    _to_result,
    cost_sensitivity,
    filter_by_drawdown as _eligible,
    grid_search,
    parameter_plateau,
    summarize_verdict,
    walk_forward,
)
from qmtquant.research.loader import (  # noqa: E402
    check_params,
    load_strategy,
    parse_params,
)
from qmtquant.universe.providers import (  # noqa: E402
    HistoricalUniverse,
    PointInTimeUniverse,
    StaticUniverse,
)
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize  # noqa: E402

#: 各策略的内置网格。参数取值刻意覆盖「明显偏松」到「明显偏紧」，
#: 这样才看得出最优点周围是平原还是孤峰。
#: 自己写的策略用 --grid 传 JSON 即可，不必改这里
DEFAULT_GRIDS = {
    "MeanReversionStrategy": {
        "lookback": [10, 15, 20, 30],
        "entry_z": [-1.2, -1.5, -2.0, -2.5],
        "exit_z": [-0.5, 0.0, 0.5],
        "max_holding_days": [10, 20, 40],
    },
    "IndexTimingStrategy": {
        "ma_window": [20, 40, 60, 120, 200],
        "band": [0.0, 0.01, 0.02, 0.03],
        "confirm_days": [1, 2, 3],
    },
    "MomentumRotationStrategy": {
        "lookback": [60, 120, 250],
        "skip_recent": [0, 5, 20],
        "rebalance_days": [10, 20, 60],
    },
    "MaCrossStrategy": {
        "fast_window": [3, 5, 10, 20],
        "slow_window": [20, 60, 120, 250],
    },
}

QUICK_GRIDS = {
    "MeanReversionStrategy": {
        "lookback": [10, 20, 30],
        "entry_z": [-1.2, -1.5, -2.0],
        "exit_z": [-0.5, 0.5],
    },
    "IndexTimingStrategy": {
        "ma_window": [20, 60, 200],
        "band": [0.0, 0.02],
    },
}


class Harness:
    """把数据装载一次、复用到所有回测。

    单次回测 2.1 秒，装载 399k 根 K 线要 11.6 秒 ——
    每次重新装载会让 100 组网格从 4 分钟变成 23 分钟。
    """

    def __init__(self, cfg, strategy_cls, sector: str, start: str, end: str,
                 min_ipo_days: int, extra_params: dict | None = None,
                 universe_csv: str | None = None,
                 symbols: list[str] | None = None,
                 use_drawdown: bool = True) -> None:
        self.cfg = cfg
        # 验证必须带上回撤控制：不带的话验的是「另一套系统」，
        # 而实盘要跑的是带风控的那套。两者的参数最优区往往不同 ——
        # 风控会把深回撤的参数组直接砍掉
        self.use_drawdown = use_drawdown
        self.strategy_cls = strategy_cls
        #: 网格之外的固定参数（如 max_holdings），每次回测都会带上
        self.extra_params = extra_params or {}

        # 择时类策略只交易一两个标的，不需要也不该走成分股那套
        if symbols:
            self.symbols = symbols
            self.universe = StaticUniverse(symbols, source="命令行指定",
                                          from_index_snapshot=False)
            self._load_bars(cfg, start, end)
            return

        if universe_csv:
            self.universe = HistoricalUniverse(universe_csv)
            self.symbols = self.universe.all_symbols()
            self._load_bars(cfg, start, end)
            return

        meta_path = (Path(cfg.data.store_dir) / "universe"
                     / f"universe_{sector}.parquet")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"缺少标的元数据 {meta_path}，"
                f"请先运行 scripts/build_universe.py --sector {sector}")

        meta = pd.read_parquet(meta_path)
        self.symbols = meta["vt_symbol"].tolist()
        base = StaticUniverse(self.symbols, source=f"{sector} 当前成分快照")
        inclusion = ({r.vt_symbol: r.inclusion_date for r in meta.itertuples()
                      if pd.notna(getattr(r, "inclusion_date", None))}
                     if "inclusion_date" in meta.columns else {})
        self.universe = PointInTimeUniverse(
            base,
            {r.vt_symbol: r.listing_date for r in meta.itertuples()
             if pd.notna(r.listing_date)},
            {r.vt_symbol: r.delist_date for r in meta.itertuples()
             if pd.notna(r.delist_date)},
            min_days_since_ipo=min_ipo_days,
            inclusion_dates=inclusion)
        self._load_bars(cfg, start, end)

    def _load_bars(self, cfg, start: str, end: str) -> None:
        feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
        t0 = time.time()
        self.bars = feed.load_bars(self.symbols, start, end, Interval.DAILY)
        if not self.bars:
            raise RuntimeError("没有可用数据，请先运行 scripts/download_data.py")
        print(f"装载 {len(self.bars):,} 根 K 线（{time.time() - t0:.1f}s，"
              f"后续回测复用）")

    def _slice(self, start: str | None, end: str | None):
        if start is None and end is None:
            return self.bars
        lo = pd.Timestamp(start) if start else None
        hi = pd.Timestamp(end) if end else None
        return [b for b in self.bars
                if (lo is None or b.datetime >= lo)
                and (hi is None or b.datetime <= hi)]

    def run(self, params: dict, start: str | None = None,
            end: str | None = None, cost_multiplier: float = 1.0,
            warmup_days: int = 0):
        """回测一个区间。

        :param warmup_days: 在 start 之前额外装载多少个自然日用于预热，
            但**绩效只从 start 起算**。

            不预热的话，短窗口测出来的是「策略来不及热身」而不是
            「参数不泛化」—— 趋势过滤要 120 根 Bar 才能算出第一个值，
            6 个月的测试窗口约 120 个交易日，等热身完窗口就结束了，
            结果全是 0 成交。
        """
        load_start = start
        if start and warmup_days > 0:
            load_start = (pd.Timestamp(start)
                          - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")

        bars = self._slice(load_start, end)
        if len(bars) < 100:
            raise RuntimeError("区间内 K 线过少")

        base = self.cfg.cost
        cost = CostConfig(
            commission_rate=base.commission_rate * cost_multiplier,
            commission_min=base.commission_min * cost_multiplier,
            stamp_tax_rate=base.stamp_tax_rate * cost_multiplier,
            transfer_fee_rate=base.transfer_fee_rate * cost_multiplier,
            slippage_tick=max(1, int(base.slippage_tick * cost_multiplier)),
        )

        drawdown = self._make_drawdown()
        engine = BacktestEngine(
            initial_capital=self.cfg.backtest.initial_capital, cost=cost,
            drawdown=drawdown)
        engine.load_data(bars)
        engine.set_universe(self.universe)
        engine.add_strategy(self.strategy_cls, self.symbols,
                            dict(params, **self.extra_params))
        stats = engine.run()

        if warmup_days > 0 and start:
            stats = self._restat_from(engine, start)
        return stats

    def _make_drawdown(self):
        """每次回测都要新建控制器 —— 复用会把上一次的峰值带进来"""
        if not self.use_drawdown:
            return None
        r = self.cfg.risk
        return DrawdownController(DrawdownConfig(
            close_only_threshold=r.drawdown_close_only,
            reduce_threshold=r.drawdown_reduce,
            reduce_keep_ratio=r.drawdown_reduce_keep,
            flat_threshold=r.drawdown_flat,
            recovery_ratio=r.drawdown_recovery_ratio,
            min_observations=r.drawdown_min_observations,
            max_freeze_observations=r.drawdown_max_freeze))

    @staticmethod
    def _restat_from(engine, start: str):
        """只用 start 之后的净值与成交重算绩效，剔除预热期"""
        from qmtquant.engine.performance import calculate_stats

        lo = pd.Timestamp(start)
        equity = pd.Series(engine.equity_curve).sort_index()
        equity = equity[equity.index >= lo]
        if len(equity) < 2:
            raise RuntimeError("预热后剩余区间过短")

        trades = [t for t in engine.trades if t.datetime >= lo]
        # 基准资金用预热期末的净值，否则收益率会把预热期的盈亏算进去
        return calculate_stats(equity, trades, float(equity.iloc[0]))


def make_progress(label: str):
    t0 = time.time()

    def _p(done: int, total: int, _info=None) -> None:
        elapsed = time.time() - t0
        eta = (total - done) / (done / elapsed) if done and elapsed else 0
        filled = int(24 * done / total) if total else 0
        sys.stdout.write(f"\r  {label} [{'#' * filled}{'-' * (24 - filled)}] "
                         f"{done}/{total}  ETA {eta / 60:4.1f}min ")
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\n")
    return _p


def main() -> int:
    p = argparse.ArgumentParser(description="策略稳健性验证")
    p.add_argument("--strategy", default="mean_reversion",
                   help="短名或完整路径 pkg.mod.Class")
    p.add_argument("--grid", default=None,
                   help="参数网格 JSON，如 '{\"ma_window\": [20, 60]}'。"
                        "不给则用该策略的内置网格")
    p.add_argument("--set", action="append", dest="fixed", metavar="K=V",
                   help="网格之外的固定参数，可重复")
    p.add_argument("--symbols", default=None,
                   help="逗号分隔的标的代码。择时类策略用这个，"
                        "指定后不走成分股逻辑")
    p.add_argument("--universe-csv", default=None,
                   help="历史成分股 CSV（推荐，无幸存者偏差）")
    p.add_argument("--sector", default="沪深300")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-08-26")
    p.add_argument("--split", default="2024-09-01",
                   help="样本内/外的分界日")
    p.add_argument("--holdings", type=int, default=10)
    p.add_argument("--min-ipo-days", type=int, default=60)
    p.add_argument("--metric", default="Sharpe")
    p.add_argument("--max-drawdown", type=float, default=0.20,
                   help="回撤硬约束。违规的参数组合在寻优阶段就被剔除，"
                        "而不是选完再看")
    p.add_argument("--warmup", type=int, default=260,
                   help="测试窗口的预热自然日数。趋势过滤需 120 个交易日，"
                        "约合 180 自然日，留足余量")
    p.add_argument("--no-drawdown", action="store_true",
                   help="验证时关闭回撤控制。**默认开启** —— "
                        "不带风控验出来的是另一套系统")
    p.add_argument("--quick", action="store_true", help="用小网格快速跑")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["grid", "split", "walkforward", "cost"])
    p.add_argument("--out", default="reports")
    args = p.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    cls = load_strategy(args.strategy)
    fixed = parse_params(args.fixed)
    # 选股类策略需要 max_holdings；择时类没这个参数，加了会被 check 拦下
    if "max_holdings" in cls.parameters and "max_holdings" not in fixed:
        fixed["max_holdings"] = args.holdings
    check_params(cls, fixed)

    if args.grid:
        try:
            grid = json.loads(args.grid)
        except json.JSONDecodeError as e:
            print(f"--grid 不是合法 JSON：{e}")
            return 1
    else:
        grid = DEFAULT_GRIDS.get(cls.__name__)
        if grid is None:
            example = '{"%s": [...]}' % cls.parameters[0]
            print(f"{cls.__name__} 没有内置网格，请用 --grid 指定，例如：")
            print(f"  --grid '{example}'")
            print(f"  可用参数: {', '.join(cls.parameters)}")
            return 1
        if args.quick:
            grid = QUICK_GRIDS.get(cls.__name__, grid)
    check_params(cls, dict.fromkeys(grid))

    print(f"策略  : {cls.__module__}.{cls.__name__}")
    print(f"网格  : {grid}")
    print(f"固定  : {fixed or '无'}")
    print()

    try:
        syms = ([normalize(x.strip()) for x in args.symbols.split(",")
                 if x.strip()] if args.symbols else None)
        h = Harness(cfg, cls, args.sector, args.start, args.end,
                    args.min_ipo_days, fixed, args.universe_csv, syms,
                    use_drawdown=not args.no_drawdown)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sections = []

    # ---- 1. 参数网格与平原检验
    plateau = None
    # 网格每维取中位数作为兜底起点，避免 grid 检验被跳过时无参可用
    best_params = {k: v[len(v) // 2] for k, v in grid.items()}
    if "grid" not in args.skip:
        n = 1
        for v in grid.values():
            n *= len(v)
        print(f"\n>>> 1/4 参数网格（{n} 组组合，约 {n * 2.1 / 60:.0f} 分钟）")
        df = grid_search(lambda p: h.run(p), grid,
                         progress=make_progress("网格"))
        if df.empty:
            print("  网格全部失败")
            return 1

        df.to_csv(out / "validation_grid.csv", index=False,
                  encoding="utf-8-sig")
        plateau = parameter_plateau(df, list(grid), args.metric,
                                    max_drawdown_limit=args.max_drawdown)
        if not plateau.best_params:
            print()
            print(plateau.summary())
            print()
            print("  网格中回撤最小的 5 组（均不合规）：")
            cols = list(grid) + ["总收益", "最大回撤", "Sharpe", "成交数"]
            print(df.reindex(df["最大回撤"].abs().nsmallest(5).index)[cols]
                  .to_string(index=False,
                             formatters={"总收益": "{:.2%}".format,
                                         "最大回撤": "{:.2%}".format,
                                         "Sharpe": "{:.3f}".format}))
            return 1
        best_params = plateau.best_params
        print()
        print(plateau.summary())
        sections.append(plateau.summary())

        print("\n  网格中表现最好的 5 组（仅合规组合）：")
        cols = list(grid) + ["总收益", "年化", "最大回撤", "Sharpe", "成交数"]
        # 只列合规组合 —— 列出违规的「最优」会让人误以为它可用
        top = _eligible(df, args.max_drawdown).nlargest(5, args.metric)[cols]
        print(top.to_string(index=False,
                            formatters={"总收益": "{:.2%}".format,
                                        "年化": "{:.2%}".format,
                                        "最大回撤": "{:.2%}".format,
                                        "Sharpe": "{:.3f}".format}))

    # ---- 2. 样本外
    split = SplitReport(params=best_params)
    if "split" not in args.skip:
        print(f"\n>>> 2/4 样本外检验（分界 {args.split}）")
        split.in_sample = _to_result(
            best_params, h.run(best_params, args.start, args.split))
        split.out_sample = _to_result(
            best_params,
            h.run(best_params, args.split, args.end, warmup_days=args.warmup))
        print(split.summary())
        sections.append(split.summary())

    # ---- 3. Walk-forward
    from qmtquant.research.validation import WalkForwardReport
    wf = WalkForwardReport()
    if "walkforward" not in args.skip:
        # walk-forward 要在每个窗口重跑整个网格，组合数乘窗口数会爆炸 ——
        # 只保留前两维、每维至多 3 个取值
        wf_grid = {k: list(v)[:3] for k, v in list(grid.items())[:2]}
        wf_n = 1
        for v in wf_grid.values():
            wf_n *= len(v)
        print(f"\n>>> 3/4 Walk-forward（每窗口 {wf_n} 组组合）")
        # 训练窗口够长不必预热；测试窗口必须预热，否则测的是热身速度
        wf = walk_forward(
            lambda p, s, e: h.run(p, s, e, warmup_days=args.warmup), wf_grid,
            args.start, args.end, train_months=24, test_months=6,
            metric=args.metric, progress=make_progress("窗口"))
        print(wf.summary())
        sections.append(wf.summary())
        if wf.windows:
            wf.to_frame().to_csv(out / "validation_walkforward.csv",
                                 index=False, encoding="utf-8-sig")

    # ---- 4. 成本敏感性
    cost_df = pd.DataFrame()
    if "cost" not in args.skip:
        print("\n>>> 4/4 成本敏感性")
        cost_df = cost_sensitivity(
            lambda m: h.run(best_params, cost_multiplier=m))
        print(cost_df.to_string(
            index=False,
            formatters={"总收益": "{:.2%}".format, "年化": "{:.2%}".format,
                        "最大回撤": "{:.2%}".format, "Sharpe": "{:.3f}".format,
                        "费率": "{:.1%}".format}))
        cost_df.to_csv(out / "validation_cost.csv", index=False,
                       encoding="utf-8-sig")

    # ---- 综合结论
    if plateau is not None:
        print()
        verdict = summarize_verdict(plateau, split, wf, cost_df)
        print(verdict)
        sections.append(verdict)

    (out / "validation_report.txt").write_text(
        "\n\n".join(sections), encoding="utf-8")
    print(f"\n明细已输出到 {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
