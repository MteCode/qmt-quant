"""策略稳健性验证。

## 这个模块要回答的问题

**不是**「哪组参数收益最高」，而是「这个策略是否只在某个特定参数点上有效」。

一组参数在网格上是孤峰（邻域急剧变差）意味着它是拟合噪声的产物，
实盘里参数会微微漂移、市场会微微变化，收益就消失了。
真正稳健的策略应该呈现**参数平原**：一整片区域都能赚钱，
最优点只是平原上略高的一块。

## 四道检验

1. **参数平原** —— 邻域表现是否与最优点接近
2. **样本外验证** —— 用前段调参、后段检验，后段崩溃即为过拟合
3. **Walk-forward** —— 滚动前推，检验参数在时间上的稳定性
4. **成本敏感性** —— 滑点与手续费翻倍后是否还有正收益

任何一道不过，都不该上实盘。
"""
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """单次回测的关键指标"""
    params: dict
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    #: 手续费占初始资金比例，过高说明收益被交易成本吃掉
    fee_ratio: float = 0.0

    def to_row(self) -> dict:
        row = dict(self.params)
        row.update({
            "总收益": self.total_return,
            "年化": self.annual_return,
            "最大回撤": self.max_drawdown,
            "Sharpe": self.sharpe_ratio,
            "Calmar": self.calmar_ratio,
            "成交数": self.total_trades,
            "胜率": self.win_rate,
            "费率": self.fee_ratio,
        })
        return row


def _to_result(params: dict, stats) -> RunResult:
    fee = (stats.total_commission / stats.initial_capital
           if stats.initial_capital else 0.0)
    return RunResult(
        params=params,
        total_return=stats.total_return,
        annual_return=stats.annual_return,
        max_drawdown=stats.max_drawdown,
        sharpe_ratio=stats.sharpe_ratio,
        calmar_ratio=stats.calmar_ratio,
        total_trades=stats.total_trades,
        win_rate=stats.win_rate,
        fee_ratio=fee,
    )


def expand_grid(grid: dict[str, Iterable]) -> list[dict]:
    """把 {参数名: 候选值列表} 展开成参数组合列表"""
    keys = list(grid)
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(grid[k] for k in keys))]


def grid_search(runner: Callable[[dict], Any], grid: dict[str, Iterable],
                progress: Callable[[int, int, dict], None] | None = None,
                skip_invalid: bool = True) -> pd.DataFrame:
    """在参数网格上逐点回测。

    :param runner: 接受参数 dict、返回 PerformanceStats 的函数
    :param skip_invalid: 参数组合非法时跳过而非中断 ——
        网格里难免有违反约束的组合（如止损 > 入场阈值），
        中断会让整轮扫描白跑
    """
    combos = expand_grid(grid)
    rows = []
    for i, params in enumerate(combos, 1):
        if progress:
            progress(i, len(combos), params)
        try:
            stats = runner(params)
        except ValueError as exc:
            if not skip_invalid:
                raise
            logger.debug("跳过非法参数组合 %s: %s", params, exc)
            continue
        except Exception:
            logger.exception("回测失败，跳过该组合: %s", params)
            continue
        rows.append(_to_result(params, stats).to_row())

    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 参数平原


@dataclass
class PlateauReport:
    """参数平原检验结果"""
    metric: str
    best_params: dict = field(default_factory=dict)
    best_value: float = 0.0
    #: 邻域内该指标的中位数
    neighbor_median: float = 0.0
    #: 邻域相对最优点的保留比例。越接近 1 说明越像平原
    retention: float = 0.0
    #: 全网格中该指标为正的比例
    positive_ratio: float = 0.0
    neighbors: int = 0

    @property
    def is_plateau(self) -> bool:
        """邻域保留 60% 以上视为平原。

        阈值是经验值：低于此说明最优点周围迅速变差，
        参数稍有偏离收益就没了，属于典型的过拟合特征。
        """
        return self.retention >= 0.6

    def summary(self) -> str:
        verdict = "参数平原（稳健）" if self.is_plateau else "参数孤峰（疑似过拟合）"
        lines = [
            f"--- 参数平原检验（{self.metric}）---",
            f"最优参数     : {self.best_params}",
            f"最优值       : {self.best_value:.4f}",
            f"邻域中位数   : {self.neighbor_median:.4f}（{self.neighbors} 个邻居）",
            f"邻域保留比例 : {self.retention:.1%}",
            f"全网格为正   : {self.positive_ratio:.1%}",
            f"结论         : {verdict}",
        ]
        if not self.is_plateau:
            lines.append("  最优点周围迅速变差，参数稍有偏离收益就消失，"
                         "不应据此上实盘")
        return "\n".join(lines)


def parameter_plateau(df: pd.DataFrame, param_cols: list[str],
                      metric: str = "Sharpe") -> PlateauReport:
    """检验最优点是否落在参数平原上。

    做法：找到最优点，取网格上每个参数相差不超过一档的邻居，
    比较邻居的表现中位数与最优值。
    """
    if df.empty:
        return PlateauReport(metric=metric)

    best_idx = df[metric].idxmax()
    best = df.loc[best_idx]
    best_value = float(best[metric])

    # df.loc[idx] 返回的 Series 会把整行统一成同一 dtype，
    # 整数参数会变成 float64。用列的原始 dtype 还原
    def _restore(col: str):
        v = best[col]
        return int(v) if pd.api.types.is_integer_dtype(df[col]) else v

    # 每个参数各自的有序取值，用于定义「相邻一档」
    levels = {c: sorted(df[c].unique()) for c in param_cols}
    best_pos = {c: levels[c].index(best[c]) for c in param_cols}

    mask = pd.Series(True, index=df.index)
    for c in param_cols:
        pos = df[c].map(lambda v, c=c: levels[c].index(v))
        mask &= (pos - best_pos[c]).abs() <= 1
    neighbors = df[mask].drop(index=best_idx, errors="ignore")

    median = float(neighbors[metric].median()) if len(neighbors) else 0.0
    # 最优值可能为负，此时保留比例无意义，直接判为不稳健
    retention = (median / best_value) if best_value > 0 else 0.0

    return PlateauReport(
        metric=metric,
        best_params={c: _restore(c) for c in param_cols},
        best_value=best_value,
        neighbor_median=median,
        retention=max(retention, 0.0),
        positive_ratio=float((df[metric] > 0).mean()),
        neighbors=len(neighbors),
    )


# ---------------------------------------------------------------- 样本外


@dataclass
class SplitReport:
    """样本内/样本外对比"""
    params: dict = field(default_factory=dict)
    in_sample: RunResult | None = None
    out_sample: RunResult | None = None
    metric: str = "sharpe_ratio"

    @property
    def decay(self) -> float:
        """样本外相对样本内的衰减比例。1.0 = 完全没衰减"""
        if self.in_sample is None or self.out_sample is None:
            return 0.0
        a = getattr(self.in_sample, self.metric)
        b = getattr(self.out_sample, self.metric)
        return (b / a) if a > 0 else 0.0

    @property
    def passed(self) -> bool:
        """样本外仍为正、且衰减不超过一半"""
        if self.out_sample is None:
            return False
        return getattr(self.out_sample, self.metric) > 0 and self.decay >= 0.5

    def summary(self) -> str:
        if self.in_sample is None or self.out_sample is None:
            return "样本外检验：数据不足"
        a, b = self.in_sample, self.out_sample
        return "\n".join([
            "--- 样本外检验 ---",
            f"参数     : {self.params}",
            f"{'':10}{'样本内':>12}{'样本外':>12}",
            f"{'总收益':<10}{a.total_return:>11.2%}{b.total_return:>12.2%}",
            f"{'年化':<10}{a.annual_return:>11.2%}{b.annual_return:>12.2%}",
            f"{'最大回撤':<10}{a.max_drawdown:>11.2%}{b.max_drawdown:>12.2%}",
            f"{'Sharpe':<10}{a.sharpe_ratio:>11.3f}{b.sharpe_ratio:>12.3f}",
            f"{'成交数':<10}{a.total_trades:>11}{b.total_trades:>12}",
            f"衰减     : {self.decay:.1%}"
            f"    结论：{'通过' if self.passed else '未通过'}",
        ])


# ---------------------------------------------------------------- Walk-forward


@dataclass
class WalkForwardReport:
    """滚动前推验证结果"""
    windows: list[dict] = field(default_factory=list)

    @property
    def positive_windows(self) -> int:
        return sum(1 for w in self.windows if w["test_return"] > 0)

    @property
    def consistency(self) -> float:
        """盈利窗口占比。衡量的是「时间上的稳定性」而非总收益 ——
        靠一个窗口暴赚、其余全亏的策略不可持续。
        """
        return (self.positive_windows / len(self.windows)
                if self.windows else 0.0)

    @property
    def passed(self) -> bool:
        """过半窗口盈利"""
        return self.consistency >= 0.5

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.windows)

    def summary(self) -> str:
        if not self.windows:
            return "Walk-forward：无有效窗口"
        lines = ["--- Walk-forward 滚动验证 ---",
                 f"{'训练区间':<24}{'测试区间':<24}{'测试收益':>10}{'最优参数'}"]
        for w in self.windows:
            lines.append(
                f"{w['train']:<24}{w['test']:<24}"
                f"{w['test_return']:>10.2%}  {w['best_params']}")
        lines.append(f"盈利窗口 {self.positive_windows}/{len(self.windows)}"
                     f"（{self.consistency:.0%}）"
                     f"    结论：{'通过' if self.passed else '未通过'}")
        if not self.passed:
            lines.append("  参数在时间上不稳定，前段最优的参数到后段就失效")
        return "\n".join(lines)


def walk_forward(runner: Callable[[dict, str, str], Any],
                 grid: dict[str, Iterable],
                 start: str, end: str,
                 train_months: int = 24, test_months: int = 6,
                 metric: str = "Sharpe",
                 progress: Callable[[int, int, str], None] | None = None
                 ) -> WalkForwardReport:
    """滚动前推验证。

    每个窗口：用训练区间做网格寻优，把最优参数拿到**紧接着的**测试区间上跑。
    测试区间的数据在寻优时完全没被看到，因此结果是真正的样本外表现。

    :param runner: (参数, 起始日, 结束日) -> PerformanceStats
    """
    report = WalkForwardReport()
    cur = pd.Timestamp(start)
    final = pd.Timestamp(end)
    windows: list[tuple] = []

    while True:
        train_end = cur + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > final:
            break
        windows.append((cur, train_end, test_end))
        cur = cur + pd.DateOffset(months=test_months)

    for i, (t0, t1, t2) in enumerate(windows, 1):
        label = f"{t0:%Y-%m} ~ {t1:%Y-%m}"
        if progress:
            progress(i, len(windows), label)

        train = grid_search(
            lambda p, a=t0, b=t1: runner(p, a.strftime("%Y-%m-%d"),
                                         b.strftime("%Y-%m-%d")),
            grid)
        if train.empty:
            continue

        param_cols = list(grid)
        best = train.loc[train[metric].idxmax()]
        best_params = {
            c: (int(best[c]) if pd.api.types.is_integer_dtype(train[c])
                else best[c])
            for c in param_cols}

        try:
            stats = runner(best_params, t1.strftime("%Y-%m-%d"),
                           t2.strftime("%Y-%m-%d"))
        except Exception:
            logger.exception("测试区间回测失败: %s", best_params)
            continue

        report.windows.append({
            "train": label,
            "test": f"{t1:%Y-%m} ~ {t2:%Y-%m}",
            "train_metric": float(best[metric]),
            "test_return": stats.total_return,
            "test_sharpe": stats.sharpe_ratio,
            "best_params": best_params,
        })

    return report


# ---------------------------------------------------------------- 成本敏感性


def cost_sensitivity(runner: Callable[[float], Any],
                     multipliers: Iterable[float] = (1, 2, 3, 5)
                     ) -> pd.DataFrame:
    """交易成本敏感性。

    把滑点与手续费成倍放大，看策略是否还有正收益。
    回测的成本模型总是偏乐观（真实还有冲击成本、部分成交、拒单重试），
    成本翻倍即亏损的策略在实盘几乎必然亏钱。
    """
    rows = []
    for m in multipliers:
        try:
            stats = runner(m)
        except Exception:
            logger.exception("成本倍数 %s 回测失败", m)
            continue
        rows.append({
            "成本倍数": m,
            "总收益": stats.total_return,
            "年化": stats.annual_return,
            "Sharpe": stats.sharpe_ratio,
            "最大回撤": stats.max_drawdown,
            "成交数": stats.total_trades,
            "费率": (stats.total_commission / stats.initial_capital
                     if stats.initial_capital else 0),
        })
    return pd.DataFrame(rows)


def cost_breakeven(df: pd.DataFrame) -> float | None:
    """成本放大到几倍时收益转负。None 表示测试范围内始终为正。"""
    if df.empty or "总收益" not in df:
        return None
    negative = df[df["总收益"] <= 0]
    return float(negative["成本倍数"].min()) if len(negative) else None


def summarize_verdict(plateau: PlateauReport, split: SplitReport,
                      wf: WalkForwardReport, cost: pd.DataFrame) -> str:
    """四道检验的综合结论"""
    breakeven = cost_breakeven(cost)
    checks = [
        ("参数平原", plateau.is_plateau,
         f"邻域保留 {plateau.retention:.0%}"),
        ("样本外", split.passed,
         f"衰减 {split.decay:.0%}"),
        ("Walk-forward", wf.passed,
         f"盈利窗口 {wf.consistency:.0%}"),
        ("成本承受", breakeven is None or breakeven >= 3,
         "全范围为正" if breakeven is None else f"{breakeven:.0f} 倍时转负"),
    ]
    lines = ["=" * 56, "稳健性综合结论", "=" * 56]
    for name, ok, detail in checks:
        lines.append(f"  [{'通过' if ok else '未通过'}] {name:<14}{detail}")

    passed = sum(1 for _, ok, _ in checks if ok)
    lines.append("-" * 56)
    lines.append(f"  {passed}/4 项通过")
    if passed == 4:
        lines.append("  四项全通过。可以考虑小资金实盘验证，"
                     "但仍需注意标的池偏差与样本期长度限制。")
    else:
        lines.append("  存在未通过项，不建议上实盘。"
                     "先弄清楚是策略逻辑问题还是参数拟合噪声。")
    return "\n".join(lines)
