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

import numpy as np
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
    #: 满足回撤硬约束的组合数
    eligible: int = 0
    #: 网格总组合数
    total: int = 0
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
            (f"合规组合     : {self.eligible}/{self.total}"
             " 组满足回撤上限" if self.total else "合规组合     : —"),
            f"最优参数     : {self.best_params}",
            f"最优值       : {self.best_value:.4f}",
            f"邻域中位数   : {self.neighbor_median:.4f}（{self.neighbors} 个邻居）",
            f"邻域保留比例 : {self.retention:.1%}",
            f"全网格为正   : {self.positive_ratio:.1%}",
            f"结论         : {verdict}",
        ]
        if self.total and self.eligible == 0:
            lines.append("  ⚠ **没有任何参数组合满足回撤上限** —— "
                         "该策略在此网格上不存在可上实盘的参数")
        elif not self.is_plateau:
            lines.append("  最优点周围迅速变差，参数稍有偏离收益就消失，"
                         "不应据此上实盘")
        return "\n".join(lines)


def filter_by_drawdown(df: pd.DataFrame,
                       max_drawdown_limit: float = 0.20) -> pd.DataFrame:
    """剔除违反回撤上限的参数组合。

    **硬约束必须在寻优之前生效，而不是选完再看。**
    只按 Sharpe 挑最优，挑出来的很可能是一组回撤 -27% 的参数 ——
    它在约束下根本不可用，却被当成「最优点」去做平原检验，
    整个结论都建立在一个不能上实盘的点上。
    """
    if df.empty or "最大回撤" not in df.columns:
        return df
    keep = df["最大回撤"].abs() <= max_drawdown_limit + 1e-9
    dropped = int((~keep).sum())
    if dropped:
        logger.info("回撤约束剔除 %d/%d 组参数（上限 %.0f%%）",
                    dropped, len(df), max_drawdown_limit * 100)
    return df[keep]


def parameter_plateau(df: pd.DataFrame, param_cols: list[str],
                      metric: str = "Sharpe",
                      max_drawdown_limit: float | None = 0.20) -> PlateauReport:
    """检验最优点是否落在参数平原上。

    做法：找到最优点，取网格上每个参数相差不超过一档的邻居，
    比较邻居的表现中位数与最优值。

    :param max_drawdown_limit: 回撤硬约束。**先剔除违规组合再找最优点** ——
        否则会拿一个不能上实盘的参数点去做平原检验。
        传 None 则不施加约束。
    """
    if df.empty:
        return PlateauReport(metric=metric)

    full = df
    if max_drawdown_limit is not None:
        df = filter_by_drawdown(df, max_drawdown_limit)
        if df.empty:
            # 全部违规也是一个明确结论：这个策略在该网格上不存在合规参数
            return PlateauReport(
                metric=metric,
                positive_ratio=float((full[metric] > 0).mean()),
                eligible=0, total=len(full))

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
        eligible=len(df),
        total=len(full),
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
                      wf: WalkForwardReport, cost: pd.DataFrame,
                      overfit: "OverfitReport | None" = None) -> str:
    """五道检验的综合结论。

    自由度是**前置条件**：数据不够拟合这么多参数时，
    后面四项无论结果如何都不能采信 —— 平原也可以是噪声形成的。
    """
    breakeven = cost_breakeven(cost)
    checks = []
    if overfit is not None:
        checks.append(("自由度", overfit.passed,
                       f"{overfit.n_params} 个参数 / 预算 "
                       f"{overfit.budget:.1f} 个"))
    checks += [
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


# ---------------------------------------------------------------- 过拟合

#: 每个可优化参数至少需要多少个独立市场状态才谈得上「拟合」而非「记忆」。
#: A 股约每 6 个月切换一次风格，10 年数据 ≈ 20 个独立状态。
#: 经验法则取 10：拟合 3 个参数需要 30 个状态，即 15 年数据
OBSERVATIONS_PER_PARAM = 10
#: 一个交易日不是一个独立观测。趋势/风格的持续期约一个季度，
#: 60 个交易日折算成 1 个独立状态
TRADING_DAYS_PER_REGIME = 60


@dataclass
class OverfitReport:
    """自由度检验：参数个数与可用信息量是否匹配。

    ## 为什么这一项要单独看

    参数平原检验回答的是「这组参数稳不稳」，但**它无法发现
    「整个网格都是噪声」这种情况** —— 噪声也可以形成平原。

    自由度检验回答的是更前置的问题：**这么多参数，数据够不够拟合。**
    不够的话，无论平原检验结果如何，得到的都是记忆而非规律。

    ## 交易日不等于独立观测

    2588 个交易日看着很多，但相邻交易日高度相关。
    A 股风格切换周期约一个季度，折算下来 10 年只有 40 个左右的独立状态。
    用它去拟合 14 个参数，等于用 40 个点拟合 14 维曲面。
    """

    #: 参与寻优的参数个数
    n_params: int = 0
    #: 策略声明的全部参数个数（含未寻优的）
    n_declared: int = 0
    #: 回测覆盖的交易日数
    trading_days: int = 0
    #: 网格组合数
    n_combos: int = 0

    @property
    def independent_regimes(self) -> float:
        """折算后的独立市场状态数"""
        return self.trading_days / TRADING_DAYS_PER_REGIME

    @property
    def budget(self) -> float:
        """按自由度可支持的参数个数上限"""
        return self.independent_regimes / OBSERVATIONS_PER_PARAM

    @property
    def passed(self) -> bool:
        """参与寻优的参数个数是否在预算之内"""
        return self.n_params <= self.budget + 1e-9

    @property
    def selection_bias(self) -> float:
        """多重比较带来的虚假发现风险。

        在 N 组参数里挑最好的一组，即使全是噪声，
        最好那组的表现也会显著优于均值。N 越大，这个虚高越严重。
        用 sqrt(2*ln(N)) 近似 N 个标准正态变量最大值的期望 ——
        扫 100 组参数，最优点的 Sharpe 天然被抬高约 3 个标准差。
        """
        if self.n_combos < 2:
            return 0.0
        return float(np.sqrt(2 * np.log(self.n_combos)))

    def summary(self) -> str:
        lines = [
            "--- 自由度检验 ---",
            f"寻优参数     : {self.n_params} 个"
            f"（策略共声明 {self.n_declared} 个）",
            f"交易日       : {self.trading_days}",
            f"独立状态     : {self.independent_regimes:.0f} 个"
            f"（每 {TRADING_DAYS_PER_REGIME} 个交易日折算 1 个）",
            f"参数预算     : {self.budget:.1f} 个",
            f"网格组合     : {self.n_combos} 组，"
            f"选优虚高约 {self.selection_bias:.1f} 个标准差",
            f"结论         : {'自由度充足' if self.passed else '**自由度不足**'}",
        ]
        if not self.passed:
            lines.append(f"  数据只够拟合 {self.budget:.1f} 个参数，"
                         f"实际寻优 {self.n_params} 个 —— "
                         "得到的是记忆而非规律，平原检验通过也不能采信")
        return "\n".join(lines)


def check_overfit(n_params: int, trading_days: int, n_combos: int,
                  n_declared: int = 0) -> OverfitReport:
    return OverfitReport(n_params=n_params, n_declared=n_declared or n_params,
                         trading_days=trading_days, n_combos=n_combos)


def robust_params(df: pd.DataFrame, param_cols: list[str],
                  metric: str = "Sharpe",
                  max_drawdown_limit: float | None = 0.20) -> dict:
    """选**平原中心**而非峰值。

    ## 为什么不能选最高分那组

    在 N 组参数里挑表现最好的一组，即使全部是噪声，
    最好那组看起来也会很好 —— 这是多重比较，不是发现。
    扫 100 组参数，最优点的 Sharpe 天然被抬高约 3 个标准差。

    本函数改为给每组参数打「邻域中位数」分：
    一组参数只有在**它自己和周围的参数都还行**时才得高分。
    孤峰的邻域中位数很低，会被自动淘汰。

    这不能凭空造出 alpha，但能避免把噪声的尖峰当成发现。
    """
    if df.empty:
        return {}
    if max_drawdown_limit is not None:
        df = filter_by_drawdown(df, max_drawdown_limit)
        if df.empty:
            return {}

    levels = {c: sorted(df[c].unique()) for c in param_cols}
    pos = {c: df[c].map(lambda v, c=c: levels[c].index(v)) for c in param_cols}

    best_score, best_row = -np.inf, None
    for idx in df.index:
        mask = pd.Series(True, index=df.index)
        for c in param_cols:
            mask &= (pos[c] - pos[c][idx]).abs() <= 1
        # 含自身在内取中位数：单点再高，周围塌了也拉不起来
        score = float(df.loc[mask, metric].median())
        if score > best_score:
            best_score, best_row = score, df.loc[idx]

    if best_row is None:
        return {}
    return {c: (int(best_row[c])
                if pd.api.types.is_integer_dtype(df[c]) else best_row[c])
            for c in param_cols}
