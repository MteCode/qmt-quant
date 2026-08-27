"""因子 IC 分析。

## IC 是什么，为什么先看它

IC（Information Coefficient）= 因子值与**未来收益**的横截面秩相关。
每个调仓日算一次，得到一条 IC 时间序列。

它回答的问题是：「按这个因子给股票排序，排在前面的后来是不是真的涨得多」。

**先做 IC 再写策略**，理由是成本：一次 IC 分析几秒钟，
一次完整回测几分钟，而且回测结果掺杂了仓位、成本、调仓频率等
一堆与因子本身无关的因素 —— 因子没有预测力的话，
回测出来的任何盈亏都只是噪声的形状。

实测教训：沪深300 的横截面动量 IC = -0.0575（t = -4.6），
在写策略之前就该知道它是负的。后来的回测 -24.20% 只是把这件事又证了一遍。

## 怎么读这些数字

============  ========================================
指标           含义
============  ========================================
IC 均值        方向与强度。A 股上 |IC| > 0.03 就算有信号
IC t 值        显著性。|t| > 2 才谈得上稳定
IC 胜率        IC 同号的比例，衡量方向是否稳定
ICIR          IC 均值 / IC 标准差，信息比率
分组单调性      Q1~Q5 收益是否随因子值单调变化
============  ========================================

**t 值比均值重要。** IC 均值 0.05 但 t 值 0.8，意味着它在不同时期
忽正忽负，平均下来碰巧为正 —— 这种因子做成策略必然失效。

## 重叠样本：t 值最容易被高估的地方

日频取截面 + 60 日未来收益 = 相邻两个观测重叠 59 天，
IC 序列有极强的自相关。此时 ``t = ICIR × sqrt(N)`` 把 671 个
高度相关的观测当成 671 个独立样本，t 值会被放大数倍。

实测：未修正时 EP/60日 的 t 值高达 20.1，修正后掉到个位数 ——
**结论从「铁证如山」变成「值得一看」**。

本模块用 Newey-West（Bartlett 核，滞后阶数 = horizon-1）修正，
这是这类研究的标准做法。``ic_t_naive`` 保留未修正值供对照。

## 用秩相关而非皮尔逊相关

因子分布普遍有极端值（PE 可以是几千，也可以是负的），
皮尔逊相关会被少数几个离群点主导。秩相关只看排序，天然抗极值 ——
而选股本来就只关心排序。
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: A 股上因子 IC 的经验阈值。超过即认为有可用信号
IC_MEANINGFUL = 0.03
#: t 检验的显著性门槛
T_SIGNIFICANT = 2.0


@dataclass
class ICReport:
    """单个因子在单个预测周期上的 IC 分析结果"""

    factor: str = ""
    horizon: int = 0
    #: 有效横截面数（调仓日数）
    periods: int = 0
    #: 每个横截面的平均标的数
    avg_universe: float = 0.0

    ic_mean: float = 0.0
    ic_std: float = 0.0
    #: Newey-West 修正后的 t 值。重叠样本下这才是可信的那个
    ic_t: float = 0.0
    #: 未修正的 t 值，仅供对照 —— 重叠样本下它会被严重高估
    ic_t_naive: float = 0.0
    #: IC 与均值同号的比例
    ic_win_rate: float = 0.0
    #: 信息比率 = ic_mean / ic_std
    icir: float = 0.0

    #: 按因子值五等分后各组的平均未来收益（Q1 = 因子值最小）
    quantile_returns: list[float] = field(default_factory=list)
    #: IC 时间序列，供画图或进一步分析
    ic_series: pd.Series = field(default_factory=pd.Series)

    @property
    def is_significant(self) -> bool:
        """有统计显著性 —— 但方向可能是负的"""
        return abs(self.ic_t) >= T_SIGNIFICANT

    @property
    def is_meaningful(self) -> bool:
        """既显著又有足够强度，才值得写成策略"""
        return self.is_significant and abs(self.ic_mean) >= IC_MEANINGFUL

    @property
    def monotonic(self) -> bool:
        """分组收益是否单调。单调说明因子在整个取值范围内都有区分度，
        而不是只有极端组有效 —— 后者往往是少数异常样本造成的"""
        q = self.quantile_returns
        if len(q) < 3:
            return False
        inc = all(q[i] <= q[i + 1] for i in range(len(q) - 1))
        dec = all(q[i] >= q[i + 1] for i in range(len(q) - 1))
        return inc or dec

    @property
    def long_short_return(self) -> float:
        """多空组合收益：最高组减最低组。因子的理论收益上限"""
        if len(self.quantile_returns) < 2:
            return 0.0
        return self.quantile_returns[-1] - self.quantile_returns[0]

    def verdict(self) -> str:
        if not self.is_significant:
            return "无信号（不显著，IC 在不同时期忽正忽负）"
        direction = "正向" if self.ic_mean > 0 else "**反向**"
        if not self.is_meaningful:
            return f"{direction}但过弱（显著却不足以覆盖交易成本）"
        mono = "单调" if self.monotonic else "非单调（仅极端组有效，慎用）"
        return f"{direction}有效 · {mono}"

    def summary(self) -> str:
        lines = [
            f"--- IC 分析：{self.factor} / 未来 {self.horizon} 日 ---",
            f"横截面数      : {self.periods}（平均 {self.avg_universe:.0f} 只）",
            f"IC 均值       : {self.ic_mean:+.4f}",
            f"IC 标准差     : {self.ic_std:.4f}",
            f"IC t 值(NW)   : {self.ic_t:+.2f}"
            + ("  ✓ 显著" if self.is_significant else "  ✗ 不显著")
            + (f"   [未修正 {self.ic_t_naive:+.2f}]"
               if abs(self.ic_t_naive - self.ic_t) > 0.5 else ""),
            f"IC 胜率       : {self.ic_win_rate:.1%}",
            f"ICIR          : {self.icir:+.3f}",
        ]
        if self.quantile_returns:
            qs = "  ".join(f"Q{i + 1} {r:+.2%}"
                           for i, r in enumerate(self.quantile_returns))
            lines.append(f"分组收益      : {qs}")
            lines.append(f"多空收益      : {self.long_short_return:+.2%}"
                         + ("  单调" if self.monotonic else "  非单调"))
        lines.append(f"结论          : {self.verdict()}")
        return "\n".join(lines)


def _newey_west_t(x: np.ndarray, lags: int) -> float:
    """Newey-West 修正的单样本 t 值（Bartlett 核）。

    普通 t 检验假设观测独立。重叠的未来收益窗口破坏了这个假设 ——
    用 60 日收益、每日取一个截面，相邻观测共享 59 天的行情，
    自相关极强。不修正的话 t 值会被放大数倍，让噪声看起来像铁证。

    :param lags: 滞后阶数，取 horizon-1（重叠的天数）
    """
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    # 自协方差的加权和，权重随滞后线性衰减（Bartlett）
    s = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1)
        s += 2.0 * weight * float(e[lag:] @ e[:-lag]) / n
    # 加权和可能为负（小样本下 Bartlett 核不保证半正定），此时退回未修正
    if s <= 0:
        return float("nan")
    return mu / np.sqrt(s / n)


def compute_ic(factor: pd.DataFrame, forward_returns: pd.DataFrame,
               factor_name: str = "", horizon: int = 0,
               quantiles: int = 5, min_universe: int = 20) -> ICReport:
    """计算因子 IC。

    :param factor: index=日期, columns=vt_symbol 的因子值矩阵
    :param forward_returns: 同形状的未来收益矩阵
    :param quantiles: 分组数，5 即五等分
    :param min_universe: 单个横截面至少要有多少只有效标的。
        太少时秩相关的抽样误差极大，算出来的 IC 是噪声
    :return: ICReport
    """
    report = ICReport(factor=factor_name, horizon=horizon)

    # 对齐到共同的日期与标的，避免因维度不一致算出无意义的相关
    dates = factor.index.intersection(forward_returns.index)
    cols = factor.columns.intersection(forward_returns.columns)
    if len(dates) == 0 or len(cols) == 0:
        logger.warning("%s 因子与收益无重叠，无法计算 IC", factor_name)
        return report

    f = factor.loc[dates, cols]
    r = forward_returns.loc[dates, cols]

    ics: list[float] = []
    ic_dates: list = []
    universe_sizes: list[int] = []
    # 累加各组收益，最后取平均。逐日归组比全样本分组更正确 ——
    # 因子的绝对水平会随市场整体涨跌漂移，跨期比较绝对值没有意义
    q_sums = np.zeros(quantiles)
    q_counts = np.zeros(quantiles)

    for d in dates:
        fv, rv = f.loc[d], r.loc[d]
        mask = fv.notna() & rv.notna()
        n = int(mask.sum())
        if n < min_universe:
            continue

        fv, rv = fv[mask], rv[mask]
        # 秩相关：抗极值，且选股本来就只关心排序
        ic = fv.rank().corr(rv.rank())
        if pd.isna(ic):
            continue

        ics.append(float(ic))
        ic_dates.append(d)
        universe_sizes.append(n)

        # 分组。qcut 在因子值大量重复时会失败（如股息率一片 0），
        # 退化为按秩均分，保证任何分布下都能分组
        try:
            groups = pd.qcut(fv.rank(method="first"), quantiles,
                             labels=False, duplicates="drop")
        except ValueError:
            continue
        for g in range(quantiles):
            sel = rv[groups == g]
            if len(sel):
                q_sums[g] += sel.mean()
                q_counts[g] += 1

    if not ics:
        logger.warning("%s 没有足够的有效横截面（需每期至少 %d 只）",
                       factor_name, min_universe)
        return report

    series = pd.Series(ics, index=pd.DatetimeIndex(ic_dates)).sort_index()
    report.ic_series = series
    report.periods = len(series)
    report.avg_universe = float(np.mean(universe_sizes))
    report.ic_mean = float(series.mean())
    report.ic_std = float(series.std(ddof=1)) if len(series) > 1 else 0.0

    if report.ic_std > 0:
        report.icir = report.ic_mean / report.ic_std
        report.ic_t_naive = report.icir * np.sqrt(report.periods)
        # 重叠样本会让未修正的 t 值严重高估，见模块文档
        nw = _newey_west_t(series.to_numpy(), lags=max(horizon - 1, 0))
        report.ic_t = nw if np.isfinite(nw) else report.ic_t_naive
    elif report.ic_mean != 0:
        # IC 每期完全相同（标准差为 0）意味着**方向绝对稳定**，
        # 是显著性的极限而非缺失。若按 t=0 处理，
        # 一个恒为 1.0 的完美因子会被判成「无信号」
        report.icir = np.inf * np.sign(report.ic_mean)
        report.ic_t = report.ic_t_naive = report.icir
    report.ic_win_rate = float((np.sign(series) == np.sign(report.ic_mean)).mean())

    with np.errstate(invalid="ignore"):
        report.quantile_returns = [
            float(s / c) if c else float("nan")
            for s, c in zip(q_sums, q_counts)
        ]
    return report


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """未来 N 期收益矩阵。

    ``shift(-horizon)`` 是这里唯一容易出错的地方：符号写反就成了
    **过去**收益，IC 会变成自相关，数值好看得离谱却毫无意义。
    """
    if horizon < 1:
        raise ValueError(f"horizon 至少为 1，实际 {horizon}")
    return prices.shift(-horizon) / prices - 1


def analyze(factor: pd.DataFrame, prices: pd.DataFrame,
            factor_name: str, horizons: list[int],
            **kwargs) -> list[ICReport]:
    """在多个预测周期上分析同一个因子。

    多周期一起看是为了识别「只在某个特定周期有效」的假信号 ——
    真实的因子通常在相邻周期上表现连续。
    """
    reports = []
    for h in horizons:
        fr = forward_returns(prices, h)
        reports.append(compute_ic(factor, fr, factor_name, h, **kwargs))
    return reports


def compare_table(reports: list[ICReport]) -> pd.DataFrame:
    """把多份报告拼成一张表，便于横向比较"""
    return pd.DataFrame([{
        "因子": r.factor,
        "周期": r.horizon,
        "IC均值": r.ic_mean,
        "IC_t(NW)": r.ic_t,
        "IC_t未修正": r.ic_t_naive,
        "ICIR": r.icir,
        "IC胜率": r.ic_win_rate,
        "多空收益": r.long_short_return,
        "单调": r.monotonic,
        "横截面数": r.periods,
        "结论": r.verdict(),
    } for r in reports])
