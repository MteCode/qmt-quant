"""因子 IC 分析测试。

最关键的一条是 shift 方向：`forward_returns` 符号写反就成了**过去**收益，
IC 变成自相关，数值好看得离谱却毫无意义 —— 而且不会报任何错。
"""
import numpy as np
import pandas as pd
import pytest

from qmtquant.research.factor_ic import (
    ICReport,
    analyze,
    compare_table,
    compute_ic,
    forward_returns,
)

SYMBOLS = [f"{i:06d}.SZSE" for i in range(50)]
DATES = pd.bdate_range("2024-01-02", periods=60)


def frame(values) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES, columns=SYMBOLS)


def random_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return frame(rng.normal(size=(len(DATES), len(SYMBOLS))))


class TestForwardReturns:
    def test_looks_forward_not_backward(self):
        """符号写反会变成过去收益 —— IC 变自相关，不报错但结论全废"""
        prices = frame(np.tile(np.arange(1, len(DATES) + 1)[:, None],
                               (1, len(SYMBOLS))).astype(float))
        fr = forward_returns(prices, 1)
        # 第 0 行：价格 1 -> 2，未来 1 日收益应为 +100%
        assert fr.iloc[0, 0] == pytest.approx(1.0)
        # 最后一行没有未来，必须是 NaN 而不是 0
        assert pd.isna(fr.iloc[-1, 0])

    def test_horizon_respected(self):
        prices = frame(np.tile(np.arange(1, len(DATES) + 1)[:, None],
                               (1, len(SYMBOLS))).astype(float))
        # 价格 1 -> 6，5 日收益 = 500%
        assert forward_returns(prices, 5).iloc[0, 0] == pytest.approx(5.0)

    def test_zero_horizon_rejected(self):
        with pytest.raises(ValueError, match="horizon"):
            forward_returns(frame(np.ones((len(DATES), len(SYMBOLS)))), 0)


class TestComputeIC:
    def test_perfect_positive_factor(self):
        """因子与未来收益完全同序时 IC 应接近 +1"""
        f = random_frame(1)
        r = f * 2.0            # 严格单调变换，秩完全一致
        rep = compute_ic(f, r, "perfect", 1)
        assert rep.ic_mean == pytest.approx(1.0, abs=1e-6)
        assert rep.is_meaningful
        assert "正向" in rep.verdict()

    def test_perfect_negative_factor(self):
        f = random_frame(2)
        rep = compute_ic(f, -f, "inverse", 1)
        assert rep.ic_mean == pytest.approx(-1.0, abs=1e-6)
        assert "反向" in rep.verdict()

    def test_pure_noise_not_significant(self):
        """无关的两组随机数不应被判为有信号"""
        rep = compute_ic(random_frame(3), random_frame(4), "noise", 1)
        assert abs(rep.ic_mean) < 0.1
        assert not rep.is_meaningful

    def test_rank_based_resists_outliers(self):
        """一个极端值不应主导结论 —— 这正是用秩相关的理由"""
        f = random_frame(5)
        r = f * 2.0
        f.iloc[:, 0] = 1e12     # 单只股票因子值离谱
        rep = compute_ic(f, r, "outlier", 1)
        assert rep.ic_mean > 0.9, "秩相关应基本不受极端值影响"

    def test_skips_thin_cross_sections(self):
        """标的太少时秩相关是噪声，必须跳过而非算出一个假 IC"""
        f = random_frame(6)
        r = random_frame(7)
        f.iloc[:, 5:] = np.nan      # 每期只剩 5 只
        rep = compute_ic(f, r, "thin", 1, min_universe=20)
        assert rep.periods == 0

    def test_nan_alignment(self):
        f = random_frame(8)
        r = f * 2.0
        r.iloc[:, :10] = np.nan
        rep = compute_ic(f, r, "partial", 1)
        assert rep.periods > 0
        assert rep.avg_universe == pytest.approx(40)

    def test_no_overlap_returns_empty(self):
        f = random_frame(9)
        r = f.copy()
        r.index = r.index + pd.Timedelta(days=5000)
        rep = compute_ic(f, r, "disjoint", 1)
        assert rep.periods == 0
        assert rep.ic_mean == 0.0


class TestSignificance:
    def _report(self, ic_mean, ic_std, periods):
        rep = ICReport(ic_mean=ic_mean, ic_std=ic_std, periods=periods)
        rep.icir = ic_mean / ic_std
        rep.ic_t = rep.icir * np.sqrt(periods)
        return rep

    def test_strong_but_unstable_is_not_meaningful(self):
        """IC 均值 0.05 但忽正忽负 —— 做成策略必然失效"""
        rep = self._report(0.05, 0.40, 30)
        assert abs(rep.ic_mean) >= 0.03
        assert not rep.is_significant
        assert "无信号" in rep.verdict()

    def test_significant_but_too_weak(self):
        """显著但强度不足以覆盖交易成本"""
        rep = self._report(0.005, 0.02, 500)
        assert rep.is_significant
        assert not rep.is_meaningful
        assert "过弱" in rep.verdict()

    def test_significant_and_strong(self):
        rep = self._report(0.06, 0.15, 200)
        assert rep.is_meaningful


class TestQuantiles:
    def test_monotonic_detected(self):
        rep = ICReport(quantile_returns=[-0.02, -0.01, 0.0, 0.01, 0.02])
        assert rep.monotonic
        assert rep.long_short_return == pytest.approx(0.04)

    def test_non_monotonic_detected(self):
        """只有极端组有效 —— 往往是少数异常样本造成的"""
        rep = ICReport(quantile_returns=[-0.05, 0.01, 0.0, 0.01, 0.05])
        assert not rep.monotonic

    def test_decreasing_is_monotonic(self):
        assert ICReport(quantile_returns=[0.03, 0.02, 0.01, 0.0]).monotonic

    def test_quantiles_ordered_by_factor(self):
        """Q1 应对应因子值最小的一组"""
        f = random_frame(11)
        rep = compute_ic(f, f * 2.0, "mono", 1)
        q = rep.quantile_returns
        assert q[0] < q[-1], "因子与收益同向时 Q5 应高于 Q1"
        assert rep.monotonic

    def test_duplicate_factor_values_handled(self):
        """股息率常常一片 0，qcut 会失败 —— 不能因此崩溃"""
        f = frame(np.zeros((len(DATES), len(SYMBOLS))))
        f.iloc[:, :10] = 1.0
        rep = compute_ic(f, random_frame(12), "dup", 1)
        assert rep.periods > 0


class TestAnalyze:
    def test_multiple_horizons(self):
        prices = frame(np.cumprod(
            1 + np.random.default_rng(13).normal(0, 0.01,
                                                 (len(DATES), len(SYMBOLS))),
            axis=0) * 10)
        reports = analyze(random_frame(14), prices, "f", [1, 5, 20])
        assert [r.horizon for r in reports] == [1, 5, 20]
        assert all(r.factor == "f" for r in reports)

    def test_compare_table_shape(self):
        f = random_frame(15)
        reports = [compute_ic(f, f * 2, "a", 1), compute_ic(f, -f, "b", 5)]
        df = compare_table(reports)
        assert len(df) == 2
        assert "IC均值" in df.columns
        assert df["结论"].str.contains("反向").any()


class TestNeweyWest:
    """重叠样本修正。

    日频截面 + 60 日未来收益 = 相邻观测重叠 59 天，IC 序列强自相关。
    不修正的话 t 值被当成 671 个独立样本算，会放大数倍 ——
    实测 EP/60日 未修正 t=20.1，修正后掉到个位数。
    """

    def _autocorrelated(self, n=400, rho=0.95, mean=0.05, seed=42):
        """构造强自相关的 IC 序列：AR(1)"""
        rng = np.random.default_rng(seed)
        x = np.empty(n)
        x[0] = mean
        for i in range(1, n):
            x[i] = mean + rho * (x[i - 1] - mean) + rng.normal(0, 0.05)
        return x

    def test_shrinks_t_on_autocorrelated_series(self):
        from qmtquant.research.factor_ic import _newey_west_t
        x = self._autocorrelated()
        naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        nw = _newey_west_t(x, lags=59)
        assert abs(nw) < abs(naive), (
            f"自相关序列的 NW t 值({nw:.2f})应小于未修正({naive:.2f})")

    def test_matches_naive_when_independent(self):
        """独立样本下 lags=0，NW 应退化为普通 t 检验"""
        from qmtquant.research.factor_ic import _newey_west_t
        rng = np.random.default_rng(7)
        x = rng.normal(0.05, 0.1, 300)
        nw = _newey_west_t(x, lags=0)
        naive = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
        assert nw == pytest.approx(naive, rel=1e-6)

    def test_too_few_observations(self):
        from qmtquant.research.factor_ic import _newey_west_t
        assert np.isnan(_newey_west_t(np.array([1.0, 2.0]), lags=1))

    def test_report_keeps_naive_for_comparison(self):
        f = random_frame(21)
        rep = compute_ic(f, forward_returns(f.abs() + 1, 5), "x", 5)
        assert hasattr(rep, "ic_t_naive")

    def test_significance_uses_corrected_t(self):
        """判显著必须用修正后的 t，否则重叠样本会让噪声通过检验"""
        rep = ICReport(ic_mean=0.05, ic_std=0.1, ic_t=1.2, ic_t_naive=20.0)
        assert not rep.is_significant, "应以 NW t 值为准"

    def test_compare_table_exposes_both(self):
        f = random_frame(22)
        df = compare_table([compute_ic(f, f * 2, "a", 20)])
        assert "IC_t(NW)" in df.columns
        assert "IC_t未修正" in df.columns
