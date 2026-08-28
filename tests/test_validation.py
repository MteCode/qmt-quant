"""策略稳健性验证测试。"""
import pandas as pd
import pytest

from qmtquant.research.validation import (
    check_overfit,
    filter_by_drawdown,
    parameter_plateau,
    robust_params,
)


class TestDrawdownConstraint:
    """回撤硬约束必须在寻优之前生效，而不是选完再看。

    真问题：指数择时的网格按 Sharpe 挑出的「最优参数」回撤 -27.67%，
    超过 20% 上限。它在约束下根本不可用，却被当成最优点去做平原检验，
    整个结论都建立在一个不能上实盘的参数上。
    """

    def _grid(self):
        return pd.DataFrame([
            {"w": 10, "最大回撤": -0.35, "Sharpe": 0.90},   # 最高分但违规
            {"w": 20, "最大回撤": -0.30, "Sharpe": 0.80},   # 违规
            {"w": 40, "最大回撤": -0.18, "Sharpe": 0.30},   # 合规
            {"w": 60, "最大回撤": -0.15, "Sharpe": 0.25},   # 合规
            {"w": 80, "最大回撤": -0.12, "Sharpe": 0.20},   # 合规
        ])

    def test_filter_drops_violations(self):
        kept = filter_by_drawdown(self._grid(), 0.20)
        assert list(kept["w"]) == [40, 60, 80]

    def test_boundary_kept(self):
        df = pd.DataFrame([{"w": 1, "最大回撤": -0.20, "Sharpe": 0.5}])
        assert len(filter_by_drawdown(df, 0.20)) == 1, "恰好 20% 应保留"

    def test_best_params_exclude_violations(self):
        """核心：最优点必须从合规组合里选"""
        rep = parameter_plateau(self._grid(), ["w"], "Sharpe",
                                max_drawdown_limit=0.20)
        assert rep.best_params["w"] == 40, (
            f"应选合规组中 Sharpe 最高的 w=40，实际 {rep.best_params}")

    def test_reports_eligible_count(self):
        rep = parameter_plateau(self._grid(), ["w"], "Sharpe",
                                max_drawdown_limit=0.20)
        assert rep.eligible == 3 and rep.total == 5
        assert "3/5" in rep.summary()

    def test_all_violating_is_explicit_conclusion(self):
        """全部违规是一个明确结论，不是「没找到」"""
        df = pd.DataFrame([
            {"w": 10, "最大回撤": -0.40, "Sharpe": 0.9},
            {"w": 20, "最大回撤": -0.35, "Sharpe": 0.8},
        ])
        rep = parameter_plateau(df, ["w"], "Sharpe", max_drawdown_limit=0.20)
        assert not rep.best_params
        assert rep.eligible == 0
        assert "没有任何参数组合满足回撤上限" in rep.summary()

    def test_limit_none_disables(self):
        rep = parameter_plateau(self._grid(), ["w"], "Sharpe",
                                max_drawdown_limit=None)
        assert rep.best_params["w"] == 10, "不设约束时应选原始最高分"


class TestOverfitBudget:
    """自由度检验：参数个数与可用信息量是否匹配。

    参数平原检验回答「这组参数稳不稳」，但发现不了「整个网格都是噪声」——
    噪声同样可以形成平原。自由度检验回答更前置的问题：
    **这么多参数，数据够不够拟合。**
    """

    def test_trading_days_are_not_independent(self):
        """2588 个交易日不等于 2588 个观测 —— 相邻交易日高度相关"""
        r = check_overfit(n_params=3, trading_days=2588, n_combos=60)
        assert r.independent_regimes < 100, (
            f"折算后应远小于交易日数，实际 {r.independent_regimes}")

    def test_budget_scales_with_data(self):
        short = check_overfit(3, trading_days=600, n_combos=10)
        long = check_overfit(3, trading_days=2400, n_combos=10)
        assert long.budget == pytest.approx(short.budget * 4)

    def test_too_many_params_fails(self):
        """14 个参数 / 10 年数据 —— 数据不够"""
        assert not check_overfit(14, trading_days=2588, n_combos=100).passed

    def test_few_params_passes(self):
        assert check_overfit(2, trading_days=2588, n_combos=20).passed

    def test_selection_bias_grows_with_grid(self):
        """扫的组合越多，最优点的虚高越严重"""
        small = check_overfit(2, 2588, n_combos=10).selection_bias
        large = check_overfit(2, 2588, n_combos=1000).selection_bias
        assert large > small > 0

    def test_single_combo_no_bias(self):
        assert check_overfit(2, 2588, n_combos=1).selection_bias == 0.0

    def test_summary_explains_failure(self):
        text = check_overfit(14, 2588, 100).summary()
        assert "自由度不足" in text
        assert "记忆而非规律" in text


class TestRobustParams:
    """选平原中心而非峰值。

    在 N 组参数里挑最高分，即使全是噪声，最好那组看起来也会很好。
    改用邻域中位数打分 —— 一组参数只有在它自己和周围都还行时才得高分。
    """

    def _spiky(self):
        """w=30 是孤峰（自己高、邻居塌），w=70~90 是平原"""
        return pd.DataFrame([
            {"w": 10, "最大回撤": -0.10, "Sharpe": 0.05},
            {"w": 20, "最大回撤": -0.10, "Sharpe": 0.02},
            {"w": 30, "最大回撤": -0.10, "Sharpe": 0.90},   # 孤峰
            {"w": 40, "最大回撤": -0.10, "Sharpe": 0.01},
            {"w": 50, "最大回撤": -0.10, "Sharpe": 0.03},
            {"w": 60, "最大回撤": -0.10, "Sharpe": 0.40},
            {"w": 70, "最大回撤": -0.10, "Sharpe": 0.45},   # 平原
            {"w": 80, "最大回撤": -0.10, "Sharpe": 0.44},
            {"w": 90, "最大回撤": -0.10, "Sharpe": 0.43},
        ])

    def test_avoids_isolated_peak(self):
        picked = robust_params(self._spiky(), ["w"], "Sharpe")
        assert picked["w"] != 30, "不应选中孤峰"

    def test_picks_plateau(self):
        picked = robust_params(self._spiky(), ["w"], "Sharpe")
        assert picked["w"] in (70, 80), f"应选平原中心，实际 {picked}"

    def test_peak_selection_differs(self):
        """对照：按峰值选会选中孤峰"""
        rep = parameter_plateau(self._spiky(), ["w"], "Sharpe")
        assert rep.best_params["w"] == 30

    def test_respects_drawdown_limit(self):
        df = self._spiky().copy()
        df.loc[df["w"].isin([70, 80, 90]), "最大回撤"] = -0.40
        picked = robust_params(df, ["w"], "Sharpe", max_drawdown_limit=0.20)
        assert picked["w"] not in (70, 80, 90)

    def test_empty_when_all_violate(self):
        df = self._spiky().copy()
        df["最大回撤"] = -0.50
        assert robust_params(df, ["w"], "Sharpe", 0.20) == {}

    def test_preserves_int_dtype(self):
        picked = robust_params(self._spiky(), ["w"], "Sharpe")
        assert isinstance(picked["w"], int)
