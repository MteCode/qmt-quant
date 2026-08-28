"""策略稳健性验证测试。"""
import pandas as pd

from qmtquant.research.validation import filter_by_drawdown, parameter_plateau


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
