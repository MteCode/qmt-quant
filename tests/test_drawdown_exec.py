"""回撤减仓数量计算 —— 回测与实盘共用的唯一实现。

## 为什么抽出来

同一件事「把持仓削减到 keep_ratio」原先在回测（`backtest_engine._enforce_drawdown`）
和实盘（`risk_monitor.sell_to_target`）里各写一遍，且已经跑偏：
回测会发出 451 股这种零股**部分**委托（实盘必被券商拒单），实盘则 floor 到 400。

A 股规则：部分卖出须为 100 的整数倍；**清空整个可卖持仓**时零股可一次性卖出。
取整只作用于「部分卖出」。整手约束作用在真实股数上，回测需按 factor 换算
（本文件里的 factor 用例）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from qmtquant.risk.drawdown_exec import plan_reduction, should_act


class TestPlanReduction:
    def test_partial_reduction_rounds_to_lots(self):
        # 卖 600，整手 → 600
        assert plan_reduction(1000, 1000, 0.4) == pytest.approx(600)

    def test_lot_rounding_uses_real_shares_with_factor(self):
        # factor=6：后复权 1000 股 = 真实 6000 股；卖 60% 真实 = 3600 股（整手）
        # → 回后复权口径 3600/6 = 600
        assert plan_reduction(1000, 1000, 0.4, factor=6.0) == pytest.approx(600)

    def test_factor_flooring_happens_in_real_shares(self):
        # factor=5.4：真实 3240 股 = 32.4 手 → floor 到 3200 真实股
        # → 回后复权口径 3200/5.4。证明取整作用在**真实**股数上，
        # 若错误地在后复权口径取整则会得到 600。
        assert plan_reduction(1000, 1000, 0.4,
                              factor=5.4) == pytest.approx(3200 / 5.4)

    def test_factor_zero_falls_back_to_one(self):
        assert plan_reduction(1000, 1000, 0.4, factor=0.0) == pytest.approx(600)

    def test_full_liquidation_keeps_odd_lots(self):
        # 清仓：零股可一次性卖出，不取整
        assert plan_reduction(199, 199, 0.0) == pytest.approx(199)
        assert plan_reduction(1537, 1537, 0.0) == pytest.approx(1537)

    def test_t_plus_one_available_zero(self):
        assert plan_reduction(1000, 0, 0.4) == 0.0

    def test_available_between_target_and_excess_uses_sellable(self):
        # available=500 恰是整手，取整后仍为 500
        assert plan_reduction(2000, 500, 0.1) == pytest.approx(500)

    def test_capped_partial_with_odd_available_is_floored(self):
        """回归钉：available=451 截断的部分减仓必须 floor 到 400。

        旧回测在此会发出 451 股（零股部分委托，实盘被拒），实盘则 floor 到 400。
        统一到券商合法口径后为 400。
        """
        assert plan_reduction(1000, 451, 0.4) == pytest.approx(400)

    def test_int_float_target_parity(self):
        """回归钉：旧实盘用 int(volume*ratio) 会多卖一整手。

        volume=199, keep=0.5：旧实盘 target=int(99.5)=99、raw_sell=100 → 卖 100；
        旧回测用浮点 excess=99.5 → floor(99.5)=0。统一后 0 股 —— 不足一手的
        减仓宁可不发单，也不发非法委托。
        """
        assert plan_reduction(199, 199, 0.5) == 0.0

    def test_tiny_reduction_returns_zero(self):
        assert plan_reduction(50, 50, 0.9) == 0.0

    def test_keep_ratio_one_returns_zero(self):
        assert plan_reduction(1000, 1000, 1.0) == 0.0

    def test_lot_size_override(self):
        # 600 本就是 200 的整数倍 → 600；不足 1000 的一手 → 0
        assert plan_reduction(1000, 1000, 0.4, lot_size=200) == pytest.approx(600)
        assert plan_reduction(1000, 1000, 0.4, lot_size=1000) == 0.0

    def test_non_positive_volume(self):
        assert plan_reduction(0, 0, 0.4) == 0.0


class TestShouldAct:
    def test_rising_edge_triggers(self):
        assert should_act(2, 1) is True

    def test_same_level_no_trigger(self):
        assert should_act(1, 1) is False

    def test_falling_edge_no_trigger(self):
        assert should_act(0, 1) is False


class TestCallSitesRouteThroughShared:
    """两个调用点必须真的走 plan_reduction，不能各自复算公式。

    复算公式的测试只能证明我会算术，证明不了代码做了这件事 —— 这正是
    两份实现当初漂移开的原因。
    """

    def _bar(self, close=10.0):
        from qmtquant.core.constants import Exchange, Interval
        from qmtquant.core.objects import BarData
        return BarData(
            symbol="600000", exchange=Exchange.SSE,
            datetime=pd.Timestamp("2026-09-04").to_pydatetime(),
            interval=Interval.DAILY, open_price=close, high_price=close,
            low_price=close, close_price=close, volume=1e6, turnover=1e7)

    def test_backtest_uses_plan_reduction(self, monkeypatch):
        from qmtquant.risk.drawdown import DrawdownController, DrawdownLevel
        import qmtquant.engine.backtest_engine as be

        calls = []

        def fake(volume, available, keep_ratio, lot_size=100, factor=1.0):
            calls.append((volume, available, keep_ratio))
            return 300.0

        monkeypatch.setattr(be, "plan_reduction", fake)

        e = be.BacktestEngine(initial_capital=1_000_000)
        bar = self._bar()
        e.load_data([bar])
        e.drawdown = DrawdownController()
        e.drawdown.config.reduce_keep_ratio = 0.3
        e.drawdown.state.level = DrawdownLevel.REDUCE
        e.positions = {"600000.SSE": {
            "volume": 1000.0, "available": 1000.0, "price": 10.0}}
        e.pending_orders = []
        e._last_enforced_level = DrawdownLevel.NORMAL

        e._enforce_drawdown({"600000.SSE": bar})

        assert calls, "回测减仓未经过共享函数 plan_reduction"
        orders = [o for o in e.pending_orders if o.reference == e.RISK_REFERENCE]
        assert orders, "应发出减仓委托"
        assert orders[0].volume == pytest.approx(300.0)

    def test_backtest_no_order_when_plan_returns_zero(self, monkeypatch):
        from qmtquant.risk.drawdown import DrawdownController, DrawdownLevel
        import qmtquant.engine.backtest_engine as be

        monkeypatch.setattr(be, "plan_reduction",
                            lambda *a, **k: 0.0)

        e = be.BacktestEngine(initial_capital=1_000_000)
        bar = self._bar()
        e.load_data([bar])
        e.drawdown = DrawdownController()
        e.drawdown.config.reduce_keep_ratio = 0.3
        e.drawdown.state.level = DrawdownLevel.REDUCE
        e.positions = {"600000.SSE": {
            "volume": 50.0, "available": 50.0, "price": 10.0}}
        e.pending_orders = []
        e._last_enforced_level = DrawdownLevel.NORMAL

        e._enforce_drawdown({"600000.SSE": bar})

        orders = [o for o in e.pending_orders if o.reference == e.RISK_REFERENCE]
        assert not orders
