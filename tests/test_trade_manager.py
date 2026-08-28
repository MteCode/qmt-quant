"""逐仓交易管理测试：止损、止盈、移动止损、基于风险的仓位。

这里验证的是「固定买卖点」真正落地所需的不变量：
- 止损价永不下调（下调等于没有止损）
- 每笔亏损上限相同，与标的波动无关
- 同一根 Bar 同时触及止损与止盈时取更坏的那个
"""
import pytest

from qmtquant.strategy.trade_manager import (
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIME,
    EXIT_TRAILING,
    ManagedPosition,
    TradeManager,
)

SYM = "600519.SSE"


def mgr(**kw) -> TradeManager:
    return TradeManager(**kw)


class TestManagedPosition:
    def test_stop_must_be_below_entry(self):
        """只做多，止损在入场价之上是逻辑错误 —— 会立刻触发"""
        with pytest.raises(ValueError, match="止损价必须低于入场价"):
            ManagedPosition(SYM, entry_price=10.0, volume=100, stop_price=11.0)

    def test_stop_equal_entry_rejected(self):
        with pytest.raises(ValueError, match="止损价必须低于入场价"):
            ManagedPosition(SYM, entry_price=10.0, volume=100, stop_price=10.0)

    def test_target_must_be_above_entry(self):
        with pytest.raises(ValueError, match="止盈价必须高于入场价"):
            ManagedPosition(SYM, entry_price=10.0, volume=100,
                            stop_price=9.0, target_price=9.5)

    def test_negative_entry_rejected(self):
        with pytest.raises(ValueError, match="入场价必须为正"):
            ManagedPosition(SYM, entry_price=-1.0, volume=100, stop_price=-2.0)

    def test_risk_per_share(self):
        p = ManagedPosition(SYM, entry_price=10.0, volume=100, stop_price=9.0)
        assert p.risk_per_share == pytest.approx(1.0)
        assert p.max_loss == pytest.approx(100.0)

    def test_r_multiple(self):
        """R 倍数：盈亏是初始风险的几倍，跨标的可比"""
        p = ManagedPosition(SYM, entry_price=10.0, volume=100, stop_price=9.0)
        assert p.r_multiple(12.0) == pytest.approx(2.0)
        assert p.r_multiple(9.0) == pytest.approx(-1.0)
        assert p.r_multiple(10.0) == pytest.approx(0.0)

    def test_initial_stop_is_current_stop(self):
        p = ManagedPosition(SYM, entry_price=10.0, volume=100, stop_price=9.0)
        assert p.current_stop == pytest.approx(9.0)
        assert p.highest == pytest.approx(10.0)


class TestPositionSize:
    def test_loss_capped_at_risk_budget(self):
        """核心不变量：每笔交易的亏损上限 = 总资产 × risk_per_trade"""
        m = mgr(risk_per_trade=0.01)
        vol = m.position_size(1_000_000, entry_price=10.0, stop_price=9.0)
        assert vol * (10.0 - 9.0) == pytest.approx(10_000)

    def test_wider_stop_smaller_position(self):
        """止损距离决定仓位 —— 波动大的票自动买得少"""
        m = mgr(risk_per_trade=0.01, max_position_ratio=1.0)
        tight = m.position_size(1_000_000, 10.0, 9.5)    # 距离 0.5
        wide = m.position_size(1_000_000, 10.0, 8.0)     # 距离 2.0
        assert tight == pytest.approx(wide * 4)

    def test_same_loss_across_volatility(self):
        """两只波动完全不同的票，亏损上限必须一样"""
        m = mgr(risk_per_trade=0.01, max_position_ratio=1.0)
        a = m.position_size(1_000_000, 100.0, 95.0)
        b = m.position_size(1_000_000, 10.0, 9.9)
        assert a * 5.0 == pytest.approx(b * 0.1)

    def test_capped_by_max_position_ratio(self):
        """低波动标的按风险算出的仓位会超过总资产，必须有绝对上限"""
        m = mgr(risk_per_trade=0.01, max_position_ratio=0.2)
        vol = m.position_size(1_000_000, 10.0, 9.999)
        assert vol == pytest.approx(1_000_000 * 0.2 / 10.0)

    def test_invalid_stop_returns_zero(self):
        m = mgr()
        assert m.position_size(1_000_000, 10.0, 10.0) == 0.0
        assert m.position_size(1_000_000, 10.0, 11.0) == 0.0
        assert m.position_size(1_000_000, 0.0, -1.0) == 0.0


class TestValidation:
    def test_risk_per_trade_range(self):
        with pytest.raises(ValueError, match="risk_per_trade"):
            mgr(risk_per_trade=0)
        with pytest.raises(ValueError, match="risk_per_trade"):
            mgr(risk_per_trade=0.9)

    def test_trailing_ratio_range(self):
        with pytest.raises(ValueError, match="trailing_ratio"):
            mgr(trailing_ratio=1.0)

    def test_max_position_ratio_range(self):
        with pytest.raises(ValueError, match="max_position_ratio"):
            mgr(max_position_ratio=1.5)


class TestStopLoss:
    def test_triggers_on_low_touching_stop(self):
        m = mgr()
        m.open(SYM, 10.0, 100, 9.0)
        assert m.check(SYM, high=10.5, low=8.9, close=9.2) == EXIT_STOP

    def test_exact_touch_triggers(self):
        """恰好触及即算触发 —— 实盘挂单也是这个行为"""
        m = mgr()
        m.open(SYM, 10.0, 100, 9.0)
        assert m.check(SYM, high=10.5, low=9.0, close=9.5) == EXIT_STOP

    def test_no_trigger_above_stop(self):
        m = mgr()
        m.open(SYM, 10.0, 100, 9.0)
        assert m.check(SYM, high=10.5, low=9.01, close=10.2) is None

    def test_unknown_symbol(self):
        assert mgr().check("999999.SSE", 10, 9, 9.5) is None


class TestTakeProfit:
    def test_triggers_on_high(self):
        m = mgr()
        m.open(SYM, 10.0, 100, 9.0, target_price=12.0)
        assert m.check(SYM, high=12.0, low=11.0, close=11.8) == EXIT_TARGET

    def test_stop_wins_when_both_hit(self):
        """同一根 Bar 同时触及止损与止盈，日线无法知道谁先发生。
        取更坏的那个是唯一诚实的做法 —— 假设止盈先到会系统性高估收益"""
        m = mgr()
        m.open(SYM, 10.0, 100, 9.0, target_price=12.0)
        assert m.check(SYM, high=12.5, low=8.5, close=10.0) == EXIT_STOP


class TestTrailingStop:
    def test_disabled_by_default_ratio_zero(self):
        m = mgr(trailing_ratio=0.0)
        m.open(SYM, 10.0, 100, 9.0)
        m.check(SYM, high=20.0, low=15.0, close=18.0)
        assert m.get(SYM).current_stop == pytest.approx(9.0)

    def test_does_not_start_below_threshold(self):
        """刚建仓就移动止损会被正常波动扫出去"""
        m = mgr(trailing_ratio=0.10, trailing_start_r=1.0)
        m.open(SYM, 10.0, 100, 9.0)           # 1R = 1.0 元
        m.check(SYM, high=10.5, low=10.0, close=10.4)   # 仅 +0.5R
        assert m.get(SYM).current_stop == pytest.approx(9.0)

    def test_starts_after_threshold(self):
        m = mgr(trailing_ratio=0.10, trailing_start_r=1.0)
        m.open(SYM, 10.0, 100, 9.0)
        m.check(SYM, high=12.0, low=11.0, close=11.8)   # +2R
        assert m.get(SYM).current_stop == pytest.approx(12.0 * 0.9)

    def test_never_moves_down(self):
        """止损下调等于没有止损 —— 这是最关键的不变量"""
        m = mgr(trailing_ratio=0.10, trailing_start_r=0.0)
        m.open(SYM, 10.0, 100, 9.0)
        m.check(SYM, high=15.0, low=14.0, close=14.5)
        raised = m.get(SYM).current_stop
        assert raised == pytest.approx(13.5)

        m.check(SYM, high=14.0, low=13.6, close=13.8)   # 价格回落
        assert m.get(SYM).current_stop == pytest.approx(raised), "止损不得下调"

    def test_reports_trailing_reason(self):
        """止损被抬高后触发，归因应是「移动止损」而非「止损」"""
        m = mgr(trailing_ratio=0.10, trailing_start_r=0.0)
        m.open(SYM, 10.0, 100, 9.0)
        m.check(SYM, high=15.0, low=14.0, close=14.5)   # 止损抬到 13.5
        assert m.check(SYM, high=14.0, low=13.0, close=13.2) == EXIT_TRAILING

    def test_trailing_checked_before_update(self):
        """必须先判断本根是否止损，再更新移动止损。
        顺序反了会用「本根抬高后的止损」判断「本根是否止损」，等于用未来信息"""
        m = mgr(trailing_ratio=0.10, trailing_start_r=0.0)
        m.open(SYM, 10.0, 100, 9.0)
        # 本根冲高到 15 又砸回 8.5：应按旧止损 9.0 判为止损
        assert m.check(SYM, high=15.0, low=8.5, close=8.6) == EXIT_STOP


class TestTimeExit:
    def test_triggers_after_max_bars(self):
        m = mgr(max_holding_bars=10)
        m.open(SYM, 10.0, 100, 9.0, bar_index=100)
        assert m.check(SYM, 10.5, 9.5, 10.0, bar_index=109) is None
        assert m.check(SYM, 10.5, 9.5, 10.0, bar_index=110) == EXIT_TIME

    def test_disabled_when_zero(self):
        m = mgr(max_holding_bars=0)
        m.open(SYM, 10.0, 100, 9.0, bar_index=0)
        assert m.check(SYM, 10.5, 9.5, 10.0, bar_index=99999) is None


class TestBookkeeping:
    def test_close_records_reason(self):
        m = mgr()
        m.open(SYM, 10.0, 100, 9.0)
        m.close(SYM, EXIT_STOP)
        assert not m.has(SYM)
        assert m.exit_stats[EXIT_STOP] == 1

    def test_close_unknown_is_noop(self):
        m = mgr()
        m.close("999999.SSE", EXIT_STOP)
        assert not m.exit_stats

    def test_summary_with_no_trades(self):
        assert "无离场记录" in mgr().summary()

    def test_summary_lists_reasons(self):
        m = mgr()
        for _ in range(3):
            m.open(SYM, 10.0, 100, 9.0)
            m.close(SYM, EXIT_STOP)
        m.open(SYM, 10.0, 100, 9.0)
        m.close(SYM, EXIT_TARGET)
        text = m.summary()
        assert "止损" in text and "75.0%" in text


class TestRiskAnchoring:
    """R 倍数锚定初始风险，max_loss 反映当前风险 —— 两个锚点不同。"""

    def test_r_multiple_unaffected_by_trailing(self):
        """「赚了 2R」= 赚到当初愿意亏的两倍。锚点随止损移动就失去意义"""
        m = mgr(trailing_ratio=0.10, trailing_start_r=0.0)
        m.open(SYM, 10.0, 100, 9.0)          # 初始 1R = 1.0 元
        m.check(SYM, high=12.0, low=11.0, close=11.8)
        pos = m.get(SYM)
        assert pos.current_stop > pos.stop_price, "前提：止损已上移"
        assert pos.risk_per_share == pytest.approx(1.0), "初始风险不变"
        assert pos.r_multiple(12.0) == pytest.approx(2.0)

    def test_max_loss_shrinks_as_stop_rises(self):
        """止损上移后真实风险下降，用初始值会系统性高估在管风险"""
        m = mgr(trailing_ratio=0.10, trailing_start_r=0.0)
        m.open(SYM, 10.0, 100, 9.0)
        assert m.get(SYM).max_loss == pytest.approx(100.0)

        m.check(SYM, high=12.0, low=11.0, close=11.8)   # 止损抬到 10.8
        assert m.get(SYM).max_loss < 0, "止损已在成本之上，这笔已锁定盈利"

    def test_current_risk_negative_when_stop_above_entry(self):
        m = mgr(trailing_ratio=0.05, trailing_start_r=0.0)
        m.open(SYM, 10.0, 100, 9.0)
        m.check(SYM, high=15.0, low=14.0, close=14.5)   # 止损抬到 14.25
        assert m.get(SYM).current_risk_per_share == pytest.approx(-4.25)
