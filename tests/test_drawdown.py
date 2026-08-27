"""回撤控制测试。

核心场景是「连续阴跌」：每天亏一点、一次都不触及日亏线，
但累计跌幅巨大 —— 这是实盘最常见的亏损形态，也是日亏线的盲区。
"""
import pytest

from qmtquant.config import RiskConfig
from qmtquant.core.constants import Direction, Exchange, RejectReason
from qmtquant.core.objects import AccountData, OrderRequest, PositionData
from qmtquant.risk.drawdown import (
    DrawdownConfig,
    DrawdownController,
    DrawdownLevel,
)
from qmtquant.risk.risk_manager import RiskManager


def make(**kw) -> DrawdownController:
    cfg = DrawdownConfig(min_observations=0, **kw)
    return DrawdownController(cfg)


def feed(ctrl: DrawdownController, values: list[float]) -> DrawdownLevel:
    for v in values:
        ctrl.update(v)
    return ctrl.level


class TestConfigValidation:
    def test_thresholds_must_be_ordered(self):
        with pytest.raises(ValueError, match="回撤阈值"):
            DrawdownController(DrawdownConfig(close_only_threshold=0.2,
                                              reduce_threshold=0.1,
                                              flat_threshold=0.3))

    def test_recovery_ratio_range(self):
        with pytest.raises(ValueError, match="recovery_ratio"):
            DrawdownController(DrawdownConfig(recovery_ratio=1.5))

    def test_keep_ratio_range(self):
        with pytest.raises(ValueError, match="reduce_keep_ratio"):
            DrawdownController(DrawdownConfig(reduce_keep_ratio=1.0))


class TestLevels:
    def test_starts_normal(self):
        ctrl = make()
        assert feed(ctrl, [100.0]) is DrawdownLevel.NORMAL

    def test_peak_tracks_high_water_mark(self):
        ctrl = make()
        feed(ctrl, [100, 120, 110])
        assert ctrl.state.peak == 120
        assert ctrl.drawdown == pytest.approx(1 - 110 / 120)

    def test_close_only_at_10pct(self):
        ctrl = make()
        assert feed(ctrl, [100, 89.9]) is DrawdownLevel.CLOSE_ONLY

    def test_reduce_at_15pct(self):
        ctrl = make()
        assert feed(ctrl, [100, 84.9]) is DrawdownLevel.REDUCE

    def test_flat_at_20pct(self):
        ctrl = make()
        assert feed(ctrl, [100, 79.9]) is DrawdownLevel.FLAT

    def test_no_trigger_just_below_threshold(self):
        ctrl = make()
        assert feed(ctrl, [100, 90.5]) is DrawdownLevel.NORMAL


class TestSlowBleed:
    """日亏线的盲区：连续小幅阴跌"""

    def test_daily_loss_line_would_miss_it(self):
        """每天跌 1%，连跌 20 天，累计 -18%，单日从未触及 3% 日亏线"""
        equity = [100.0]
        for _ in range(20):
            equity.append(equity[-1] * 0.99)

        # 单日跌幅始终 1%，远低于 3% 的日亏线
        daily = [1 - equity[i + 1] / equity[i] for i in range(len(equity) - 1)]
        assert max(daily) < 0.03

        # 但回撤控制能抓到：累计 1-0.99^20 = 18.2%，落在二档区间
        ctrl = make()
        level = feed(ctrl, equity)
        assert ctrl.drawdown == pytest.approx(1 - 0.99 ** 20, abs=1e-6)
        assert level is DrawdownLevel.REDUCE

    def test_escalates_through_levels(self):
        """缓慢下跌应依次经过三个档位，而非直接跳到最深档"""
        ctrl = make()
        equity = [100.0]
        for _ in range(30):
            equity.append(equity[-1] * 0.99)
        feed(ctrl, equity)

        seen = [new for _, _, _, new in ctrl.state.transitions]
        assert seen == [DrawdownLevel.CLOSE_ONLY, DrawdownLevel.REDUCE,
                        DrawdownLevel.FLAT]


class TestRecoveryHysteresis:
    def test_does_not_downgrade_immediately(self):
        """回撤刚好回到阈值下方一点点时不应降档，否则会反复横跳"""
        ctrl = make(close_only_threshold=0.10, recovery_ratio=0.7)
        feed(ctrl, [100, 89])            # 回撤 11% -> CLOSE_ONLY
        assert ctrl.level is DrawdownLevel.CLOSE_ONLY
        # 回撤 9.5%，低于阈值但未达 10%*0.7=7% 的恢复线
        feed(ctrl, [90.5])
        assert ctrl.level is DrawdownLevel.CLOSE_ONLY

    def test_downgrades_after_sufficient_recovery(self):
        ctrl = make(close_only_threshold=0.10, recovery_ratio=0.7)
        feed(ctrl, [100, 89])
        feed(ctrl, [94])                 # 回撤 6% < 7%
        assert ctrl.level is DrawdownLevel.NORMAL

    def test_no_flapping_around_threshold(self):
        """在阈值附近反复抖动，档位切换次数必须有限"""
        ctrl = make(close_only_threshold=0.10, recovery_ratio=0.7)
        seq = [100.0]
        for _ in range(10):
            seq += [89.5, 90.5]          # 回撤在 9.5% ~ 10.5% 之间来回
        feed(ctrl, seq)
        assert len(ctrl.state.transitions) <= 2

    def test_upgrade_is_immediate(self):
        """升档不打折 —— 风险来临时不能犹豫"""
        ctrl = make()
        feed(ctrl, [100, 89])
        assert ctrl.level is DrawdownLevel.CLOSE_ONLY
        feed(ctrl, [79])
        assert ctrl.level is DrawdownLevel.FLAT


class TestMinObservations:
    def test_ignores_early_noise(self):
        """刚启动时峰值就是当前值，任何下跌都算回撤，必须先攒够观测点"""
        ctrl = DrawdownController(DrawdownConfig(min_observations=20))
        feed(ctrl, [100, 70])
        assert ctrl.level is DrawdownLevel.NORMAL

    def test_active_after_enough_observations(self):
        ctrl = DrawdownController(DrawdownConfig(min_observations=5))
        feed(ctrl, [100] * 5 + [70])
        assert ctrl.level is DrawdownLevel.FLAT


class TestActions:
    def test_allow_open_only_when_normal(self):
        ctrl = make()
        assert ctrl.allow_open()
        feed(ctrl, [100, 89])
        assert not ctrl.allow_open()

    def test_target_ratio_by_level(self):
        ctrl = make(reduce_keep_ratio=0.4)
        assert ctrl.target_position_ratio() == 1.0
        feed(ctrl, [100, 84])
        assert ctrl.level is DrawdownLevel.REDUCE
        assert ctrl.target_position_ratio() == 0.4
        feed(ctrl, [79])
        assert ctrl.target_position_ratio() == 0.0

    def test_disabled_never_triggers(self):
        ctrl = DrawdownController(DrawdownConfig(enabled=False,
                                                 min_observations=0))
        assert feed(ctrl, [100, 50]) is DrawdownLevel.NORMAL

    def test_ignores_non_positive_equity(self):
        ctrl = make()
        feed(ctrl, [100])
        ctrl.update(0)
        assert ctrl.state.peak == 100

    def test_summary_renders(self):
        ctrl = make()
        feed(ctrl, [100, 85])
        s = ctrl.summary()
        assert "当前回撤" in s
        assert "强制减仓" in s


class TestRiskManagerIntegration:
    def _rm(self, **kw) -> RiskManager:
        # 关掉日亏线：本组测试要隔离出回撤这一项。
        # 净值一次性跳变会先触发日亏线，测到的就不是回撤逻辑了
        kw.setdefault("daily_loss_limit_ratio", 1.0)
        cfg = RiskConfig(drawdown_min_observations=0, max_order_value=10_000_000,
                         **kw)
        return RiskManager(cfg)

    def _buy(self, price=10.0, volume=1000):
        return OrderRequest(symbol="000001", exchange=Exchange.SZSE,
                            direction=Direction.LONG, price=price, volume=volume)

    def _sell(self, volume=1000):
        return OrderRequest(symbol="000001", exchange=Exchange.SZSE,
                            direction=Direction.SHORT, price=10.0, volume=volume)

    def test_buy_blocked_after_drawdown(self):
        rm = self._rm()
        rm.update_account(AccountData(balance=1_000_000, available=1_000_000))
        assert rm.check(self._buy())[0]

        # 回撤 12% -> 只平不开
        rm.update_account(AccountData(balance=880_000, available=880_000))
        ok, reason = rm.check(self._buy())
        assert not ok
        assert reason is RejectReason.DRAWDOWN_LIMIT

    def test_sell_still_allowed(self):
        """回撤触发后仍须允许卖出，否则无法减仓自救"""
        rm = self._rm()
        rm.update_account(AccountData(balance=1_000_000, available=1_000_000))
        rm.update_account(AccountData(balance=750_000, available=750_000))
        rm.update_position(PositionData(symbol="000001", exchange=Exchange.SZSE,
                                        volume=1000, frozen=0, yd_volume=1000))
        assert rm.check(self._sell())[0]

    def test_stats_expose_drawdown(self):
        rm = self._rm()
        rm.update_account(AccountData(balance=1_000_000, available=1_000_000))
        rm.update_account(AccountData(balance=800_000, available=800_000))
        st = rm.stats()
        assert st["drawdown"] == pytest.approx(0.2)
        assert st["drawdown_level"] == "全平停止"
        assert st["target_position_ratio"] == 0.0

    def test_can_be_disabled(self):
        rm = self._rm(drawdown_enabled=False)
        rm.update_account(AccountData(balance=1_000_000, available=1_000_000))
        rm.update_account(AccountData(balance=500_000, available=500_000))
        assert rm.check(self._buy())[0]


class TestFloatBoundary:
    """浮点边界：回撤 20% 常算成 0.19999999999999996，
    直接 >= 比较会漏掉。风控在边界上必须保守触发。"""

    @pytest.mark.parametrize("peak,trough,expected", [
        (1_000_000, 800_000, DrawdownLevel.FLAT),      # 恰好 20%
        (1_000_000, 850_000, DrawdownLevel.REDUCE),    # 恰好 15%
        (1_000_000, 900_000, DrawdownLevel.CLOSE_ONLY),  # 恰好 10%
    ])
    def test_exact_threshold_triggers(self, peak, trough, expected):
        ctrl = make()
        feed(ctrl, [float(peak), float(trough)])
        assert ctrl.level is expected

    def test_just_below_does_not_trigger(self):
        ctrl = make()
        feed(ctrl, [1_000_000.0, 900_001.0])   # 回撤 9.9999%
        assert ctrl.level is DrawdownLevel.NORMAL
