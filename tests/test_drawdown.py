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


class TestFreezeTimeout:
    """最长冻结期：避免单次深回撤把策略永久锁死。

    关键点是重置必须**同时重置峰值** —— 只降档不动峰值的话，
    下一次 update() 算出的回撤依然超标，会立刻重新触发。
    """

    def test_disabled_by_default(self):
        ctrl = make()
        feed(ctrl, [100.0] + [80.0] * 200)
        assert ctrl.level is DrawdownLevel.FLAT
        assert ctrl.state.peak_resets == 0

    def test_resets_after_freeze_limit(self):
        ctrl = make(max_freeze_observations=10)
        feed(ctrl, [100.0] + [80.0] * 12)
        assert ctrl.level is DrawdownLevel.NORMAL
        assert ctrl.state.peak_resets == 1

    def test_peak_reset_to_current(self):
        """峰值必须重置为当前净值，否则回撤依旧超标会立刻重新触发"""
        ctrl = make(max_freeze_observations=5)
        feed(ctrl, [100.0] + [80.0] * 7)
        assert ctrl.state.peak == pytest.approx(80.0)
        assert ctrl.drawdown == pytest.approx(0.0)

    def test_does_not_retrigger_immediately_after_reset(self):
        """重置后继续横盘不应再次触发 —— 这正是只降档不重置峰值的失败模式"""
        ctrl = make(max_freeze_observations=5)
        feed(ctrl, [100.0] + [80.0] * 7)
        assert ctrl.level is DrawdownLevel.NORMAL
        feed(ctrl, [80.0] * 3)
        assert ctrl.level is DrawdownLevel.NORMAL

    def test_can_trigger_again_from_new_peak(self):
        """重置后若继续下跌，应基于新峰值重新触发"""
        ctrl = make(max_freeze_observations=5)
        feed(ctrl, [100.0] + [80.0] * 7)
        assert ctrl.level is DrawdownLevel.NORMAL
        feed(ctrl, [63.0])               # 相对新峰值 80 回撤 21%
        assert ctrl.level is DrawdownLevel.FLAT

    def test_counter_resets_on_level_change(self):
        """档位正常变化时冻结计数应清零，不能累积到无关的档位上"""
        ctrl = make(max_freeze_observations=100)
        feed(ctrl, [100.0, 89.0])
        assert ctrl.level is DrawdownLevel.CLOSE_ONLY
        feed(ctrl, [95.0])               # 恢复到 NORMAL
        assert ctrl.level is DrawdownLevel.NORMAL
        assert ctrl.state.observations_at_level == 0

    def test_repeated_resets_counted(self):
        """反复重置次数要能被观测到 —— 次数多说明策略本身有问题"""
        ctrl = make(max_freeze_observations=5)
        equity = [100.0]
        for _ in range(4):
            equity += [equity[-1] * 0.75] * 6
        feed(ctrl, equity)
        assert ctrl.state.peak_resets >= 3
        assert "峰值重置" in ctrl.summary()

    def test_negative_limit_rejected(self):
        with pytest.raises(ValueError, match="max_freeze_observations"):
            DrawdownController(DrawdownConfig(max_freeze_observations=-1))


class TestForcedReduction:
    """回撤档位必须**真的卖出**。

    真 bug：控制器有 CLOSE_ONLY/REDUCE/FLAT 三档，
    但引擎只调用了 allow_open() —— 后两档从未被执行。
    结果是回撤中「不开新仓但也不卖」，行情继续跌则回撤继续深，
    20% 的上限根本压不住。
    """

    def _crash_bars(self, n_up=30, n_down=60):
        import pandas as pd
        from qmtquant.core.constants import Exchange, Interval
        from qmtquant.core.objects import BarData
        prices = [10 * 1.02 ** i for i in range(n_up)]
        prices += [prices[-1] * 0.97 ** i for i in range(1, n_down)]
        dates = pd.bdate_range("2023-01-02", periods=len(prices))
        return [BarData(symbol="600519", exchange=Exchange.SSE,
                        datetime=d.to_pydatetime(), interval=Interval.DAILY,
                        open_price=p, high_price=p * 1.001,
                        low_price=p * 0.999, close_price=p,
                        volume=1_000_000, turnover=p * 1_000_000)
                for d, p in zip(dates, prices)]

    def _run(self, dd_cfg):
        from qmtquant.engine.backtest_engine import BacktestEngine
        from qmtquant.strategy.base import StrategyBase

        class BuyAndHold(StrategyBase):
            """买入后死不撒手 —— 只有风控能让它出场"""
            parameters: list[str] = []

            def on_bar(self, bar):
                if self.trading and self.get_pos(bar.vt_symbol) == 0:
                    self.buy(bar.vt_symbol, bar.close_price * 1.05,
                             self.get_cash() * 0.95 / bar.close_price)

        ctrl = DrawdownController(dd_cfg) if dd_cfg else None
        engine = BacktestEngine(initial_capital=1_000_000, drawdown=ctrl)
        engine.load_data(self._crash_bars())
        engine.add_strategy(BuyAndHold, ["600519.SSE"], {})
        stats = engine.run()
        return engine, stats

    def test_reduces_drawdown_versus_none(self):
        _, no_ctrl = self._run(None)
        engine, with_ctrl = self._run(
            DrawdownConfig(close_only_threshold=0.05, reduce_threshold=0.08,
                           flat_threshold=0.12, min_observations=5))
        assert with_ctrl.max_drawdown > no_ctrl.max_drawdown, (
            f"回撤控制应减小回撤：{with_ctrl.max_drawdown:.2%} "
            f"vs 无控制 {no_ctrl.max_drawdown:.2%}")

    def test_issues_risk_sell_orders(self):
        """必须真的发出卖单，而不只是拦住买单"""
        engine, _ = self._run(
            DrawdownConfig(close_only_threshold=0.05, reduce_threshold=0.08,
                           flat_threshold=0.12, min_observations=5))
        assert engine.risk_exit_orders > 0, "REDUCE/FLAT 档必须发出卖单"

    def test_risk_orders_tagged(self):
        """风控减仓要与策略主动交易区分开，否则归因会算错"""
        engine, _ = self._run(
            DrawdownConfig(close_only_threshold=0.05, reduce_threshold=0.08,
                           flat_threshold=0.12, min_observations=5))
        tagged = [t for t in engine.trades
                  if t.reference == engine.RISK_REFERENCE]
        assert tagged, "风控委托必须带标记"

    def test_no_orders_when_normal(self):
        engine = self._run(DrawdownConfig(
            close_only_threshold=0.90, reduce_threshold=0.95,
            flat_threshold=0.99, min_observations=5))[0]
        assert engine.risk_exit_orders == 0

    def test_flat_sells_everything(self):
        engine, _ = self._run(
            DrawdownConfig(close_only_threshold=0.03, reduce_threshold=0.05,
                           flat_threshold=0.07, min_observations=5))
        assert engine.get_pos("600519.SSE") == 0, "FLAT 档应清空持仓"


class TestNonMonotonicTightening:
    """收紧档位不是单调变好的 —— 这是选参数时最容易犯的错。

    实测（突破策略 · 沪深300 · 2016-2026）：

        一档/二档/清仓    最大回撤    总收益
           8/11/15       -22.67%    +95.38%
           6/ 9/12       -18.35%    +22.91%
           5/ 8/11       -32.42%    -22.12%   ← 收得更紧反而更差

    阈值低于策略常态波动时会被反复触发，在局部低点被迫卖出，
    峰值重置后再吃一轮完整回撤。
    """

    def test_default_config_targets_20_percent(self):
        """默认档位必须留出执行滞后的余量，不能直接照 20% 设"""
        from qmtquant.config import RiskConfig
        r = RiskConfig()
        assert r.drawdown_flat < 0.20, (
            "清仓线必须明显低于 20%：信号收盘产生、次日开盘成交，"
            "加上 T+1 当日买入不可卖，从触发到出清有滞后")

    def test_tiers_strictly_increasing(self):
        from qmtquant.config import RiskConfig
        r = RiskConfig()
        assert r.drawdown_close_only < r.drawdown_reduce < r.drawdown_flat

    def test_freeze_enabled_by_default(self):
        """不解冻会让策略停摆：实测成交从 3742 掉到 1184，
        回撤反而从 -18.35% 升到 -21.55%"""
        from qmtquant.config import RiskConfig
        assert RiskConfig().drawdown_max_freeze > 0


class TestNoRepeatedReduction:
    """同一档位内不得反复减仓。

    真 bug：原本每根 Bar 都重算「卖到 volume × ratio」，
    而 volume 已经是上次卖完之后的值 —— 指数衰减地反复卖。
    实测在低换手策略上制造出 6556 笔成交、胜率 2.25%
    （正常应为 610 笔、49.66%），全是被反复切割出来的微亏平仓。
    """

    def _run(self, dd_cfg):
        import pandas as pd
        from qmtquant.core.constants import Exchange, Interval
        from qmtquant.core.objects import BarData
        from qmtquant.engine.backtest_engine import BacktestEngine
        from qmtquant.strategy.base import StrategyBase

        # 长期缓跌：档位升上去后长时间停留，正是复现该 bug 的场景
        prices = [10 * 1.01 ** i for i in range(20)]
        prices += [prices[-1] * 0.995 ** i for i in range(1, 120)]
        dates = pd.bdate_range("2023-01-02", periods=len(prices))
        bars = [BarData(symbol="600519", exchange=Exchange.SSE,
                        datetime=d.to_pydatetime(), interval=Interval.DAILY,
                        open_price=p, high_price=p * 1.001,
                        low_price=p * 0.999, close_price=p,
                        volume=1_000_000, turnover=p * 1_000_000)
                for d, p in zip(dates, prices)]

        class BuyAndHold(StrategyBase):
            parameters: list[str] = []

            def on_bar(self, bar):
                if self.trading and self.get_pos(bar.vt_symbol) == 0:
                    self.buy(bar.vt_symbol, bar.close_price * 1.05,
                             self.get_cash() * 0.95 / bar.close_price)

        engine = BacktestEngine(initial_capital=1_000_000,
                                drawdown=DrawdownController(dd_cfg))
        engine.load_data(bars)
        engine.add_strategy(BuyAndHold, ["600519.SSE"], {})
        engine.run()
        return engine

    def test_reduce_fires_once_per_level(self):
        """REDUCE 档停留 100 根 Bar，减仓委托不该有几十笔"""
        engine = self._run(DrawdownConfig(
            close_only_threshold=0.03, reduce_threshold=0.05,
            reduce_keep_ratio=0.3, flat_threshold=0.60,
            min_observations=5, max_freeze_observations=0))
        assert engine.risk_exit_orders <= 3, (
            f"同一档位内反复减仓：发出 {engine.risk_exit_orders} 笔")

    def test_still_reduces_at_all(self):
        """修复不能把功能修没了"""
        engine = self._run(DrawdownConfig(
            close_only_threshold=0.03, reduce_threshold=0.05,
            reduce_keep_ratio=0.3, flat_threshold=0.60,
            min_observations=5, max_freeze_observations=0))
        assert engine.risk_exit_orders >= 1, "达到 REDUCE 档必须减仓"

    def test_escalation_triggers_again(self):
        """档位继续升高（REDUCE -> FLAT）时应再次减仓"""
        engine = self._run(DrawdownConfig(
            close_only_threshold=0.03, reduce_threshold=0.05,
            reduce_keep_ratio=0.5, flat_threshold=0.10,
            min_observations=5, max_freeze_observations=0))
        assert engine.risk_exit_orders >= 2, "升档应触发新一轮减仓"
        assert engine.get_pos("600519.SSE") == 0, "FLAT 档应清空"
