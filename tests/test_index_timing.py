"""指数择时策略测试。

重点验证三道防打脸机制：缓冲带、确认天数、最短持有期。
均线择时最大的敌人不是大跌，是横盘震荡下的反复穿越 ——
实测沪深300 上 MA20 择时亏 29%，比买入持有还差，正是死于此。
"""
import pandas as pd
import pytest

from qmtquant.core.constants import Direction, Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.backtest_engine import BacktestEngine
from qmtquant.strategy.index_timing import (
    SIGNAL_FLAT,
    SIGNAL_LONG,
    IndexTimingStrategy,
)

SYMBOL = "510300.SSE"


def bars_of(closes: list[float], symbol: str = "510300",
            exchange=Exchange.SSE) -> list[BarData]:
    dates = pd.bdate_range("2023-01-02", periods=len(closes))
    return [BarData(symbol=symbol, exchange=exchange,
                    datetime=d.to_pydatetime(), interval=Interval.DAILY,
                    open_price=c, high_price=c, low_price=c, close_price=c,
                    volume=1_000_000, turnover=c * 1_000_000)
            for d, c in zip(dates, closes)]


def run(closes, setting=None, capital=1_000_000):
    engine = BacktestEngine(initial_capital=capital)
    engine.load_data(bars_of(closes))
    engine.add_strategy(IndexTimingStrategy, [SYMBOL], setting or {})
    engine.run()
    return engine


def make(setting=None) -> IndexTimingStrategy:
    return IndexTimingStrategy(BacktestEngine(), "T", [SYMBOL], setting or {})


BASE = {"ma_window": 10, "band": 0.0, "confirm_days": 1,
        "min_holding_days": 0}


class TestValidation:
    def test_single_symbol_only(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="只交易单一标的"):
            engine.add_strategy(IndexTimingStrategy,
                                ["510300.SSE", "159919.SZSE"], {})

    def test_ma_window_minimum(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="ma_window"):
            engine.add_strategy(IndexTimingStrategy, [SYMBOL], {"ma_window": 3})

    def test_band_range(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="band"):
            engine.add_strategy(IndexTimingStrategy, [SYMBOL], {"band": 0.8})

    def test_confirm_days_minimum(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="confirm_days"):
            engine.add_strategy(IndexTimingStrategy, [SYMBOL],
                                {"confirm_days": 0})

    def test_float_params_coerced(self):
        """参数寻优会从 DataFrame 传入 float，用作 deque(maxlen=) 会报错"""
        s = make({"ma_window": 60.0, "confirm_days": 2.0,
                  "min_holding_days": 5.0})
        assert isinstance(s.ma_window, int)
        assert isinstance(s.confirm_days, int)
        assert isinstance(s.min_holding_days, int)


class TestMovingAverage:
    def test_none_until_enough_bars(self):
        s = make(BASE)
        for c in [10.0] * 9:
            s.closes.append(c)
        assert s.compute_ma() is None
        s.closes.append(10.0)
        assert s.compute_ma() == pytest.approx(10.0)

    def test_matches_manual(self):
        s = make(BASE)
        vals = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        for c in vals:
            s.closes.append(float(c))
        assert s.compute_ma() == pytest.approx(sum(vals) / len(vals))


class TestBand:
    """缓冲带：价格贴着均线时既不算站上也不算跌破"""

    def test_inside_band_resets_both_streaks(self):
        s = make(dict(BASE, band=0.05))
        s._long_streak = 3
        s._flat_streak = 0
        s._update_streaks(price=10.02, ma=10.0)   # 偏离仅 0.2% < 5%
        assert s._long_streak == 0
        assert s._flat_streak == 0

    def test_above_band_counts_long(self):
        s = make(dict(BASE, band=0.05))
        s._update_streaks(price=10.6, ma=10.0)    # 偏离 6% > 5%
        assert s._long_streak == 1

    def test_below_band_counts_flat(self):
        s = make(dict(BASE, band=0.05))
        s._update_streaks(price=9.4, ma=10.0)
        assert s._flat_streak == 1

    def test_zero_band_triggers_immediately(self):
        s = make(dict(BASE, band=0.0))
        s._update_streaks(price=10.01, ma=10.0)
        assert s._long_streak == 1

    def test_band_reduces_switches_in_choppy_market(self):
        """横盘震荡：缓冲带应显著减少交易次数"""
        # 在 10 上下 ±1.5% 来回震荡
        chop = []
        for i in range(120):
            chop.append(10.0 * (1 + (0.015 if i % 2 == 0 else -0.015)))
        prices = [10.0] * 15 + chop

        no_band = run(prices, dict(BASE, band=0.0))
        with_band = run(prices, dict(BASE, band=0.03))
        assert (with_band.strategy.switch_count
                <= no_band.strategy.switch_count), "缓冲带应减少切换"


class TestConfirmDays:
    def test_single_day_spike_ignored(self):
        """单日假突破不应触发 —— 需连续 N 日确认"""
        s = make(dict(BASE, confirm_days=3))
        s.signal = SIGNAL_FLAT
        s._update_streaks(11.0, 10.0)
        assert s._resolve_signal() == SIGNAL_FLAT

    def test_triggers_after_enough_days(self):
        s = make(dict(BASE, confirm_days=3))
        s.signal = SIGNAL_FLAT
        for _ in range(3):
            s._update_streaks(11.0, 10.0)
        assert s._resolve_signal() == SIGNAL_LONG

    def test_streak_broken_by_reversal(self):
        """中途反向会清零计数，需重新累积"""
        s = make(dict(BASE, confirm_days=3))
        s.signal = SIGNAL_FLAT
        s._update_streaks(11.0, 10.0)
        s._update_streaks(11.0, 10.0)
        s._update_streaks(9.0, 10.0)        # 反向，清零
        s._update_streaks(11.0, 10.0)
        assert s._resolve_signal() == SIGNAL_FLAT

    def test_holds_current_signal_when_ambiguous(self):
        s = make(dict(BASE, confirm_days=3, band=0.05))
        s.signal = SIGNAL_LONG
        s._update_streaks(10.01, 10.0)      # 落在缓冲带内
        assert s._resolve_signal() == SIGNAL_LONG


class TestMinHolding:
    def test_blocks_immediate_reversal(self):
        """刚建仓就反向的信号必须忽略"""
        s = make(dict(BASE, min_holding_days=10))
        s.trading = True
        s._bar_count = 100
        s._last_action_bar = 95        # 距上次动作仅 5 日
        s.signal = SIGNAL_LONG

        bar = bars_of([10.0])[0]
        s._switch_to(SIGNAL_FLAT, bar)
        assert s.signal == SIGNAL_LONG, "最短持有期内不应切换"

    def test_allows_after_min_holding(self):
        s = make(dict(BASE, min_holding_days=5))
        s.trading = True
        s._bar_count = 100
        s._last_action_bar = 90
        s.signal = SIGNAL_LONG
        s.pos[SYMBOL] = 0              # 无持仓，切 FLAT 只更新信号

        s._switch_to(SIGNAL_FLAT, bars_of([10.0])[0])
        assert s.signal == SIGNAL_FLAT


class TestTradingBehaviour:
    def test_buys_on_uptrend(self):
        prices = [10.0] * 12 + [10 * 1.01 ** i for i in range(1, 20)]
        engine = run(prices, BASE)
        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        assert buys, "上行趋势应买入"

    def test_sells_on_downtrend(self):
        up = [10 * 1.01 ** i for i in range(30)]
        down = [up[-1] * 0.99 ** i for i in range(1, 30)]
        engine = run(prices := up + down, BASE)
        sells = [t for t in engine.trades if t.direction == Direction.SHORT]
        assert sells, "跌破均线应清仓"

    def test_flat_at_end_of_downtrend(self):
        up = [10 * 1.01 ** i for i in range(30)]
        down = [up[-1] * 0.98 ** i for i in range(1, 40)]
        engine = run(up + down, BASE)
        assert engine.strategy.signal == SIGNAL_FLAT
        assert engine.strategy.get_pos(SYMBOL) == 0

    def test_no_trade_before_warmup(self):
        engine = run([10.0] * 8, dict(BASE, ma_window=10))
        assert not engine.trades

    def test_switch_count_tracked(self):
        up = [10 * 1.01 ** i for i in range(25)]
        down = [up[-1] * 0.98 ** i for i in range(1, 25)]
        up2 = [down[-1] * 1.01 ** i for i in range(1, 25)]
        engine = run(up + down + up2, BASE)
        assert engine.strategy.switch_count >= 2

    def test_avoids_drawdown_versus_buy_and_hold(self):
        """择时的核心价值：躲开系统性下跌。
        构造先涨后暴跌，择时的期末净值应高于买入持有。"""
        up = [10 * 1.01 ** i for i in range(40)]
        crash = [up[-1] * 0.97 ** i for i in range(1, 40)]
        prices = up + crash

        engine = run(prices, BASE)
        equity = engine.get_equity_df()["equity"]
        timing_ret = equity.iloc[-1] / equity.iloc[0] - 1
        hold_ret = prices[-1] / prices[0] - 1
        assert timing_ret > hold_ret, "择时应躲开暴跌"


class TestStatePersistence:
    def test_signal_in_variables(self):
        """信号必须可持久化 —— 重启后不知道当前该持有还是空仓，
        会在下一根 Bar 重新建仓或清仓，白付一次交易成本"""
        s = make(BASE)
        assert "signal" in s.variables
        assert "switch_count" in s.variables

    def test_restore_signal(self):
        s = make(BASE)
        s.restore_variables({"signal": SIGNAL_LONG, "switch_count": 7,
                             "pos": {SYMBOL: 1000}})
        assert s.signal == SIGNAL_LONG
        assert s.switch_count == 7
        assert s.get_pos(SYMBOL) == 1000


class TestAsymmetricBuffer:
    """买卖限价缓冲必须不对称。

    真实 bug：原先买卖都用 2% 缓冲，遇到 -3%/日的崩盘时卖单限价
    （收盘×0.98）高于次日开盘（收盘×0.97），委托被拒 ——
    恰恰在最需要离场时离不掉，择时反而跑输买入持有。
    """

    def test_exit_buffer_must_not_be_smaller(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="卖出缓冲"):
            engine.add_strategy(IndexTimingStrategy, [SYMBOL],
                                {"price_buffer": 0.05,
                                 "exit_price_buffer": 0.02})

    def test_default_exit_buffer_wider(self):
        s = make(BASE)
        assert s.exit_price_buffer > s.price_buffer

    def test_sell_fills_during_gap_down_crash(self):
        """-3%/日的连续跳空中，卖单必须能成交"""
        up = [10 * 1.01 ** i for i in range(40)]
        crash = [up[-1] * 0.97 ** i for i in range(1, 40)]
        engine = run(up + crash, BASE)

        sells = [t for t in engine.trades if t.direction == Direction.SHORT]
        assert sells, "崩盘中必须能卖出"
        rejected = [o for o in engine.orders
                    if "限价高于开盘价" in (o.message or "")]
        assert not rejected, f"卖单不应被限价拒绝，实际被拒 {len(rejected)} 笔"
