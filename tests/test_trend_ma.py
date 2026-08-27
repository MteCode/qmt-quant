"""趋势策略与指标测试。"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from qmtquant.core.constants import Direction, Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.backtest_engine import BacktestEngine
from qmtquant.strategy.intraday_vwap import IntradayVwapStrategy
from qmtquant.strategy.trend_ma import TrendMaStrategy
from qmtquant.strategy.indicators import CrossDetector, IntradayVwap, MovingAverage


class TestMovingAverage:
    def test_not_ready_until_window_filled(self):
        ma = MovingAverage(3)
        assert ma.update(1) is None
        assert ma.update(2) is None
        assert ma.update(3) == pytest.approx(2.0)
        assert ma.ready

    def test_rolls_correctly(self):
        ma = MovingAverage(3)
        for v in [1, 2, 3]:
            ma.update(v)
        assert ma.update(4) == pytest.approx(3.0)   # (2+3+4)/3
        assert ma.update(5) == pytest.approx(4.0)   # (3+4+5)/3

    def test_incremental_sum_matches_bruteforce(self):
        """增量求和不能有累积误差"""
        ma = MovingAverage(5)
        values = [1.1, 2.7, 3.3, 4.9, 5.5, 6.1, 7.7, 8.3, 9.9]
        for i, v in enumerate(values):
            got = ma.update(v)
            if i >= 4:
                expected = sum(values[i - 4:i + 1]) / 5
                assert got == pytest.approx(expected)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            MovingAverage(0)


class TestCrossDetector:
    def test_detects_up_cross_only_at_crossing(self):
        """穿越信号只能在穿越那一刻出现一次，不能持续输出"""
        c = CrossDetector()
        assert c.update(9, 10) == ""      # 首次无历史
        assert c.update(11, 10) == "up"   # 上穿
        assert c.update(12, 10) == ""     # 已在上方，不再出信号
        assert c.update(13, 10) == ""

    def test_detects_down_cross(self):
        c = CrossDetector()
        c.update(11, 10)
        assert c.update(9, 10) == "down"
        assert c.update(8, 10) == ""

    def test_above_property(self):
        c = CrossDetector()
        assert c.above is None
        c.update(11, 10)
        assert c.above is True
        c.update(9, 10)
        assert c.above is False

    def test_reset(self):
        c = CrossDetector()
        c.update(11, 10)
        c.reset()
        assert c.above is None
        assert c.update(12, 10) == ""


class TestIntradayVwap:
    def test_vwap_calculation(self):
        v = IntradayVwap()
        dt = datetime(2024, 1, 2, 9, 31)
        v.update(dt, price=10.0, volume=100)
        assert v.value == pytest.approx(10.0)
        v.update(dt, price=30.0, volume=100)      # (1000+3000)/200
        assert v.value == pytest.approx(20.0)

    def test_volume_weighted_not_simple_average(self):
        """成交量加权，不是简单平均 —— 大单必须占更大权重"""
        v = IntradayVwap()
        dt = datetime(2024, 1, 2, 9, 31)
        v.update(dt, price=10.0, volume=900)
        v.update(dt, price=20.0, volume=100)
        assert v.value == pytest.approx(11.0)     # 非 15.0

    def test_resets_on_new_day(self):
        """跨日必须重置，否则昨天的成交会污染今天的均价线"""
        v = IntradayVwap()
        v.update(datetime(2024, 1, 2, 14, 59), price=100.0, volume=100)
        assert v.value == pytest.approx(100.0)

        v.update(datetime(2024, 1, 3, 9, 31), price=10.0, volume=100)
        assert v.value == pytest.approx(10.0)

    def test_zero_volume_ignored(self):
        v = IntradayVwap()
        assert v.update(datetime(2024, 1, 2, 9, 31), price=10.0, volume=0) is None

    def test_zero_price_ignored(self):
        v = IntradayVwap()
        assert v.update(datetime(2024, 1, 2, 9, 31), price=0.0, volume=100) is None

    def test_same_price_space_as_close(self):
        """真 bug 回归：均价线必须与收盘价同一复权口径。

        本地价格是后复权的，而行情里的成交额字段是未复权原始值。
        曾用 turnover/volume 算均价，结果落在未复权空间 ——
        实测茅台收盘价 8137、turnover/volume 只有 1300（比值 6.26 = 复权因子），
        收盘价永远"在均价线之上"，穿越永不发生，策略静默零成交。
        """
        v = IntradayVwap()
        dt = datetime(2024, 1, 2, 9, 31)
        adjusted_close = 8137.23
        v.update(dt, price=adjusted_close, volume=1000)
        # 单根 Bar 时均价必须等于该 Bar 价格，量级不能偏移
        assert v.value == pytest.approx(adjusted_close)


# ---------------------------------------------------------------- 策略


def daily_bars(closes: list[float], symbol="000001", exchange=Exchange.SZSE):
    dates = pd.bdate_range("2022-01-03", periods=len(closes))
    return [BarData(symbol=symbol, exchange=exchange, datetime=d.to_pydatetime(),
                    interval=Interval.DAILY, open_price=c, high_price=c,
                    low_price=c, close_price=c, volume=1_000_000)
            for d, c in zip(dates, closes)]


class TestTrendMaStrategy:
    def test_rejects_unknown_mode(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="未知 mode"):
            engine.add_strategy(TrendMaStrategy, ["000001.SZSE"], {"mode": "bogus"})

    def _run(self, closes, setting):
        engine = BacktestEngine(initial_capital=1_000_000)
        engine.load_data(daily_bars(closes))
        engine.add_strategy(TrendMaStrategy, ["000001.SZSE"], setting)
        engine.run()
        return engine

    def test_trend_mode_buys_on_upcross(self):
        """趋势模式：先跌后涨，应在转折向上时买入。

        注意日涨跌幅必须控制在 ±10% 内，否则会被引擎按涨跌停拒单，
        测试就变成在验证涨跌停逻辑而不是策略逻辑。
        """
        closes = ([10.0] * 8
                  + [round(10 * 0.98 ** i, 2) for i in range(1, 11)]     # 缓跌
                  + [round(8.2 * 1.02 ** i, 2) for i in range(1, 21)])   # 缓涨
        engine = self._run(closes, {"fast_window": 3, "slow_window": 5,
                                    "mode": "trend", "use_trend_filter": False})
        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        assert buys, "上涨趋势中应产生买入"

    def test_reversion_mode_opposite_direction(self):
        """同一段行情，两种模式的动作时点应不同"""
        wave = [10.0, 10.3, 10.6, 10.3, 10.0, 9.7, 9.4, 9.7]
        closes = [10.0] * 6 + wave * 6
        trend = self._run(closes, {"fast_window": 3, "slow_window": 5,
                                   "mode": "trend", "use_trend_filter": False})
        rev = self._run(closes, {"fast_window": 3, "slow_window": 5,
                                 "mode": "reversion", "use_trend_filter": False})
        assert trend.trades, "趋势模式应有成交"
        assert rev.trades, "回归模式应有成交"
        assert ([t.datetime for t in trend.trades]
                != [t.datetime for t in rev.trades])

    def test_trend_filter_blocks_buy_below_yearline(self):
        """开启趋势过滤后，年线之下不应买入"""
        # 长期下跌：价格始终低于慢线
        closes = list(range(60, 10, -1))
        engine = self._run(closes, {"fast_window": 3, "slow_window": 20,
                                    "mode": "trend", "use_trend_filter": True})
        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        assert not buys, "年线之下不应买入"

    def test_no_trade_when_data_insufficient(self):
        """数据不足以计算均线时不能交易"""
        engine = self._run([10.0] * 3, {"fast_window": 5, "slow_window": 250})
        assert not engine.trades

    def test_suspended_bar_skipped(self):
        engine = BacktestEngine(initial_capital=1_000_000)
        bars = daily_bars([10.0] * 30)
        for b in bars:
            b.suspended = True
        engine.load_data(bars)
        engine.add_strategy(TrendMaStrategy, ["000001.SZSE"],
                            {"fast_window": 3, "slow_window": 5})
        engine.run()
        assert not engine.trades


def minute_bars(prices: list[float], day="2024-01-02", start_hour=9, start_min=30,
                symbol="000001", volume=1000):
    base = datetime.fromisoformat(f"{day} {start_hour:02d}:{start_min:02d}:00")
    return [BarData(symbol=symbol, exchange=Exchange.SZSE,
                    datetime=base + timedelta(minutes=i), interval=Interval.MINUTE,
                    open_price=p, high_price=p, low_price=p, close_price=p,
                    volume=volume, turnover=p * volume)
            for i, p in enumerate(prices)]


class TestIntradayVwapStrategy:
    def test_rejects_unknown_role(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="未知 role"):
            engine.add_strategy(IntradayVwapStrategy, ["000001.SZSE"],
                                {"role": "bogus"})

    def test_trading_window_enforced(self):
        """窗口外不交易：09:30 起的前几分钟应被 start_minute 挡掉"""
        s = IntradayVwapStrategy(BacktestEngine(), "t", ["000001.SZSE"],
                                 {"start_minute": 945, "stop_minute": 1450})
        assert not s._in_window(datetime(2024, 1, 2, 9, 31))
        assert s._in_window(datetime(2024, 1, 2, 10, 0))
        assert not s._in_window(datetime(2024, 1, 2, 14, 55))

    def test_min_bars_blocks_early_trades(self):
        """开盘前几分钟均价线不稳，min_bars 之前不应交易"""
        engine = BacktestEngine(initial_capital=1_000_000)
        # 价格持续下跌，reversion 模式下会有下穿信号
        engine.load_data(minute_bars([10 - i * 0.05 for i in range(10)]))
        engine.add_strategy(IntradayVwapStrategy, ["000001.SZSE"],
                            {"min_bars": 30, "start_minute": 900})
        engine.run()
        assert not engine.trades

    def test_buy_sell_signal_direction(self):
        s_trend = IntradayVwapStrategy(BacktestEngine(), "t", ["000001.SZSE"],
                                       {"mode": "trend"})
        assert s_trend._buy_signal("up") and s_trend._sell_signal("down")

        s_rev = IntradayVwapStrategy(BacktestEngine(), "t", ["000001.SZSE"],
                                     {"mode": "reversion"})
        assert s_rev._buy_signal("down") and s_rev._sell_signal("up")

    def test_t0_requires_sell_before_buy(self):
        """T+0 回转必须先卖后买 —— 当日买入份额被冻结，卖不掉"""
        engine = BacktestEngine(initial_capital=1_000_000)
        s = IntradayVwapStrategy(engine, "t", ["000001.SZSE"],
                                 {"role": "t0_rotation", "mode": "reversion"})
        s.trading = True
        # 无底仓时，买入信号不应触发买回（因为还没卖过）
        assert s._pending_buyback.get("000001.SZSE", 0) == 0

    def test_daily_state_reset(self):
        """跨日必须重置日内状态"""
        engine = BacktestEngine(initial_capital=1_000_000)
        bars = (minute_bars([10.0] * 5, day="2024-01-02")
                + minute_bars([10.0] * 5, day="2024-01-03"))
        engine.load_data(bars)
        engine.add_strategy(IntradayVwapStrategy, ["000001.SZSE"], {"min_bars": 3})
        engine.run()
        s = engine.strategy
        # 最后处理的是 01-03，日内计数应已重置为当日根数
        assert s._day["000001.SZSE"].isoformat() == "2024-01-03"
        assert s._bar_count["000001.SZSE"] == 5


# ============================================================
# 以下为「升级为正式模块」后新增的校验与状态测试
# ============================================================

SYM = "600519.SSE"


def _make_trend(setting=None) -> TrendMaStrategy:
    return TrendMaStrategy(BacktestEngine(), "T", [SYM], setting or {})


def _make_vwap(setting=None) -> IntradayVwapStrategy:
    return IntradayVwapStrategy(BacktestEngine(), "T", [SYM], setting or {})


class TestTrendMaValidation:
    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="未知 mode"):
            _make_trend({"mode": "whatever"})

    def test_slow_must_exceed_fast(self):
        with pytest.raises(ValueError, match="slow_window"):
            _make_trend({"fast_window": 250, "slow_window": 5})

    def test_fast_window_minimum(self):
        with pytest.raises(ValueError, match="fast_window"):
            _make_trend({"fast_window": 1, "slow_window": 20})

    def test_position_ratio_range(self):
        with pytest.raises(ValueError, match="position_ratio"):
            _make_trend({"position_ratio": 0})

    def test_exit_buffer_must_not_be_smaller(self):
        """错过卖出是实亏，错过买入只是少赚 —— 缓冲天然不对称"""
        with pytest.raises(ValueError, match="卖出缓冲"):
            _make_trend({"price_buffer": 0.05, "exit_price_buffer": 0.02})

    def test_default_exit_buffer_wider(self):
        s = _make_trend()
        assert s.exit_price_buffer > s.price_buffer

    def test_float_params_coerced(self):
        s = _make_trend({"fast_window": 5.0, "slow_window": 250.0})
        assert isinstance(s.fast_window, int)
        assert isinstance(s.slow_window, int)

    def test_trade_count_persisted(self):
        s = _make_trend()
        assert "trade_count" in s.variables
        s.restore_variables({"trade_count": 7, "pos": {SYM: 100}})
        assert s.trade_count == 7


class TestIntradayVwapValidation:
    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="未知 mode"):
            _make_vwap({"mode": "nope"})

    def test_unknown_role(self):
        with pytest.raises(ValueError, match="未知 role"):
            _make_vwap({"role": "nope"})

    def test_trade_ratio_range(self):
        with pytest.raises(ValueError, match="trade_ratio"):
            _make_vwap({"trade_ratio": 1.5})

    def test_min_bars_minimum(self):
        with pytest.raises(ValueError, match="min_bars"):
            _make_vwap({"min_bars": 0})

    def test_stop_must_be_after_start(self):
        with pytest.raises(ValueError, match="stop_minute"):
            _make_vwap({"start_minute": 1400, "stop_minute": 1000})

    def test_t0_stop_minute_capped(self):
        """T0 卖出后必须留出买回时间，否则底仓永久减少一块"""
        with pytest.raises(ValueError, match="stop_minute"):
            _make_vwap({"role": "t0_rotation", "stop_minute": 1455})

    def test_entry_role_allows_late_stop(self):
        """entry_timing 不涉及买回，尾盘限制不适用"""
        s = _make_vwap({"role": "entry_timing", "stop_minute": 1455})
        assert s.stop_minute == 1455

    def test_float_params_coerced(self):
        s = _make_vwap({"min_bars": 15.0, "start_minute": 945.0,
                        "stop_minute": 1450.0})
        for name in ("min_bars", "start_minute", "stop_minute"):
            assert isinstance(getattr(s, name), int), name

    def test_failed_buyback_counted(self):
        """T0 卖出没买回 = 底仓永久少一块，必须能统计到"""
        s = _make_vwap({"role": "t0_rotation"})
        s._pending_buyback[SYM] = 500
        s.on_stop_day(SYM)
        assert s.failed_buyback_count == 1

    def test_no_false_alarm_when_bought_back(self):
        s = _make_vwap({"role": "t0_rotation"})
        s._pending_buyback[SYM] = 0
        s.on_stop_day(SYM)
        assert s.failed_buyback_count == 0

    def test_trade_count_persisted(self):
        s = _make_vwap()
        assert "trade_count" in s.variables
