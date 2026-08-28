"""通道突破策略测试。

最容易写错的一处：通道必须**先取值再 update**。
反了的话「突破 N 日新高」恒真 —— 今天的最高价当然是含今天在内的
最高价之一，信号每根 Bar 都触发。
"""
import pandas as pd
import pytest

from qmtquant.core.constants import Direction, Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.backtest_engine import BacktestEngine
from qmtquant.strategy.breakout import BreakoutStrategy
from qmtquant.strategy.indicators import AverageTrueRange, Donchian

SYMBOL = "600519.SSE"
BASE = {"entry_window": 10, "exit_window": 5, "atr_window": 5}


def bars_from(rows: list[tuple[float, float, float]]) -> list[BarData]:
    """rows = [(high, low, close), ...]"""
    dates = pd.bdate_range("2023-01-02", periods=len(rows))
    return [BarData(symbol="600519", exchange=Exchange.SSE,
                    datetime=d.to_pydatetime(), interval=Interval.DAILY,
                    open_price=c, high_price=h, low_price=lo, close_price=c,
                    volume=1_000_000, turnover=c * 1_000_000)
            for d, (h, lo, c) in zip(dates, rows)]


def flat_then_breakout(flat_n: int = 15, up_n: int = 20,
                       base: float = 20.0, step: float = 0.02):
    """先横盘蓄势、再稳步上行 —— 制造一次干净的向上突破。

    涨幅必须小于 price_buffer（3%），否则买单限价低于次日开盘价，
    会被引擎判为「限价低于开盘价，未成交」
    """
    rows = [(base * 1.005, base * 0.995, base) for _ in range(flat_n)]
    p = base
    for _ in range(up_n):
        p *= 1 + step
        rows.append((p * 1.005, p * 0.995, p))
    return rows


def run(rows, setting=None, capital=1_000_000):
    engine = BacktestEngine(initial_capital=capital)
    engine.load_data(bars_from(rows))
    engine.add_strategy(BreakoutStrategy, [SYMBOL],
                        dict(BASE, **(setting or {})))
    engine.run()
    return engine


def make(setting=None) -> BreakoutStrategy:
    return BreakoutStrategy(BacktestEngine(), "T", [SYMBOL],
                            dict(BASE, **(setting or {})))


class TestDonchian:
    def test_not_ready_until_filled(self):
        d = Donchian(3)
        d.update(10, 9)
        d.update(11, 10)
        assert d.upper is None and not d.ready
        d.update(12, 11)
        assert d.upper == 12 and d.lower == 9

    def test_rolls(self):
        d = Donchian(3)
        for h, lo in [(10, 9), (11, 10), (12, 11), (13, 12)]:
            d.update(h, lo)
        assert d.upper == 13 and d.lower == 10

    def test_window_minimum(self):
        with pytest.raises(ValueError, match="window"):
            Donchian(1)


class TestATR:
    def test_includes_gaps(self):
        """真实波幅必须含跳空 —— 只看当日振幅会严重低估风险"""
        atr = AverageTrueRange(1)
        atr.update(high=10.0, low=9.0, close=10.0)
        # 次日整体跳空到 12-13：当日振幅仅 1.0，但含跳空的 TR 是 3.0
        assert atr.update(high=13.0, low=12.0, close=12.5) == pytest.approx(3.0)

    def test_first_bar_uses_range(self):
        atr = AverageTrueRange(1)
        assert atr.update(10.0, 9.0, 9.5) == pytest.approx(1.0)

    def test_not_ready_until_window(self):
        atr = AverageTrueRange(3)
        assert atr.update(10, 9, 9.5) is None
        assert atr.update(10, 9, 9.5) is None
        assert atr.update(10, 9, 9.5) == pytest.approx(1.0)

    def test_reset(self):
        atr = AverageTrueRange(2)
        atr.update(10, 9, 9.5)
        atr.reset()
        assert not atr.ready and atr.value is None


class TestValidation:
    def test_exit_window_must_be_smaller(self):
        """进场要慢（确认趋势），出场要快（快速认错）。
        相等的话进出同一条线，来回打脸"""
        with pytest.raises(ValueError, match="exit_window 必须小于"):
            make({"entry_window": 20, "exit_window": 20})

    def test_entry_window_minimum(self):
        with pytest.raises(ValueError, match="entry_window"):
            make({"entry_window": 1, "exit_window": 1})

    def test_stop_atr_positive(self):
        with pytest.raises(ValueError, match="stop_atr"):
            make({"stop_atr": 0})

    def test_max_units_minimum(self):
        with pytest.raises(ValueError, match="max_units"):
            make({"max_units": 0})

    def test_exit_buffer_not_smaller(self):
        with pytest.raises(ValueError, match="卖出缓冲"):
            make({"price_buffer": 0.05, "exit_price_buffer": 0.01})

    def test_float_params_coerced(self):
        s = make({"entry_window": 20.0, "exit_window": 10.0,
                  "atr_window": 14.0, "max_units": 4.0})
        for name in ("entry_window", "exit_window", "atr_window", "max_units"):
            assert isinstance(getattr(s, name), int), name


class TestChannelLookahead:
    def test_channel_excludes_current_bar(self):
        """含当根的话「突破新高」恒真，每根 Bar 都会触发信号"""
        rows = [(20 + i, 19 + i, 19.5 + i) for i in range(30)]
        engine = run(rows)
        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        # 单调上行 30 根，若通道含当根会买 20+ 次
        assert len(buys) <= 5, f"疑似前视：买入 {len(buys)} 次"

    def test_no_trade_before_warmup(self):
        engine = run([(20.1, 19.9, 20.0)] * 8)
        assert not engine.trades


class TestEntryAndStop:
    def test_enters_on_breakout(self):
        engine = run(flat_then_breakout())
        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        assert buys, "向上突破应买入"

    def test_stop_is_atr_based(self):
        """止损价 = 入场价 − stop_atr × ATR"""
        s = make({"stop_atr": 2.0})
        s.trading = True
        for b in bars_from(flat_then_breakout(flat_n=15, up_n=3)):
            s.on_bar(b)
        pos = s.manager.get(SYMBOL)
        if pos is not None:
            atr = s.atr[SYMBOL].value
            assert pos.stop_price == pytest.approx(
                pos.entry_price - 2.0 * atr, rel=0.3)

    def test_position_size_bounded_by_risk(self):
        """单份仓位的亏损上限 = 总资产 × risk_per_trade。

        必须关掉加仓才测得准：金字塔每加一份都按 1% 算，
        4 份在止损上移前合计敞口可达 4% —— 这是海龟法则的设计，
        不是 bug，但会掩盖单份的不变量。
        """
        s = make({"risk_per_trade": 0.01, "add_atr": 0.0})
        s.trading = True
        for b in bars_from(flat_then_breakout(flat_n=15, up_n=3)):
            s.on_bar(b)
        pos = s.manager.get(SYMBOL)
        if pos is not None:
            assert pos.max_loss <= 1_000_000 * 0.01 * 1.01

    def test_no_entry_when_atr_exceeds_price(self):
        """ATR 大于价格一半时止损价为负，这种极端波动标的不碰"""
        s = make({"stop_atr": 2.0, "atr_window": 2})
        s.trading = True
        # 每根 Bar 振幅接近价格本身
        rows = [(20.0, 1.0, 10.0)] * 20
        for b in bars_from(rows):
            s.on_bar(b)
        assert not s.manager.has(SYMBOL)


class TestExit:
    def test_exits_on_channel_break(self):
        rows = flat_then_breakout()
        rows += [(rows[-1][2] * 0.97 ** i, rows[-1][2] * 0.96 ** i,
                  rows[-1][2] * 0.965 ** i) for i in range(1, 15)]
        engine = run(rows)
        sells = [t for t in engine.trades if t.direction == Direction.SHORT]
        assert sells, "跌破离场通道应卖出"

    def test_exit_stats_recorded(self):
        rows = flat_then_breakout()
        rows += [(rows[-1][2] * 0.97 ** i, rows[-1][2] * 0.96 ** i,
                  rows[-1][2] * 0.965 ** i) for i in range(1, 15)]
        engine = run(rows)
        assert engine.strategy.exit_stats, "离场必须有归因记录"

    def test_sell_fills_during_crash(self):
        """连续跳空下跌中必须能离场 —— 对称缓冲会在此处失效"""
        rows = flat_then_breakout()
        last = rows[-1][2]
        rows += [(last * 0.96 ** i, last * 0.95 ** i, last * 0.955 ** i)
                 for i in range(1, 20)]
        engine = run(rows)
        rejected = [o for o in engine.orders
                    if "限价高于开盘价" in (o.message or "")]
        assert not rejected, f"卖单被限价拒绝 {len(rejected)} 笔"


class TestPyramiding:
    def test_adds_units_on_profit(self):
        engine = run(flat_then_breakout(up_n=40), {"add_atr": 0.3,
                                                   "max_units": 4})
        assert engine.strategy.units.get(SYMBOL, 0) > 1, "持续上涨应加仓"

    def test_respects_max_units(self):
        engine = run(flat_then_breakout(up_n=60), {"add_atr": 0.1,
                                                   "max_units": 3})
        assert engine.strategy.units.get(SYMBOL, 0) <= 3

    def test_disabled_when_add_atr_zero(self):
        engine = run(flat_then_breakout(up_n=40), {"add_atr": 0.0})
        assert engine.strategy.units.get(SYMBOL, 0) == 1

    def test_stop_raised_on_add(self):
        """加仓后止损必须上移，否则份数越多总风险敞口越大"""
        s = make({"add_atr": 0.3, "max_units": 4})
        s.trading = True
        for b in bars_from(flat_then_breakout(up_n=40)):
            s.on_bar(b)
        pos = s.manager.get(SYMBOL)
        if pos is not None and s.units.get(SYMBOL, 1) > 1:
            assert pos.current_stop > pos.stop_price


class TestStatePersistence:
    def test_counters_in_variables(self):
        s = make()
        assert "trade_count" in s.variables
        assert "units" in s.variables

    def test_restore(self):
        s = make()
        s.restore_variables({"trade_count": 5, "units": {SYMBOL: 2},
                             "pos": {SYMBOL: 300}})
        assert s.trade_count == 5
        assert s.units == {SYMBOL: 2}
        assert s.get_pos(SYMBOL) == 300


class TestConcurrencyLimit:
    """并发上限：单笔风险控制管个股意外，并发上限管系统性风险。

    实测教训：只控单笔（每笔 1%）在 949 只候选上跑时，
    同时开出大量仓位，年化波动率 32%、最大回撤 -54%，比买入持有还差。
    「每笔只亏 1%」和「组合只亏 1%」是两回事 ——
    一次系统性下跌会让所有仓位**一起**触发止损。
    """

    def test_max_positions_validated(self):
        with pytest.raises(ValueError, match="max_positions"):
            make({"max_positions": 0})

    def test_coerced_to_int(self):
        assert isinstance(make({"max_positions": 5.0}).max_positions, int)

    def test_caps_concurrent_positions(self):
        syms = [f"60000{i}.SSE" for i in range(8)]
        engine = BacktestEngine(initial_capital=10_000_000)
        rows = flat_then_breakout()
        bars = []
        for i, sym in enumerate(syms):
            for b in bars_from(rows):
                bars.append(BarData(
                    symbol=sym.split(".")[0], exchange=Exchange.SSE,
                    datetime=b.datetime, interval=Interval.DAILY,
                    open_price=b.open_price, high_price=b.high_price,
                    low_price=b.low_price, close_price=b.close_price,
                    volume=b.volume, turnover=b.turnover))
        engine.load_data(bars)
        engine.add_strategy(BreakoutStrategy, syms,
                            dict(BASE, max_positions=3))
        engine.run()
        assert len(engine.strategy.manager.positions) <= 3

    def test_single_symbol_unaffected(self):
        engine = run(flat_then_breakout(), {"max_positions": 1})
        assert [t for t in engine.trades if t.direction == Direction.LONG]
