"""双均线金叉死叉测试。

重点：穿越检测的边界。`prev_diff <= 0 < diff` 这类条件里等号放哪边，
决定了「两线相等后分开」算不算穿越 —— 写错会漏掉一半信号。
"""
import pandas as pd
import pytest

from qmtquant.core.constants import Direction, Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.backtest_engine import BacktestEngine
from qmtquant.strategy.ma_cross import MaCrossStrategy

SYMBOL = "600519.SSE"
BASE = {"fast_window": 3, "slow_window": 5}


def v_shape(down: int = 8, up: int = 14, down_step: float = 0.03,
            up_step: float = 0.02, start: float = 20.0) -> list[float]:
    """先跌后涨的 V 形，用于制造金叉。

    两个约束限制了这里能用的斜率，都是踩出来的：

    1. **单日涨跌幅须在 10% 以内** —— 主板涨跌停是引擎的硬约束，
       构造 +19% 的数据会被判为「开盘涨停，无法买入」，拿不到成交。
    2. **上涨斜率须小于 price_buffer（默认 3%）** —— 信号在 T 日收盘产生、
       T+1 开盘成交，日涨 3% 时买单限价（收盘×1.03）恰好等于次日开盘价，
       会被判「限价低于开盘价，未成交」。故上行取 2%。
    """
    prices = [start * (1 - down_step) ** i for i in range(down)]
    prices += [prices[-1] * (1 + up_step) ** i for i in range(1, up + 1)]
    return prices


def bars_of(closes: list[float]) -> list[BarData]:
    dates = pd.bdate_range("2023-01-02", periods=len(closes))
    return [BarData(symbol="600519", exchange=Exchange.SSE,
                    datetime=d.to_pydatetime(), interval=Interval.DAILY,
                    open_price=c, high_price=c, low_price=c, close_price=c,
                    volume=1_000_000, turnover=c * 1_000_000)
            for d, c in zip(dates, closes)]


def run(closes, setting=None, capital=1_000_000):
    engine = BacktestEngine(initial_capital=capital)
    engine.load_data(bars_of(closes))
    engine.add_strategy(MaCrossStrategy, [SYMBOL], dict(BASE, **(setting or {})))
    engine.run()
    return engine


def make(setting=None) -> MaCrossStrategy:
    return MaCrossStrategy(BacktestEngine(), "T", [SYMBOL],
                           dict(BASE, **(setting or {})))


class TestValidation:
    def test_slow_must_exceed_fast(self):
        with pytest.raises(ValueError, match="slow_window"):
            make({"fast_window": 20, "slow_window": 5})

    def test_equal_windows_rejected(self):
        """两条窗口相同则 diff 恒为 0，永远不会穿越"""
        with pytest.raises(ValueError, match="slow_window"):
            make({"fast_window": 10, "slow_window": 10})

    def test_fast_window_minimum(self):
        with pytest.raises(ValueError, match="fast_window"):
            make({"fast_window": 0, "slow_window": 5})

    def test_position_ratio_range(self):
        with pytest.raises(ValueError, match="position_ratio"):
            make({"position_ratio": 1.5})

    def test_exit_buffer_must_not_be_smaller(self):
        with pytest.raises(ValueError, match="卖出缓冲"):
            make({"price_buffer": 0.05, "exit_price_buffer": 0.02})

    def test_default_exit_buffer_wider(self):
        assert make().exit_price_buffer > make().price_buffer

    def test_float_params_coerced(self):
        s = make({"fast_window": 5.0, "slow_window": 20.0})
        assert isinstance(s.fast_window, int)
        assert isinstance(s.slow_window, int)


class TestCrossDetection:
    def test_golden_cross_buys(self):
        engine = run(v_shape())
        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        assert buys, "金叉应买入"

    def test_death_cross_sells(self):
        up = v_shape()
        # 涨完再跌回去，制造死叉
        down = [up[-1] * 0.97 ** i for i in range(1, 15)]
        engine = run(up + down)
        sells = [t for t in engine.trades if t.direction == Direction.SHORT]
        assert sells, "死叉应卖出"

    def test_no_signal_before_slow_window(self):
        engine = run([10, 11, 12], {"fast_window": 3, "slow_window": 5})
        assert not engine.trades

    def test_flat_prices_never_cross(self):
        """价格恒定时两线重合，diff 恒为 0，不应产生任何交易"""
        engine = run([10.0] * 40)
        assert not engine.trades

    def test_trades_strictly_alternate(self):
        """买卖必须严格交替 —— 连续两次买入意味着持仓期间重复建仓，
        会突破仓位上限；连续两次卖出意味着卖了不持有的量。

        注意不能简单断言「只买一次」：回调触发死叉离场后再次金叉买入
        是正确行为，此时确实会有两次买入。真正的不变量是交替性。
        """
        up = v_shape()
        # 反复小幅震荡，制造多次穿越
        wobble = list(up)
        for _ in range(4):
            wobble += [wobble[-1] * 0.98 ** i for i in range(1, 6)]
            wobble += [wobble[-1] * 1.02 ** i for i in range(1, 8)]
        engine = run(wobble)

        dirs = [t.direction for t in engine.trades]
        assert dirs, "震荡行情应产生交易"
        assert dirs[0] == Direction.LONG, "首笔必须是买入"
        for i in range(1, len(dirs)):
            assert dirs[i] != dirs[i - 1], (
                f"第 {i + 1} 笔与前一笔同向（{dirs[i]}），买卖未交替")


class TestMemoryBound:
    def test_window_does_not_grow(self):
        """长期运行时收盘价列表必须有界，否则内存持续增长"""
        s = make({"fast_window": 3, "slow_window": 5})
        s.trading = False
        for b in bars_of([10.0 + i * 0.01 for i in range(500)]):
            s.on_bar(b)
        assert len(s.closes[SYMBOL]) <= s.slow_window + 5


class TestBarFiltering:
    def test_suspended_ignored(self):
        s = make()
        b = bars_of([10.0])[0]
        b.suspended = True
        s.on_bar(b)
        assert len(s.closes[SYMBOL]) == 0

    def test_zero_price_ignored(self):
        s = make()
        s.on_bar(bars_of([0.0])[0])
        assert len(s.closes[SYMBOL]) == 0


class TestAsymmetricBuffer:
    def test_sell_fills_during_gap_down(self):
        """连续跳空下跌中卖单必须能成交 —— 对称缓冲会在此处失效"""
        up = v_shape()
        crash = [up[-1] * 0.97 ** i for i in range(1, 25)]
        engine = run(up + crash)
        sells = [t for t in engine.trades if t.direction == Direction.SHORT]
        assert sells, "崩盘中必须能卖出"


class TestStatePersistence:
    def test_trade_count_in_variables(self):
        assert "trade_count" in make().variables

    def test_restore(self):
        s = make()
        s.restore_variables({"trade_count": 12, "pos": {SYMBOL: 500}})
        assert s.trade_count == 12
        assert s.get_pos(SYMBOL) == 500
