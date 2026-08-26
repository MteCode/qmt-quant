"""选股策略与调仓逻辑测试。"""
import pandas as pd

from qmtquant.core.constants import Direction, Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.backtest_engine import BacktestEngine
from qmtquant.strategy.portfolio import PortfolioStrategy
from qmtquant.universe.providers import StaticUniverse

SYMBOLS = ["000001.SZSE", "000002.SZSE", "600000.SSE", "600519.SSE"]


def make_bars(days: int, prices: dict[str, list[float]]) -> list[BarData]:
    """按 {symbol: [每日收盘价]} 构造多标的日线"""
    dates = pd.bdate_range("2023-01-02", periods=days)
    bars = []
    for vt_symbol, closes in prices.items():
        symbol, ex = vt_symbol.split(".")
        for dt, c in zip(dates, closes):
            bars.append(BarData(
                symbol=symbol, exchange=Exchange(ex), datetime=dt.to_pydatetime(),
                interval=Interval.DAILY, open_price=c, high_price=c,
                low_price=c, close_price=c, volume=1_000_000,
            ))
    return bars


class TopNStrategy(PortfolioStrategy):
    """固定选前 N 个候选，便于断言调仓行为"""
    rebalance_days = 2
    max_holdings = 2
    cash_buffer = 0.05

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.picks: list[str] = []

    def select(self, bars, candidates):
        return self.picks or sorted(candidates)


class TestRebalance:
    def _run(self, picks_by_call=None, days=8):
        prices = {s: [10.0] * days for s in SYMBOLS}
        engine = BacktestEngine(initial_capital=1_000_000)
        engine.load_data(make_bars(days, prices))
        engine.set_universe(StaticUniverse(SYMBOLS))
        engine.add_strategy(TopNStrategy, SYMBOLS)
        if picks_by_call:
            engine.strategy.picks = picks_by_call
        engine.run()
        return engine

    def test_buys_selected_symbols(self):
        engine = self._run(picks_by_call=["000001.SZSE", "600519.SSE"])
        bought = {t.vt_symbol for t in engine.trades if t.direction == Direction.LONG}
        assert bought == {"000001.SZSE", "600519.SSE"}

    def test_respects_max_holdings(self):
        """选出 4 只但 max_holdings=2，只能买 2 只"""
        engine = self._run(picks_by_call=SYMBOLS)
        bought = {t.vt_symbol for t in engine.trades if t.direction == Direction.LONG}
        assert len(bought) == 2

    def test_equal_weight_allocation(self):
        engine = self._run(picks_by_call=["000001.SZSE", "600519.SSE"])
        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        # 等权 + 5% 现金缓冲 -> 每只约 47.5 万 / 10 元 = 47500 股，取整到百股
        for t in buys:
            assert 40000 <= t.volume <= 50000
        assert buys[0].volume == buys[1].volume

    def test_all_volumes_are_round_lots(self):
        engine = self._run(picks_by_call=["000001.SZSE", "600519.SSE"])
        for t in engine.trades:
            if t.direction == Direction.LONG:
                assert t.volume % 100 == 0

    def test_suspended_symbol_not_bought(self):
        days = 8
        prices = {s: [10.0] * days for s in SYMBOLS}
        bars = make_bars(days, prices)
        for b in bars:
            if b.vt_symbol == "000001.SZSE":
                b.suspended = True
        engine = BacktestEngine(initial_capital=1_000_000)
        engine.load_data(bars)
        engine.set_universe(StaticUniverse(SYMBOLS))
        engine.add_strategy(TopNStrategy, SYMBOLS)
        engine.strategy.picks = ["000001.SZSE", "600519.SSE"]
        engine.run()

        bought = {t.vt_symbol for t in engine.trades if t.direction == Direction.LONG}
        assert "000001.SZSE" not in bought


class TestSellBeforeBuy:
    """调仓时卖单必须先撮合，否则买单会因资金不足大面积失败"""

    def test_sells_matched_before_buys(self):
        engine = BacktestEngine(initial_capital=100_000)
        # 手工挂：先买单后卖单，撮合时应被重排为卖在前
        engine.send_order("s", "000001.SZSE", Direction.LONG, 10.0, 1000)
        engine.send_order("s", "600519.SSE", Direction.SHORT, 10.0, 1000)

        directions = [r.direction for r in engine.pending_orders]
        assert directions == [Direction.LONG, Direction.SHORT]

        # 触发一次撮合，验证排序生效
        bars = {b.vt_symbol: b for b in make_bars(1, {s: [10.0] for s in SYMBOLS})}
        engine._prev_bars = {}
        engine._current_dt = list(bars.values())[0].datetime
        engine.positions["600519.SSE"] = {"volume": 1000.0, "available": 1000.0,
                                          "price": 10.0}
        engine._match_pending(bars)

        assert len(engine.trades) == 2
        # 第一笔成交必须是卖出
        assert engine.trades[0].direction is Direction.SHORT

    def test_rotation_frees_cash_for_new_buys(self):
        """满仓时轮动：卖旧买新应当都能成交"""
        days = 10
        prices = {s: [10.0] * days for s in SYMBOLS}
        engine = BacktestEngine(initial_capital=100_000)
        engine.load_data(make_bars(days, prices))
        engine.set_universe(StaticUniverse(SYMBOLS))
        engine.add_strategy(TopNStrategy, SYMBOLS)

        strategy = engine.strategy
        strategy.picks = ["000001.SZSE", "000002.SZSE"]

        # 第 4 次调仓后切换目标，制造轮动
        original_select = strategy.select

        def switching_select(bars, candidates):
            if strategy._bar_count >= 6:
                return ["600000.SSE", "600519.SSE"]
            return original_select(bars, candidates)

        strategy.select = switching_select
        engine.run()

        sells = [t for t in engine.trades if t.direction == Direction.SHORT]
        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        assert sells, "轮动应产生卖出"
        # 换入的标的确实买到了
        assert {"600000.SSE", "600519.SSE"} & {t.vt_symbol for t in buys}


class TestUniverseIntegration:
    def test_engine_uses_universe(self):
        engine = BacktestEngine()
        engine.set_universe(StaticUniverse(["000001.SZ", "600519.SH"]))
        assert engine.get_universe(pd.Timestamp("2024-01-02")) == \
            ["000001.SZSE", "600519.SSE"]

    def test_no_universe_falls_back_to_strategy_symbols(self):
        engine = BacktestEngine()
        engine.load_data(make_bars(3, {s: [10.0] * 3 for s in SYMBOLS}))
        engine.add_strategy(TopNStrategy, SYMBOLS)
        assert engine.get_universe() == SYMBOLS
