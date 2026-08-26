"""实盘引擎测试：用 SimGateway 走完整链路，验证风控不可绕过与回报路由。"""
import time

import pytest

from qmtquant.config import CostConfig, RiskConfig
from qmtquant.core.constants import Direction, Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.live_engine import LiveEngine
from qmtquant.event.engine import EVENT_BAR, Event, EventEngine
from qmtquant.gateway.sim_gateway import SimGateway
from qmtquant.risk.risk_manager import RiskManager
from qmtquant.strategy.base import StrategyBase


class PassiveStrategy(StrategyBase):
    """不主动下单，只记录收到的回报，由测试手工驱动"""
    parameters = []

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.orders_received = []
        self.trades_received = []

    def on_order(self, order):
        self.orders_received.append(order)

    def on_trade(self, trade):
        super().on_trade(trade)
        self.trades_received.append(trade)


@pytest.fixture
def setup():
    ee = EventEngine(timer_interval=0.05)
    ee.start()
    gw = SimGateway(ee, "SIM", initial_capital=1_000_000, cost=CostConfig())
    rm = RiskManager(RiskConfig(max_order_value=1_000_000), ee)
    engine = LiveEngine(ee, gw, rm)
    gw.connect({})
    strategy = engine.add_strategy(PassiveStrategy, "S1", ["000001.SZSE"])
    strategy.inited = True
    strategy.trading = True
    _drain(ee)
    yield engine, gw, rm, strategy
    ee.stop()


def _drain(ee: EventEngine, timeout: float = 2.0) -> None:
    """等事件队列处理干净"""
    deadline = time.time() + timeout
    while ee.qsize > 0 and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.15)


def _push_bar(ee: EventEngine, price: float, symbol="000001") -> None:
    from datetime import datetime
    bar = BarData(symbol=symbol, exchange=Exchange.SZSE, datetime=datetime.now(),
                  interval=Interval.DAILY, open_price=price, high_price=price,
                  low_price=price, close_price=price, volume=1_000_000)
    ee.put(Event(EVENT_BAR, bar))
    _drain(ee)


class TestLiveEngine:
    def test_order_flow_and_routing(self, setup):
        """下单 → 成交 → 回报正确路由回下单的策略"""
        engine, gw, rm, strategy = setup
        _push_bar(engine.event_engine, 10.0)

        vt_orderid = engine.send_order("S1", "000001.SZSE", Direction.LONG, 10.0, 1000)
        assert vt_orderid
        _drain(engine.event_engine)

        assert len(strategy.trades_received) == 1
        assert strategy.get_pos("000001.SZSE") == 1000

    def test_risk_blocks_order_gateway_untouched(self, setup):
        """风控拒单时，网关不应收到任何委托"""
        engine, gw, rm, strategy = setup
        _push_bar(engine.event_engine, 10.0)
        rm.activate_kill_switch("test")

        before = gw._order_count
        vt_orderid = engine.send_order("S1", "000001.SZSE", Direction.LONG, 10.0, 1000)
        assert vt_orderid == ""
        assert gw._order_count == before

    def test_disconnected_gateway_rejects(self, setup):
        """网关断开时直接拒绝下单，不依赖风控"""
        engine, gw, rm, strategy = setup
        gw.connected = False
        assert engine.send_order("S1", "000001.SZSE", Direction.LONG, 10.0, 1000) == ""

    def test_odd_lot_rounded_down(self, setup):
        """买入零股向下取整到 100 股"""
        engine, gw, rm, strategy = setup
        _push_bar(engine.event_engine, 10.0)

        engine.send_order("S1", "000001.SZSE", Direction.LONG, 10.0, 1550)
        _drain(engine.event_engine)
        assert strategy.trades_received[0].volume == 1500

        # 不足 100 股直接不下单
        assert engine.send_order("S1", "000001.SZSE", Direction.LONG, 10.0, 50) == ""

    def test_strategy_cannot_bypass_risk(self, setup):
        """策略对象上不应存在 gateway 引用"""
        engine, gw, rm, strategy = setup
        assert not hasattr(strategy, "gateway")
        assert strategy.engine is engine

    def test_account_updates_risk_manager(self, setup):
        engine, gw, rm, strategy = setup
        gw.query_account()
        _drain(engine.event_engine)
        assert engine.account is not None
        assert rm.account is not None
        assert engine.get_cash() == pytest.approx(1_000_000)

    def test_stop_all_sets_trading_false(self, setup):
        engine, gw, rm, strategy = setup
        engine.stop_all()
        assert not strategy.trading

    def test_duplicate_strategy_name_rejected(self, setup):
        engine, gw, rm, strategy = setup
        with pytest.raises(ValueError):
            engine.add_strategy(PassiveStrategy, "S1", ["000002.SZSE"])


class TestTradingTime:
    @pytest.mark.parametrize("dt_str,expected", [
        ("2024-01-02 09:35:00", True),
        ("2024-01-02 11:35:00", False),   # 午休
        ("2024-01-02 14:00:00", True),
        ("2024-01-02 15:30:00", False),   # 收盘后
        ("2024-01-06 10:00:00", False),   # 周六
    ])
    def test_is_trading_time(self, dt_str, expected):
        from datetime import datetime
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        assert LiveEngine._is_trading_time(dt) is expected
