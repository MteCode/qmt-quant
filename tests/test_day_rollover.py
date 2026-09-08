"""引擎日切 —— 跨日重置风控额度、收盘撤单。

## 这些用例守的是什么

审计发现 `risk_manager.new_day()` 全仓没有任何调用者，`daily_settle()`
同样从未被调用，而 run_live.py 的主循环是 `while running: sleep(1)`，
跨日不做任何事。后果链条：

- `_order_count` 跨日累加 → 跑到第三天撞上 max_order_count_per_day，全天拒单
- `_day_start_balance` 停在进程启动那天 → 日亏 3% 线拿今天和三天前比
- `close_only` 一旦触发永不复位 → 之后每个交易日都只平不开
- 未成交挂单留到隔夜 → 券商已撤而本地仍认为它活着
"""
from datetime import date, datetime, timedelta

import pytest

from qmtquant.config import RiskConfig
from qmtquant.core.constants import Direction, Exchange, OrderType, Status
from qmtquant.core.objects import AccountData, OrderData, OrderRequest
from qmtquant.engine.live_engine import LiveEngine
from qmtquant.event.engine import EventEngine
from qmtquant.gateway.base import BaseGateway
from qmtquant.risk.risk_manager import RiskManager


class _StubGateway(BaseGateway):
    """只记录调用，不连任何外部系统。"""

    def __init__(self, event_engine):
        super().__init__(event_engine, "STUB")
        self.connected = True
        self.cancelled: list[str] = []

    def connect(self, setting):
        return True

    def close(self):
        pass

    def subscribe(self, req):
        pass

    def send_order(self, req):
        return "stub_1"

    def cancel_order(self, req):
        # 网关收到的是 CancelRequest，取 orderid 便于断言
        self.cancelled.append(getattr(req, 'orderid', req))

    def query_account(self):
        pass

    def query_position(self):
        pass

    def query_orders(self):
        return []

    def query_trades(self):
        return []


def _cfg():
    return RiskConfig(
        max_order_value=1_000_000, max_position_ratio=1.0,
        max_total_position_ratio=1.0, max_order_count_per_day=200,
        max_turnover_per_day=100_000_000, daily_loss_limit_ratio=0.03,
        forbid_st=False, blacklist=[], drawdown_enabled=False,
        drawdown_close_only=0.06, drawdown_reduce=0.09,
        drawdown_reduce_keep=0.3, drawdown_flat=0.12,
        drawdown_recovery_ratio=0.7, drawdown_min_observations=20,
        drawdown_max_freeze=120,
    )


@pytest.fixture
def engine():
    ee = EventEngine()
    gw = _StubGateway(ee)
    rm = RiskManager(_cfg())
    eng = LiveEngine(ee, gw, rm, store=None)
    eng.account = AccountData(accountid="T", balance=1_000_000,
                              available=1_000_000, market_value=0, frozen=0)
    rm.update_account(eng.account)
    yield eng, gw, rm


# ------------------------------------------------------------ 跨日重置

def test_跨日重置委托计数(engine):
    """计数跨日累加会导致第三天撞上限全天拒单。"""
    eng, _, rm = engine
    rm._order_count = 180
    eng._check_day_rollover(datetime.now() + timedelta(days=1))
    assert rm._order_count == 0


def test_跨日重置日初资产(engine):
    """日初资产不更新，日亏线就是拿今天和进程启动那天比。"""
    eng, _, rm = engine
    rm._day_start_balance = 2_000_000          # 陈旧值
    eng.account.balance = 950_000
    eng._check_day_rollover(datetime.now() + timedelta(days=1))
    assert rm._day_start_balance == pytest.approx(950_000)


def test_跨日复位只平不开(engine):
    """close_only 永不复位的话，之后每个交易日都不能开仓。"""
    eng, _, rm = engine
    rm.close_only = True
    eng._check_day_rollover(datetime.now() + timedelta(days=1))
    assert rm.close_only is False


def test_跨日清空在途预留(engine):
    """隔夜挂单已被券商清掉，预留带到第二天会误占额度。"""
    eng, _, rm = engine
    rm.check(OrderRequest(symbol="000001", exchange=Exchange.SZSE,
                          direction=Direction.LONG, order_type=OrderType.LIMIT,
                          volume=1000, price=10.0, reference="t"))
    assert rm.pending_buy_value > 0
    eng._check_day_rollover(datetime.now() + timedelta(days=1))
    assert rm.pending_buy_value == pytest.approx(0)


def test_同一天不重复重置(engine):
    """同日多次触发定时器不能反复清计数。"""
    eng, _, rm = engine
    rm._order_count = 42
    now = datetime.now().replace(hour=10, minute=0)
    eng._check_day_rollover(now)
    eng._check_day_rollover(now)
    assert rm._order_count == 42


# ------------------------------------------------------------ 收盘撤单

def test_收盘撤掉未成交挂单(engine):
    eng, gw, _ = engine
    eng.orders["o1"] = OrderData(
        symbol="000001", exchange=Exchange.SZSE, orderid="o1",
        direction=Direction.LONG, price=10.0, volume=1000, traded=0,
        status=Status.NOTTRADED)
    close_time = datetime.now().replace(hour=15, minute=1)
    if close_time.weekday() >= 5:                    # 用例不受周末影响
        close_time += timedelta(days=(7 - close_time.weekday()))
    eng._check_day_rollover(close_time)
    assert gw.cancelled == ["o1"]


def test_盘中不撤单(engine):
    eng, gw, _ = engine
    eng.orders["o1"] = OrderData(
        symbol="000001", exchange=Exchange.SZSE, orderid="o1",
        direction=Direction.LONG, price=10.0, volume=1000, traded=0,
        status=Status.NOTTRADED)
    eng._check_day_rollover(datetime.now().replace(hour=10, minute=30))
    assert gw.cancelled == []


def test_收盘只撤一次(engine):
    """重复触发会对同一笔单反复发撤单指令。"""
    eng, gw, _ = engine
    eng.orders["o1"] = OrderData(
        symbol="000001", exchange=Exchange.SZSE, orderid="o1",
        direction=Direction.LONG, price=10.0, volume=1000, traded=0,
        status=Status.NOTTRADED)
    t = datetime.now().replace(hour=15, minute=5)
    if t.weekday() >= 5:
        t += timedelta(days=(7 - t.weekday()))
    eng._check_day_rollover(t)
    eng._check_day_rollover(t)
    assert gw.cancelled == ["o1"]


def test_已成交的不撤(engine):
    eng, gw, _ = engine
    eng.orders["o1"] = OrderData(
        symbol="000001", exchange=Exchange.SZSE, orderid="o1",
        direction=Direction.LONG, price=10.0, volume=1000, traded=1000,
        status=Status.ALLTRADED)
    t = datetime.now().replace(hour=15, minute=5)
    if t.weekday() >= 5:
        t += timedelta(days=(7 - t.weekday()))
    eng._check_day_rollover(t)
    assert gw.cancelled == []


def test_周末不触发收盘处理(engine):
    eng, gw, _ = engine
    eng.orders["o1"] = OrderData(
        symbol="000001", exchange=Exchange.SZSE, orderid="o1",
        direction=Direction.LONG, price=10.0, volume=1000, traded=0,
        status=Status.NOTTRADED)
    sat = datetime.now()
    while sat.weekday() != 5:
        sat += timedelta(days=1)
    eng._check_day_rollover(sat.replace(hour=15, minute=5))
    assert gw.cancelled == []


def test_跨日后收盘标记复位(engine):
    """标记不复位的话，第二天收盘就不撤单了。"""
    eng, _, _ = engine
    eng._closed_today = True
    eng._check_day_rollover(datetime.now() + timedelta(days=1))
    assert eng._closed_today is False
