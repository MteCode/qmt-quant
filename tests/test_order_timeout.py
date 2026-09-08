"""委托超时撤单 + 报单失败时释放预留。

## 这些用例守什么

审计发现全仓找不到一处「挂单超过 N 秒未成交就撤」的逻辑，
LiveEngine 只在 stop_all()/daily_settle() 时 cancel_all。

不撤的后果不是「单还挂着」这么轻描淡写：限价单挂一天不成交，
目标持仓永远达不到，而系统不会告诉你 —— 调仓差异下次还这么算，
风控的在途预留也一直占着额度。

另外补一个预留泄漏：风控放行后网关仍可能报单失败，
这时预留不还回去，额度会被一笔根本不存在的委托永久占着。
"""
import time
from datetime import datetime

import pytest

from qmtquant.config import RiskConfig
from qmtquant.core.constants import (Direction, Exchange, OrderType, Status)
from qmtquant.core.objects import AccountData, OrderData
from qmtquant.engine.live_engine import LiveEngine
from qmtquant.event.engine import EventEngine
from qmtquant.gateway.base import BaseGateway
from qmtquant.risk.risk_manager import RiskManager


class _StubGateway(BaseGateway):
    def __init__(self, event_engine, send_ok=True):
        super().__init__(event_engine, "STUB")
        self.connected = True
        self.cancelled: list[str] = []
        self.send_ok = send_ok
        self.sent: list = []

    def connect(self, setting):
        return True

    def close(self):
        pass

    def subscribe(self, req):
        pass

    def send_order(self, req):
        self.sent.append(req)
        return "STUB.1" if self.send_ok else ""

    def cancel_order(self, req):
        self.cancelled.append(getattr(req, "orderid", req))

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
        max_order_value=10_000_000, max_position_ratio=1.0,
        max_total_position_ratio=1.0, max_order_count_per_day=200,
        max_turnover_per_day=1e9, daily_loss_limit_ratio=0.03,
        forbid_st=False, blacklist=[], drawdown_enabled=False,
        drawdown_close_only=0.06, drawdown_reduce=0.09,
        drawdown_reduce_keep=0.3, drawdown_flat=0.12,
        drawdown_recovery_ratio=0.7, drawdown_min_observations=20,
        drawdown_max_freeze=120,
    )


def _make(send_ok=True, timeout=300):
    ee = EventEngine()
    gw = _StubGateway(ee, send_ok=send_ok)
    rm = RiskManager(_cfg())
    eng = LiveEngine(ee, gw, rm, store=None, order_timeout=timeout)
    acct = AccountData(accountid="T", balance=1_000_000,
                       available=1_000_000, market_value=0, frozen=0)
    eng.account = acct
    rm.update_account(acct)
    return eng, gw, rm


def _active_order(orderid="STUB.1", volume=1000, traded=0,
                  status=Status.NOTTRADED):
    return OrderData(
        symbol="000001", exchange=Exchange.SZSE, orderid=orderid.split(".")[-1],
        direction=Direction.LONG, order_type=OrderType.LIMIT,
        price=10.0, volume=volume, traded=traded, status=status,
        datetime=datetime.now())


# ------------------------------------------------------------ 超时撤单

def test_挂太久的委托被撤():
    eng, gw, _ = _make(timeout=60)
    o = _active_order()
    eng.orders[o.vt_orderid] = o
    eng._order_submit_time[o.vt_orderid] = time.time() - 120
    eng._check_order_timeout()
    assert gw.cancelled == [o.orderid]


def test_未到超时不撤():
    eng, gw, _ = _make(timeout=300)
    o = _active_order()
    eng.orders[o.vt_orderid] = o
    eng._order_submit_time[o.vt_orderid] = time.time() - 10
    eng._check_order_timeout()
    assert gw.cancelled == []


def test_只撤一次():
    """每次定时器都重发撤单会刷屏，也可能被券商判为异常操作。"""
    eng, gw, _ = _make(timeout=60)
    o = _active_order()
    eng.orders[o.vt_orderid] = o
    eng._order_submit_time[o.vt_orderid] = time.time() - 120
    eng._check_order_timeout()
    eng._check_order_timeout()
    eng._check_order_timeout()
    assert gw.cancelled == [o.orderid]


def test_已成交的不撤():
    eng, gw, _ = _make(timeout=60)
    o = _active_order(traded=1000, status=Status.ALLTRADED)
    eng.orders[o.vt_orderid] = o
    eng._order_submit_time[o.vt_orderid] = time.time() - 120
    eng._check_order_timeout()
    assert gw.cancelled == []


def test_部分成交的也撤():
    """撤的是未成交余量，已成交部分不受影响。"""
    eng, gw, _ = _make(timeout=60)
    o = _active_order(volume=1000, traded=300, status=Status.PARTTRADED)
    eng.orders[o.vt_orderid] = o
    eng._order_submit_time[o.vt_orderid] = time.time() - 120
    eng._check_order_timeout()
    assert gw.cancelled == [o.orderid]


def test_超时设为0则不撤():
    eng, gw, _ = _make(timeout=0)
    o = _active_order()
    eng.orders[o.vt_orderid] = o
    eng._order_submit_time[o.vt_orderid] = time.time() - 100_000
    eng._check_order_timeout()
    assert gw.cancelled == []


def test_重启同步回来的单给完整超时窗口():
    """从券商同步回来的单没有本地报单时刻。当成 0 会在重启瞬间
    把所有在途单全撤掉 —— 那是灾难，不是保护。"""
    eng, gw, _ = _make(timeout=60)
    o = _active_order()
    eng.orders[o.vt_orderid] = o          # 没有 _order_submit_time
    eng._check_order_timeout()
    assert gw.cancelled == [], "重启瞬间不该撤单"
    assert o.vt_orderid in eng._order_submit_time, "应开始计时"


def test_成交后清理计时记录():
    """不清理的话字典会随交易日无限增长。"""
    eng, _, _ = _make(timeout=60)
    o = _active_order()
    eng.orders[o.vt_orderid] = o
    eng._order_submit_time[o.vt_orderid] = time.time()
    o.status = Status.ALLTRADED
    o.traded = o.volume
    eng._check_order_timeout()
    assert o.vt_orderid not in eng._order_submit_time


# ------------------------------------------------------------ 预留泄漏

def test_报单失败时释放预留():
    """风控放行后网关仍可能报单失败。预留不还回去，额度会被一笔
    根本不存在的委托永久占着，当天后续下单被逐渐挤死。"""
    eng, gw, rm = _make(send_ok=False)
    assert rm.pending_buy_value == pytest.approx(0)
    oid = eng.send_order("s", "000001.SZSE", Direction.LONG, 10.0, 1000)
    assert oid == ""
    assert rm.pending_buy_value == pytest.approx(0), "报单失败但预留没还"


def test_报单成功保留预留():
    eng, gw, rm = _make(send_ok=True)
    oid = eng.send_order("s", "000001.SZSE", Direction.LONG, 10.0, 1000)
    assert oid
    assert rm.pending_buy_value == pytest.approx(10_000)


def test_报单成功记录报单时刻():
    eng, _, _ = _make()
    oid = eng.send_order("s", "000001.SZSE", Direction.LONG, 10.0, 1000)
    assert oid in eng._order_submit_time


def test_连续报单失败不累积占用额度():
    """10 次失败下单后额度应该完好如初。"""
    eng, gw, rm = _make(send_ok=False)
    for _ in range(10):
        eng.send_order("s", "000001.SZSE", Direction.LONG, 10.0, 1000)
    assert rm.pending_buy_value == pytest.approx(0)
