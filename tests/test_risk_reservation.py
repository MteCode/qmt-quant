"""风控在途预留 —— 防止批量下单击穿额度。

## 这些用例守的是什么

审计发现：`check()` 放行后只加计数，不动账户快照。批量调仓时在循环里
连发 N 笔买单，每一笔都拿同一份没变过的快照去比 —— 50 万可用资金，
10 笔各 45 万的买单会全部放行，因为每笔单独看都「不超过可用资金」。

单票占比、总仓位上限同理，因为它们读的也是静态的
`account.market_value` 和 `positions`。

修复方式是放行即记预留，后续校验扣掉在途量。下面每个用例对应一条
曾经能被击穿的约束。
"""
import pytest

from qmtquant.config import RiskConfig
from qmtquant.core.constants import (Direction, Exchange, OrderType,
                                     RejectReason, Status)
from qmtquant.core.objects import (AccountData, OrderData, OrderRequest,
                                   PositionData, TradeData)
from qmtquant.risk.risk_manager import RiskManager


def _cfg(**kw):
    base = dict(
        max_order_value=1_000_000,
        max_position_ratio=0.2,
        max_total_position_ratio=0.95,
        max_order_count_per_day=200,
        max_turnover_per_day=100_000_000,
        daily_loss_limit_ratio=0.03,
        forbid_st=False,
        blacklist=[],
        drawdown_enabled=False,
        drawdown_close_only=0.06,
        drawdown_reduce=0.09,
        drawdown_reduce_keep=0.3,
        drawdown_flat=0.12,
        drawdown_recovery_ratio=0.7,
        drawdown_min_observations=20,
        drawdown_max_freeze=120,
    )
    base.update(kw)
    return RiskConfig(**base)


def _rm(available=500_000, balance=1_000_000, market_value=0, **kw):
    rm = RiskManager(_cfg(**kw))
    rm.update_account(AccountData(
        accountid="T", balance=balance, available=available,
        market_value=market_value, frozen=0))
    return rm


def _buy(symbol="000001", price=10.0, volume=1000):
    return OrderRequest(symbol=symbol, exchange=Exchange.SZSE,
                        direction=Direction.LONG, order_type=OrderType.LIMIT,
                        volume=volume, price=price, reference="t")


def _sell(symbol="000001", price=10.0, volume=1000):
    return OrderRequest(symbol=symbol, exchange=Exchange.SZSE,
                        direction=Direction.SHORT, order_type=OrderType.LIMIT,
                        volume=volume, price=price, reference="t")


# ------------------------------------------------------------ 可用资金

# 单票/总仓位上限放开，让用例只考验「可用资金」这一条约束
_CASH_ONLY = dict(max_position_ratio=1.0, max_total_position_ratio=1.0)


def test_批量买单不能击穿可用资金():
    """50 万可用，连发 45 万的单：第一笔过，第二笔必须被拒。

    这正是审计发现的击穿场景 —— 修复前两笔都会放行。
    """
    rm = _rm(available=500_000, **_CASH_ONLY)
    ok1, _ = rm.check(_buy(price=450.0, volume=1000))    # 45 万
    ok2, reason = rm.check(_buy(symbol="000002", price=450.0, volume=1000))
    assert ok1 is True
    assert ok2 is False
    assert reason == RejectReason.INSUFFICIENT_CASH


def test_累计到刚好用完额度():
    """10 万可用，5 笔 2 万：全过；第 6 笔被拒。"""
    rm = _rm(available=100_000, **_CASH_ONLY)
    for i in range(5):
        ok, _ = rm.check(_buy(symbol=f"00000{i}", price=20.0, volume=1000))
        assert ok is True, f"第 {i+1} 笔不该被拒"
    ok, reason = rm.check(_buy(symbol="000009", price=20.0, volume=1000))
    assert ok is False
    assert reason == RejectReason.INSUFFICIENT_CASH


# ------------------------------------------------------------ 单票占比

def test_同一标的多笔买单不能击穿单票占比():
    """单票上限 20%，总资产 100 万 → 单票最多 20 万。
    连发 3 笔 8 万：前两笔过（16 万），第三笔会到 24 万，必须拒。"""
    rm = _rm(available=1_000_000, balance=1_000_000)
    ok1, _ = rm.check(_buy(price=80.0, volume=1000))
    ok2, _ = rm.check(_buy(price=80.0, volume=1000))
    ok3, reason = rm.check(_buy(price=80.0, volume=1000))
    assert (ok1, ok2) == (True, True)
    assert ok3 is False
    assert reason == RejectReason.POSITION_RATIO_LIMIT


# ------------------------------------------------------------ 总仓位

def test_批量买单不能击穿总仓位上限():
    """总仓位上限 95%，已有 90 万市值，总资产 100 万 →
    还能买 5 万。连发 2 笔 4 万，第二笔必须拒。"""
    rm = _rm(available=1_000_000, balance=1_000_000, market_value=900_000)
    ok1, _ = rm.check(_buy(price=40.0, volume=1000))
    ok2, reason = rm.check(_buy(symbol="000002", price=40.0, volume=1000))
    assert ok1 is True
    assert ok2 is False
    assert reason == RejectReason.TOTAL_POSITION_LIMIT


# ------------------------------------------------------------ 卖出

def test_重复卖单不能超过可卖数量():
    """持仓 1000 股可卖，连发两笔卖 600：第二笔必须拒。"""
    rm = _rm()
    rm.update_position(PositionData(
        symbol="000001", exchange=Exchange.SZSE,
        volume=1000, frozen=0, yd_volume=1000, price=10.0))
    ok1, _ = rm.check(_sell(volume=600))
    ok2, reason = rm.check(_sell(volume=600))
    assert ok1 is True
    assert ok2 is False
    assert reason == RejectReason.INSUFFICIENT_POSITION


# ------------------------------------------------------------ 预留释放

def test_成交后释放预留():
    """买单成交即落地，预留该还回去 —— 否则额度会一直被占着。"""
    rm = _rm(available=100_000)
    rm.check(_buy(price=100.0, volume=1000))            # 预留 10 万
    assert rm.pending_buy_value == pytest.approx(100_000)

    rm.on_trade(TradeData(
        symbol="000001", exchange=Exchange.SZSE, orderid="1", tradeid="1",
        direction=Direction.LONG, price=100.0, volume=1000))
    assert rm.pending_buy_value == pytest.approx(0)


def test_部分成交只释放成交部分():
    rm = _rm(available=100_000)
    rm.check(_buy(price=100.0, volume=1000))            # 预留 10 万
    rm.on_trade(TradeData(
        symbol="000001", exchange=Exchange.SZSE, orderid="1", tradeid="1",
        direction=Direction.LONG, price=100.0, volume=400))
    assert rm.pending_buy_value == pytest.approx(60_000)


def test_撤单释放未成交部分():
    """撤单后额度必须还回来，否则当天后续调仓会被误拒。"""
    rm = _rm(available=100_000)
    rm.check(_buy(price=100.0, volume=1000))
    rm.on_order(OrderData(
        symbol="000001", exchange=Exchange.SZSE, orderid="1",
        direction=Direction.LONG, price=100.0, volume=1000, traded=0,
        status=Status.CANCELLED))
    assert rm.pending_buy_value == pytest.approx(0)


def test_废单释放预留():
    rm = _rm(available=100_000)
    rm.check(_buy(price=100.0, volume=1000))
    rm.on_order(OrderData(
        symbol="000001", exchange=Exchange.SZSE, orderid="1",
        direction=Direction.LONG, price=100.0, volume=1000, traded=0,
        status=Status.REJECTED))
    assert rm.pending_buy_value == pytest.approx(0)


def test_部成后撤单只释放剩余部分():
    rm = _rm(available=100_000)
    rm.check(_buy(price=100.0, volume=1000))            # 预留 10 万
    rm.on_trade(TradeData(
        symbol="000001", exchange=Exchange.SZSE, orderid="1", tradeid="1",
        direction=Direction.LONG, price=100.0, volume=300))   # 成交 3 万
    rm.on_order(OrderData(
        symbol="000001", exchange=Exchange.SZSE, orderid="1",
        direction=Direction.LONG, price=100.0, volume=1000, traded=300,
        status=Status.CANCELLED))                             # 撤掉 7 万
    assert rm.pending_buy_value == pytest.approx(0)


def test_活动委托不释放预留():
    """已报/部成属于在途，额度不能提前还。"""
    rm = _rm(available=100_000)
    rm.check(_buy(price=100.0, volume=1000))
    rm.on_order(OrderData(
        symbol="000001", exchange=Exchange.SZSE, orderid="1",
        direction=Direction.LONG, price=100.0, volume=1000, traded=0,
        status=Status.NOTTRADED))
    assert rm.pending_buy_value == pytest.approx(100_000)


def test_日切清空预留():
    """隔夜挂单已被券商清掉，预留不能带到第二天。"""
    rm = _rm(available=100_000)
    rm.check(_buy(price=100.0, volume=1000))
    assert rm.pending_buy_value > 0
    rm.new_day(1_000_000)
    assert rm.pending_buy_value == pytest.approx(0)


# ------------------------------------------------------------ 回归

def test_拒单不占预留():
    """被拒的单没发出去，不该占额度。"""
    rm = _rm(available=10_000)
    ok, _ = rm.check(_buy(price=100.0, volume=1000))    # 10 万 > 1 万
    assert ok is False
    assert rm.pending_buy_value == pytest.approx(0)


def test_单笔仍按原上限校验():
    """预留机制不能放松单笔金额上限。"""
    rm = _rm(available=1_000_000, max_order_value=50_000)
    ok, reason = rm.check(_buy(price=100.0, volume=1000))   # 10 万
    assert ok is False
    assert reason == RejectReason.ORDER_VALUE_LIMIT
