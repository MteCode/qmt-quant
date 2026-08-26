"""核心模块单元测试。"""
import pytest

from qmtquant.config import CostConfig, RiskConfig
from qmtquant.core.constants import Direction, Exchange, RejectReason, Status
from qmtquant.core.objects import AccountData, OrderData, OrderRequest, PositionData
from qmtquant.gateway.sim_gateway import calc_cost
from qmtquant.risk.risk_manager import RiskManager
from qmtquant.utils.symbol import from_xt_symbol, normalize, to_xt_symbol


class TestSymbol:
    @pytest.mark.parametrize("raw,expected", [
        ("000001", "000001.SZSE"),
        ("000001.SZ", "000001.SZSE"),
        ("000001.SZSE", "000001.SZSE"),
        ("600519", "600519.SSE"),
        ("600519.SH", "600519.SSE"),
        ("688001", "688001.SSE"),
        ("830799", "830799.BSE"),
    ])
    def test_normalize(self, raw, expected):
        assert normalize(raw) == expected

    def test_roundtrip(self):
        assert to_xt_symbol("000001.SZSE") == "000001.SZ"
        assert from_xt_symbol("600519.SH") == "600519.SSE"
        assert from_xt_symbol(to_xt_symbol("000001.SZSE")) == "000001.SZSE"


class TestCost:
    def test_buy_no_stamp_tax(self):
        """买入不收印花税"""
        cost = CostConfig()
        fee = calc_cost(10.0, 10000, Direction.LONG, cost)
        assert fee == pytest.approx(100000 * 0.00025 + 100000 * 0.00001)

    def test_sell_has_stamp_tax(self):
        cost = CostConfig()
        fee = calc_cost(10.0, 10000, Direction.SHORT, cost)
        expected = 100000 * (0.00025 + 0.001 + 0.00001)
        assert fee == pytest.approx(expected)

    def test_minimum_commission(self):
        """小额成交按 5 元最低佣金收取"""
        cost = CostConfig()
        fee = calc_cost(1.0, 100, Direction.LONG, cost)
        assert fee >= cost.commission_min


class TestOrderData:
    def test_is_active(self):
        order = OrderData(status=Status.NOTTRADED)
        assert order.is_active()
        order.status = Status.ALLTRADED
        assert not order.is_active()
        order.status = Status.REJECTED
        assert not order.is_active()

    def test_vt_ids(self):
        order = OrderData(symbol="000001", exchange=Exchange.SZSE,
                          orderid="X1", gateway_name="SIM")
        assert order.vt_symbol == "000001.SZSE"
        assert order.vt_orderid == "SIM.X1"


def _make_risk(**overrides) -> RiskManager:
    cfg = RiskConfig(**overrides)
    rm = RiskManager(cfg)
    rm.update_account(AccountData(accountid="T", balance=1_000_000,
                                  available=1_000_000, market_value=0))
    return rm


def _buy_req(price=10.0, volume=1000, symbol="000001") -> OrderRequest:
    return OrderRequest(symbol=symbol, exchange=Exchange.SZSE,
                        direction=Direction.LONG, price=price, volume=volume)


class TestRiskManager:
    def test_normal_order_passes(self):
        rm = _make_risk()
        ok, reason = rm.check(_buy_req())
        assert ok and reason is None

    def test_kill_switch_blocks_all(self):
        rm = _make_risk()
        rm.activate_kill_switch("test")
        ok, reason = rm.check(_buy_req())
        assert not ok and reason is RejectReason.KILL_SWITCH

    def test_order_value_limit(self):
        rm = _make_risk(max_order_value=5000)
        ok, reason = rm.check(_buy_req(price=10, volume=1000))
        assert not ok and reason is RejectReason.ORDER_VALUE_LIMIT

    def test_odd_lot_buy_rejected(self):
        """买入必须 100 股整数倍"""
        rm = _make_risk()
        ok, reason = rm.check(_buy_req(volume=150))
        assert not ok and reason is RejectReason.INVALID_VOLUME

    def test_odd_lot_sell_allowed(self):
        """卖出允许零股（清仓场景）"""
        rm = _make_risk()
        rm.update_position(PositionData(symbol="000001", exchange=Exchange.SZSE,
                                        volume=150, frozen=0, yd_volume=150))
        req = OrderRequest(symbol="000001", exchange=Exchange.SZSE,
                           direction=Direction.SHORT, price=10, volume=150)
        ok, reason = rm.check(req)
        assert ok, reason

    def test_position_ratio_limit(self):
        # 放开单笔金额限制，隔离出占比这一项
        rm = _make_risk(max_position_ratio=0.05, max_order_value=10_000_000)
        ok, reason = rm.check(_buy_req(price=10, volume=10000))  # 10万 > 100万*5%
        assert not ok and reason is RejectReason.POSITION_RATIO_LIMIT

    def test_blacklist(self):
        rm = _make_risk(blacklist=["000001.SZSE"])
        ok, reason = rm.check(_buy_req())
        assert not ok and reason is RejectReason.BLACKLIST

    def test_insufficient_cash(self):
        rm = _make_risk(max_order_value=10_000_000)
        rm.update_account(AccountData(balance=1_000_000, available=5000, market_value=0))
        ok, reason = rm.check(_buy_req(price=10, volume=1000))
        assert not ok and reason is RejectReason.INSUFFICIENT_CASH

    def test_sell_without_position(self):
        rm = _make_risk()
        req = OrderRequest(symbol="000002", exchange=Exchange.SZSE,
                           direction=Direction.SHORT, price=10, volume=100)
        ok, reason = rm.check(req)
        assert not ok and reason is RejectReason.INSUFFICIENT_POSITION

    def test_t1_frozen_blocks_sell(self):
        """当日买入被冻结，不可卖出"""
        rm = _make_risk()
        rm.update_position(PositionData(symbol="000001", exchange=Exchange.SZSE,
                                        volume=1000, frozen=1000, yd_volume=0))
        req = OrderRequest(symbol="000001", exchange=Exchange.SZSE,
                           direction=Direction.SHORT, price=10, volume=1000)
        ok, reason = rm.check(req)
        assert not ok and reason is RejectReason.INSUFFICIENT_POSITION

    def test_order_count_limit(self):
        rm = _make_risk(max_order_count_per_day=2)
        assert rm.check(_buy_req())[0]
        assert rm.check(_buy_req())[0]
        ok, reason = rm.check(_buy_req())
        assert not ok and reason is RejectReason.ORDER_COUNT_LIMIT

    def test_daily_loss_triggers_close_only(self):
        """当日亏损触线后禁止买入，但允许卖出"""
        rm = _make_risk(daily_loss_limit_ratio=0.03)
        rm.update_account(AccountData(balance=960_000, available=500_000, market_value=0))
        assert rm.close_only

        ok, reason = rm.check(_buy_req())
        assert not ok and reason is RejectReason.DAILY_LOSS_LIMIT

        rm.update_position(PositionData(symbol="000001", exchange=Exchange.SZSE,
                                        volume=1000, frozen=0, yd_volume=1000))
        sell = OrderRequest(symbol="000001", exchange=Exchange.SZSE,
                            direction=Direction.SHORT, price=10, volume=1000)
        assert rm.check(sell)[0]

    def test_new_day_resets_counters(self):
        rm = _make_risk(max_order_count_per_day=1)
        rm.check(_buy_req())
        assert not rm.check(_buy_req())[0]
        rm.new_day(1_000_000)
        assert rm.check(_buy_req())[0]
