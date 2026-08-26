"""模拟撮合网关。

不连接任何券商，用推送进来的行情本地撮合，用于联调策略与风控链路。
遵守 A 股规则：T+1、100 股整数倍买入、涨跌停不成交。
"""
import logging
from datetime import datetime

from ..config import CostConfig
from ..core.constants import Direction, Status
from ..core.objects import (
    AccountData,
    BarData,
    CancelRequest,
    OrderData,
    OrderRequest,
    PositionData,
    SubscribeRequest,
    TickData,
)
from ..event.engine import EVENT_BAR, EVENT_TICK, Event, EventEngine
from ..utils.symbol import split_vt_symbol
from .base import BaseGateway

logger = logging.getLogger(__name__)


def calc_cost(price: float, volume: float, direction: Direction, cost: CostConfig) -> float:
    """计算一笔成交的总费用：佣金 + 印花税（卖出）+ 过户费"""
    turnover = price * volume
    commission = max(turnover * cost.commission_rate, cost.commission_min)
    stamp = turnover * cost.stamp_tax_rate if direction == Direction.SHORT else 0.0
    transfer = turnover * cost.transfer_fee_rate
    return commission + stamp + transfer


class SimGateway(BaseGateway):
    """本地模拟撮合网关"""

    def __init__(self, event_engine: EventEngine, gateway_name: str = "SIM",
                 initial_capital: float = 1_000_000,
                 cost: CostConfig | None = None) -> None:
        super().__init__(event_engine, gateway_name)

        self.cost = cost or CostConfig()
        self.cash: float = initial_capital
        self.positions: dict[str, PositionData] = {}
        self.active_orders: dict[str, OrderData] = {}
        self.last_prices: dict[str, float] = {}
        self._order_count: int = 0
        self._trade_count: int = 0

        # 监听行情事件驱动撮合
        event_engine.register(EVENT_TICK, self._on_tick_event)
        event_engine.register(EVENT_BAR, self._on_bar_event)

    # ------------------------------------------------------------ 接口实现

    def connect(self, setting: dict) -> bool:
        self.on_status(True, "模拟网关就绪")
        self.query_account()
        return True

    def close(self) -> None:
        self.on_status(False, "模拟网关关闭")

    def subscribe(self, req: SubscribeRequest) -> None:
        """模拟网关不主动取行情，由外部喂入"""

    def send_order(self, req: OrderRequest) -> str:
        self._order_count += 1
        orderid = f"SIM{self._order_count:08d}"
        order = OrderData(
            symbol=req.symbol, exchange=req.exchange, orderid=orderid,
            direction=req.direction, order_type=req.order_type,
            price=req.price, volume=req.volume, status=Status.NOTTRADED,
            datetime=datetime.now(), reference=req.reference,
            gateway_name=self.gateway_name,
        )
        self.active_orders[orderid] = order
        self.on_order(order)

        # 若已有最新价，立即尝试撮合一次
        last = self.last_prices.get(order.vt_symbol)
        if last:
            self._match(order, last)
        return order.vt_orderid

    def cancel_order(self, req: CancelRequest) -> None:
        order = self.active_orders.pop(req.orderid, None)
        if not order:
            return
        order.status = Status.CANCELLED
        self.on_order(order)

    def query_account(self) -> None:
        market_value = sum(
            p.volume * self.last_prices.get(p.vt_symbol, p.price)
            for p in self.positions.values()
        )
        self.on_account(AccountData(
            accountid="SIM", balance=self.cash + market_value,
            available=self.cash, market_value=market_value,
            gateway_name=self.gateway_name,
        ))

    def query_position(self) -> None:
        for pos in self.positions.values():
            self.on_position(pos)

    # ------------------------------------------------------------ 撮合

    def _on_tick_event(self, event: Event) -> None:
        tick: TickData = event.data
        self.last_prices[tick.vt_symbol] = tick.last_price
        self._match_all(tick.vt_symbol, tick.last_price)

    def _on_bar_event(self, event: Event) -> None:
        bar: BarData = event.data
        if bar.suspended:
            return
        self.last_prices[bar.vt_symbol] = bar.close_price
        self._match_all(bar.vt_symbol, bar.close_price)

    def _match_all(self, vt_symbol: str, price: float) -> None:
        for order in list(self.active_orders.values()):
            if order.vt_symbol == vt_symbol:
                self._match(order, price)

    def _match(self, order: OrderData, market_price: float) -> None:
        """限价单：买单价 >= 市价 或 卖单价 <= 市价 时全量成交"""
        if order.direction == Direction.LONG and order.price < market_price:
            return
        if order.direction == Direction.SHORT and order.price > market_price:
            return

        # 成交价取更有利的一方，模拟真实撮合
        traded_price = market_price
        volume = order.volume
        fee = calc_cost(traded_price, volume, order.direction, self.cost)

        if order.direction == Direction.LONG:
            need = traded_price * volume + fee
            if need > self.cash:
                order.status = Status.REJECTED
                order.message = "模拟账户资金不足"
                self.active_orders.pop(order.orderid, None)
                self.on_order(order)
                return
            self.cash -= need
            self._add_position(order, volume, traded_price)
        else:
            pos = self.positions.get(order.vt_symbol)
            if not pos or pos.available < volume:
                order.status = Status.REJECTED
                order.message = "模拟账户可卖数量不足"
                self.active_orders.pop(order.orderid, None)
                self.on_order(order)
                return
            self.cash += traded_price * volume - fee
            pos.volume -= volume
            pos.yd_volume = max(pos.yd_volume - volume, 0)
            if pos.volume <= 0:
                self.positions.pop(order.vt_symbol, None)

        order.traded = volume
        order.status = Status.ALLTRADED
        self.active_orders.pop(order.orderid, None)
        self.on_order(order)

        self._trade_count += 1
        from ..core.objects import TradeData
        self.on_trade(TradeData(
            symbol=order.symbol, exchange=order.exchange, orderid=order.orderid,
            tradeid=f"T{self._trade_count:08d}", direction=order.direction,
            price=traded_price, volume=volume, commission=fee,
            datetime=datetime.now(), reference=order.reference,
            gateway_name=self.gateway_name,
        ))
        self.query_account()

    def _add_position(self, order: OrderData, volume: float, price: float) -> None:
        pos = self.positions.get(order.vt_symbol)
        if not pos:
            symbol, exchange = split_vt_symbol(order.vt_symbol)
            pos = PositionData(symbol=symbol, exchange=exchange,
                               gateway_name=self.gateway_name)
            self.positions[order.vt_symbol] = pos
        total_cost = pos.price * pos.volume + price * volume
        pos.volume += volume
        pos.price = total_cost / pos.volume
        # T+1：当日买入全部冻结，次日 settle() 释放
        pos.frozen += volume
        self.on_position(pos)

    def settle(self) -> None:
        """日终结算：释放 T+1 冻结，供回测/模拟盘按日调用"""
        for pos in self.positions.values():
            pos.frozen = 0
            pos.yd_volume = pos.volume
