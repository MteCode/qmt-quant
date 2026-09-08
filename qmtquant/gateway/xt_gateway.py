"""miniQMT 交易网关。

通过 xtquant 连接 miniQMT 客户端，支持模拟盘和实盘。
使用前需先启动 miniQMT 并登录。
"""
import logging
import time
from datetime import datetime
from pathlib import Path

from ..config import CostConfig, GatewayConfig
from ..core.constants import Direction, Exchange, Offset, OrderType, Status
from ..core.objects import (
    AccountData,
    CancelRequest,
    ContractData,
    OrderData,
    OrderRequest,
    PositionData,
    SubscribeRequest,
    TickData,
    TradeData,
)
from ..event.engine import EventEngine
from .base import BaseGateway

logger = logging.getLogger(__name__)

EXCHANGE_MAP = {
    "SH": Exchange.SSE,
    "SZ": Exchange.SZSE,
}
EXCHANGE_MAP_INV = {v: k for k, v in EXCHANGE_MAP.items()}

DIRECTION_MAP = {
    Direction.LONG: 0,   # xtconstant.STOCK_BUY
    Direction.SHORT: 1,  # xtconstant.STOCK_SELL
}


def _to_vt_symbol(stock_code: str) -> tuple[str, Exchange]:
    """xtquant 代码 -> (symbol, Exchange)。如 '600000.SH' -> ('600000', Exchange.SSE)"""
    parts = stock_code.split(".")
    symbol = parts[0]
    ex_str = parts[1] if len(parts) > 1 else "SH"
    exchange = EXCHANGE_MAP.get(ex_str, Exchange.SSE)
    return symbol, exchange


def _to_xt_code(symbol: str, exchange: Exchange) -> str:
    """(symbol, Exchange) -> xtquant 代码。如 ('600000', Exchange.SSE) -> '600000.SH'"""
    ex_str = EXCHANGE_MAP_INV.get(exchange, "SH")
    return f"{symbol}.{ex_str}"


class XtGateway(BaseGateway):
    """miniQMT 网关"""

    default_setting = {
        "qmt_path": "D:/qmtApp/userdata_mini",
        "account_id": "",
        "account_type": "STOCK",
    }

    def __init__(self, event_engine: EventEngine,
                 gateway_name: str = "XtQuant") -> None:
        super().__init__(event_engine, gateway_name)

        self.trader = None
        self.account_id = ""
        self._order_count = 0

    def connect(self, setting: dict) -> bool:
        from xtquant import xttrader
        from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback

        qmt_path = setting.get("qmt_path", self.default_setting["qmt_path"])
        self.account_id = setting.get("account_id", "")
        session_id = int(time.time())

        if not Path(qmt_path).exists():
            logger.error("miniQMT 路径不存在: %s", qmt_path)
            self.on_status(False, f"路径不存在: {qmt_path}")
            return False

        class Callback(XtQuantTraderCallback):
            def __init__(self, gateway):
                super().__init__()
                self.gateway = gateway

            def on_disconnected(self):
                logger.warning("miniQMT 连接断开")
                self.gateway.on_status(False, "连接断开")

            def on_account_status(self, status):
                logger.info("账户状态: %s", status)

            def on_order_stock_async_response(self, response):
                pass

            def on_order_error(self, order_error):
                logger.error("委托错误: %s %s",
                             order_error.order_id, order_error.error_msg)

            # 方法名必须是 on_stock_order / on_stock_trade —— SDK 在
            # xttrader.py:302,308 调的是这两个名字。曾经写成
            # on_order_callback / on_trade_callback，因为本类继承了
            # XtQuantTraderCallback，SDK 命中的是基类的空实现，
            # 自定义回调成了永不执行的死代码：下完单没有任何委托或成交
            # 回流，系统不知道成交了没有、成交价多少、是不是废单。
            def on_stock_order(self, order_info):
                self.gateway._on_order(order_info)

            def on_stock_trade(self, trade_info):
                self.gateway._on_trade(trade_info)

        try:
            self.trader = XtQuantTrader(qmt_path, session_id)
            callback = Callback(self)
            self.trader.register_callback(callback)
            self.trader.start()

            result = self.trader.connect()
            if result != 0:
                logger.error("miniQMT 连接失败, code=%s", result)
                self.on_status(False, f"连接失败 code={result}")
                return False

            logger.info("miniQMT 连接成功: %s", self.account_id)
            self.on_status(True, "连接成功")
            return True

        except Exception as e:
            logger.error("miniQMT 连接异常: %s", e)
            self.on_status(False, str(e))
            return False

    def close(self) -> None:
        if self.trader:
            self.trader.stop()
            self.trader = None
        self.on_status(False, "已断开")

    def subscribe(self, req: SubscribeRequest) -> None:
        pass

    def send_order(self, req: OrderRequest) -> str:
        from xtquant.xttype import StockAccount

        if not self.trader:
            logger.error("未连接，无法下单")
            return ""

        from xtquant import xtconstant as c

        stock_code = _to_xt_code(req.symbol, req.exchange)
        account = StockAccount(self.account_id)

        order_type = (c.STOCK_BUY if req.direction == Direction.LONG
                      else c.STOCK_SELL)
        # SDK 签名是 (account, stock_code, order_type, order_volume,
        # price_type, price, strategy_name, order_remark) —— 曾经漏传
        # price_type，把 req.price 顶到了 price_type 的位置，price 无默认值，
        # 必抛 TypeError。之所以线上没炸，是因为没有任何代码调用这个方法：
        # 生产路径直接调 gateway.trader.order_stock 绕过了整层网关抽象。
        price_type = (c.FIX_PRICE if req.order_type == OrderType.LIMIT
                      else getattr(c, "LATEST_PRICE", c.FIX_PRICE))

        self._order_count += 1
        order_id = self.trader.order_stock(
            account, stock_code, order_type,
            int(req.volume), price_type, req.price,
            req.reference or "qmtquant",
            f"signal_{self._order_count}",
        )
        if order_id is None or order_id < 0:
            logger.error("报单失败 code=%s symbol=%s dir=%s vol=%s price=%.3f",
                         order_id, stock_code, req.direction.value,
                         req.volume, req.price)
            return ""

        vt_orderid = f"{self.gateway_name}.{order_id}"
        logger.info("委托已发送: %s %s %s %d股 %.2f",
                    vt_orderid, req.direction.value, stock_code,
                    req.volume, req.price)
        return vt_orderid

    def cancel_order(self, req: CancelRequest) -> None:
        if not self.trader:
            return
        from xtquant.xttype import StockAccount
        account = StockAccount(self.account_id)
        order_id = int(req.orderid.split(".")[-1])
        self.trader.cancel_order_stock(account, order_id)

    def query_account(self) -> None:
        if not self.trader:
            return
        from xtquant.xttype import StockAccount
        account = StockAccount(self.account_id)
        asset = self.trader.query_stock_asset(account)
        if asset:
            self.on_account(AccountData(
                accountid=self.account_id,
                balance=asset.total_asset,
                available=asset.cash,
                frozen=asset.frozen_cash,
            ))

    def query_position(self) -> None:
        if not self.trader:
            return
        from xtquant.xttype import StockAccount
        account = StockAccount(self.account_id)
        positions = self.trader.query_stock_positions(account)
        if not positions:
            return
        for pos in positions:
            if pos.volume <= 0:
                continue
            symbol, exchange = _to_vt_symbol(pos.stock_code)
            self.on_position(PositionData(
                symbol=symbol,
                exchange=exchange,
                volume=pos.volume,
                frozen=pos.volume - pos.can_use_volume,
                price=pos.avg_price,
                pnl=pos.market_value - pos.volume * pos.avg_price,
            ))

    def _on_order(self, order_info) -> None:
        symbol, exchange = _to_vt_symbol(order_info.stock_code)
        # 从 xtconstant 读常量而非硬编码。原先手写的表整体错位约 2：
        # 50(已报) 当成部成、54(已撤) 当成废单、55(部成) 当成全成，
        # 且完全没有 57(废单) —— 废单会 fallback 成 SUBMITTING，
        # 于是一笔被交易所拒掉的单，系统认为它还活着、is_active() 为真，
        # 撤单会去撤一笔不存在的单，风控也不知道这笔钱其实没花出去。
        from xtquant import xtconstant as c
        status_map = {
            c.ORDER_UNREPORTED: Status.SUBMITTING,       # 48 未报
            c.ORDER_WAIT_REPORTING: Status.SUBMITTING,   # 49 待报
            c.ORDER_REPORTED: Status.NOTTRADED,          # 50 已报
            c.ORDER_REPORTED_CANCEL: Status.NOTTRADED,   # 51 已报待撤
            c.ORDER_PARTSUCC_CANCEL: Status.PARTTRADED,  # 52 部成待撤
            c.ORDER_PART_CANCEL: Status.CANCELLED,       # 53 部撤
            c.ORDER_CANCELED: Status.CANCELLED,          # 54 已撤
            c.ORDER_PART_SUCC: Status.PARTTRADED,        # 55 部成
            c.ORDER_SUCCEEDED: Status.ALLTRADED,         # 56 已成
            c.ORDER_JUNK: Status.REJECTED,               # 57 废单
        }
        self.on_order(OrderData(
            symbol=symbol,
            exchange=exchange,
            orderid=f"{self.gateway_name}.{order_info.order_id}",
            direction=Direction.LONG if order_info.order_type == 23 else Direction.SHORT,
            price=order_info.price,
            volume=order_info.order_volume,
            traded=order_info.traded_volume,
            status=status_map.get(order_info.order_status, Status.SUBMITTING),
            datetime=datetime.now(),
        ))

    def _on_trade(self, trade_info) -> None:
        symbol, exchange = _to_vt_symbol(trade_info.stock_code)
        self.on_trade(TradeData(
            symbol=symbol,
            exchange=exchange,
            orderid=f"{self.gateway_name}.{trade_info.order_id}",
            tradeid=str(trade_info.traded_id),
            direction=Direction.LONG if trade_info.order_type == 23 else Direction.SHORT,
            price=trade_info.traded_price,
            volume=trade_info.traded_volume,
            datetime=datetime.now(),
        ))
