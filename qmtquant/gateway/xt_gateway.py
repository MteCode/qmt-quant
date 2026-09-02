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

            def on_order_callback(self, order_info):
                self.gateway._on_order(order_info)

            def on_trade_callback(self, trade_info):
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

        stock_code = _to_xt_code(req.symbol, req.exchange)
        account = StockAccount(self.account_id)

        if req.direction == Direction.LONG:
            order_type = 23  # STOCK_BUY
        else:
            order_type = 24  # STOCK_SELL

        self._order_count += 1
        order_id = self.trader.order_stock(
            account, stock_code, order_type,
            int(req.volume), req.price,
            strategy_name="ALSTM_PPO",
            order_remark=f"signal_{self._order_count}",
        )

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
        status_map = {
            48: Status.SUBMITTING,
            49: Status.NOTTRADED,
            50: Status.PARTTRADED,
            51: Status.PARTTRADED,
            52: Status.CANCELLED,
            53: Status.CANCELLED,
            54: Status.REJECTED,
            55: Status.ALLTRADED,
            56: Status.ALLTRADED,
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
