"""miniQMT 交易网关（基于 xtquant）。

依赖：miniQMT 或大 QMT 客户端已启动并登录，Python 环境中安装了 `xtquant`。

注意：xtquant 为券商闭源 SDK，不同版本枚举值与方法签名可能有差异。
本模块把所有 SDK 调用集中在这里，升级 SDK 时只需改这一个文件。
首次接入请先用 `scripts/check_env.py` 核对本机 SDK 版本的常量取值。
"""
import logging
import time
from datetime import datetime
from threading import Thread

from ..core.constants import Direction, OrderType, Status
from ..core.objects import (
    AccountData,
    CancelRequest,
    OrderData,
    OrderRequest,
    PositionData,
    SubscribeRequest,
    TickData,
    TradeData,
)
from ..event.engine import EventEngine
from ..utils.symbol import from_xt_symbol, split_vt_symbol, to_xt_symbol
from .base import BaseGateway

logger = logging.getLogger(__name__)


def _load_xt_constants():
    """从 xtconstant 读取枚举值。

    这些值曾经在不同版本间变动过（例如深市限价一度用 101，新版统一为
    FIX_PRICE=11），硬编码字面量会在 SDK 升级后静默失效 —— 报单返回 -1
    但看不出原因。因此一律从 SDK 读，读不到再退回已知默认值。
    """
    try:
        from xtquant import xtconstant as c
    except ImportError:
        return None

    return {
        "status": {
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
        },
        "direction": {Direction.LONG: c.STOCK_BUY, Direction.SHORT: c.STOCK_SELL},
        # FIX_PRICE 沪深通用；旧版按交易所区分的做法已废弃
        "limit": c.FIX_PRICE,
        "market": getattr(c, "LATEST_PRICE", c.FIX_PRICE),
    }


_XT = _load_xt_constants()

#: xtquant 委托状态 → 内部状态
STATUS_XT2VT: dict[int, Status] = (_XT or {}).get("status") or {
    48: Status.SUBMITTING, 49: Status.SUBMITTING, 50: Status.NOTTRADED,
    51: Status.NOTTRADED, 52: Status.PARTTRADED, 53: Status.CANCELLED,
    54: Status.CANCELLED, 55: Status.PARTTRADED, 56: Status.ALLTRADED,
    57: Status.REJECTED,
}

#: 内部方向 → xtquant 委托类型
DIRECTION_VT2XT = (_XT or {}).get("direction") or {
    Direction.LONG: 23, Direction.SHORT: 24,
}
DIRECTION_XT2VT = {v: k for k, v in DIRECTION_VT2XT.items()}

#: 限价 / 市价的报价类型
PRICE_TYPE_LIMIT_VALUE = (_XT or {}).get("limit", 11)
PRICE_TYPE_MARKET_VALUE = (_XT or {}).get("market", 5)


class MiniQmtGateway(BaseGateway):
    """miniQMT 网关"""

    default_setting = {
        "qmt_path": "",      # 形如 D:\\国金QMT交易端\\userdata_mini
        "account_id": "",
        "account_type": "STOCK",
    }

    def __init__(self, event_engine: EventEngine, gateway_name: str = "MINIQMT") -> None:
        super().__init__(event_engine, gateway_name)

        self._trader = None          # XtQuantTrader
        self._account = None         # StockAccount
        self._setting: dict = {}

        # 本地订单簿：内部 orderid -> OrderData
        self._orders: dict[str, OrderData] = {}
        # 内部 orderid -> 券商返回的 order_id，撤单需要
        self._orderid_map: dict[str, int] = {}
        self._local_id: int = 0

        self._subscribed: set[str] = set()
        self._reconnecting: bool = False

    # ------------------------------------------------------------ 连接

    def connect(self, setting: dict) -> bool:
        self._setting = setting
        try:
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
        except ImportError:
            logger.error("未安装 xtquant，无法使用 miniQMT 网关；请从 QMT 客户端安装目录获取")
            self.on_status(False, "xtquant 未安装")
            return False

        path = setting["qmt_path"]
        # session_id 必须每次不同，复用会导致连接被拒
        session_id = int(time.time())

        self._trader = XtQuantTrader(path, session_id)
        self._trader.register_callback(_TraderCallback(self))
        self._trader.start()

        if self._trader.connect() != 0:
            logger.error("miniQMT 连接失败，请确认客户端已登录且路径正确: %s", path)
            self.on_status(False, "连接交易端失败")
            return False

        self._account = StockAccount(setting["account_id"], setting.get("account_type", "STOCK"))
        if self._trader.subscribe(self._account) != 0:
            logger.error("订阅账户回报失败: %s", setting["account_id"])
            self.on_status(False, "订阅账户失败")
            return False

        self.on_status(True, "miniQMT 已连接")
        logger.info("miniQMT 已连接，账户=%s", setting["account_id"])

        # 连接成功后立即对账，避免本地状态与券商不一致
        self.query_account()
        self.query_position()
        self._sync_orders()
        return True

    def close(self) -> None:
        if self._trader:
            try:
                self._trader.stop()
            except Exception:
                logger.exception("关闭 miniQMT 连接异常")
        self.on_status(False, "已主动断开")

    def _reconnect(self) -> None:
        """指数退避重连。重连期间 connected=False，引擎据此禁止开仓。"""
        if self._reconnecting:
            return
        self._reconnecting = True

        def _worker() -> None:
            max_retry = self._setting.get("reconnect_max_retry", 10)
            base = self._setting.get("reconnect_base_delay", 2.0)
            for i in range(max_retry):
                delay = min(base * (2 ** i), 60)
                logger.warning("miniQMT 断线，%.0fs 后第 %d 次重连", delay, i + 1)
                time.sleep(delay)
                try:
                    if self.connect(self._setting):
                        logger.info("miniQMT 重连成功")
                        break
                except Exception:
                    logger.exception("重连异常")
            else:
                logger.error("miniQMT 重连失败已达上限，进入只读状态，请人工介入")
            self._reconnecting = False

        Thread(target=_worker, name="QmtReconnect", daemon=True).start()

    # ------------------------------------------------------------ 行情

    def subscribe(self, req: SubscribeRequest) -> None:
        try:
            from xtquant import xtdata
        except ImportError:
            return

        xt_symbol = to_xt_symbol(req.vt_symbol)
        if xt_symbol in self._subscribed:
            return

        xtdata.subscribe_quote(xt_symbol, period="tick", callback=self._on_xt_tick)
        self._subscribed.add(xt_symbol)
        logger.info("已订阅行情: %s", req.vt_symbol)

    def _on_xt_tick(self, datas: dict) -> None:
        """xtdata tick 回调，datas 形如 {'000001.SZ': [ {...} ]}"""
        for xt_symbol, items in datas.items():
            for d in items:
                try:
                    self.on_tick(self._convert_tick(xt_symbol, d))
                except Exception:
                    logger.exception("解析 tick 失败: %s", xt_symbol)

    def _convert_tick(self, xt_symbol: str, d: dict) -> TickData:
        symbol, exchange = split_vt_symbol(from_xt_symbol(xt_symbol))
        bid_p = list(d.get("bidPrice", []) or [])
        bid_v = list(d.get("bidVol", []) or [])
        ask_p = list(d.get("askPrice", []) or [])
        ask_v = list(d.get("askVol", []) or [])
        return TickData(
            symbol=symbol,
            exchange=exchange,
            # xtdata 的 time 字段是毫秒时间戳
            datetime=datetime.fromtimestamp(d["time"] / 1000),
            last_price=d.get("lastPrice", 0),
            open_price=d.get("open", 0),
            high_price=d.get("high", 0),
            low_price=d.get("low", 0),
            pre_close=d.get("lastClose", 0),
            volume=d.get("volume", 0),
            turnover=d.get("amount", 0),
            bid_price_1=bid_p[0] if bid_p else 0,
            bid_volume_1=bid_v[0] if bid_v else 0,
            ask_price_1=ask_p[0] if ask_p else 0,
            ask_volume_1=ask_v[0] if ask_v else 0,
            bid_prices=bid_p,
            bid_volumes=bid_v,
            ask_prices=ask_p,
            ask_volumes=ask_v,
            gateway_name=self.gateway_name,
        )

    # ------------------------------------------------------------ 交易

    def _new_orderid(self) -> str:
        self._local_id += 1
        return f"{datetime.now():%Y%m%d}_{self._local_id:06d}"

    def send_order(self, req: OrderRequest) -> str:
        if not self.connected or not self._trader:
            logger.error("网关未连接，拒绝报单: %s", req.vt_symbol)
            return ""

        orderid = self._new_orderid()
        order = OrderData(
            symbol=req.symbol, exchange=req.exchange, orderid=orderid,
            direction=req.direction, order_type=req.order_type,
            price=req.price, volume=req.volume, status=Status.SUBMITTING,
            datetime=datetime.now(), reference=req.reference,
            gateway_name=self.gateway_name,
        )
        self._orders[orderid] = order

        price_type = (PRICE_TYPE_LIMIT_VALUE if req.order_type == OrderType.LIMIT
                      else PRICE_TYPE_MARKET_VALUE)

        try:
            broker_id = self._trader.order_stock(
                self._account,
                to_xt_symbol(req.vt_symbol),
                DIRECTION_VT2XT[req.direction],
                int(req.volume),
                price_type,
                req.price,
                req.reference or "qmtquant",
                orderid,          # strategy_name / order_remark，用于回报关联
            )
        except Exception:
            logger.exception("报单调用异常: %s", req.vt_symbol)
            broker_id = -1

        if broker_id is None or broker_id < 0:
            order.status = Status.REJECTED
            # 带上返回码与入参：xtquant 报单失败只返回负数，不给原因，
            # 不记下这些的话事后完全无从排查
            order.message = (f"交易端拒绝 code={broker_id} "
                             f"price_type={price_type} price={req.price} "
                             f"volume={req.volume}")
            logger.error("报单被拒 %s: %s。常见原因：非交易时段、"
                         "无量化交易权限、价格超出涨跌停、资金/持仓不足",
                         req.vt_symbol, order.message)
            self.on_order(order)
            return ""

        self._orderid_map[orderid] = broker_id
        self.on_order(order)
        return order.vt_orderid

    def cancel_order(self, req: CancelRequest) -> None:
        broker_id = self._orderid_map.get(req.orderid)
        if broker_id is None:
            logger.warning("撤单失败，找不到券商订单号: %s", req.orderid)
            return
        try:
            self._trader.cancel_order_stock(self._account, broker_id)
        except Exception:
            logger.exception("撤单调用异常: %s", req.orderid)

    # ------------------------------------------------------------ 查询

    def query_account(self) -> None:
        if not self._trader:
            return
        asset = self._trader.query_stock_asset(self._account)
        if not asset:
            return
        self.on_account(AccountData(
            accountid=self._setting.get("account_id", ""),
            balance=asset.total_asset,
            available=asset.cash,
            market_value=asset.market_value,
            frozen=asset.frozen_cash,
            gateway_name=self.gateway_name,
        ))

    def query_position(self) -> None:
        if not self._trader:
            return
        for p in self._trader.query_stock_positions(self._account) or []:
            symbol, exchange = split_vt_symbol(from_xt_symbol(p.stock_code))
            self.on_position(PositionData(
                symbol=symbol, exchange=exchange,
                volume=p.volume,
                frozen=p.volume - p.can_use_volume,
                yd_volume=p.can_use_volume,
                price=p.open_price,
                pnl=(p.market_value - p.open_price * p.volume),
                gateway_name=self.gateway_name,
            ))

    def query_orders(self) -> list[OrderData]:
        """查询当日全部委托，供 LiveEngine.reconcile() 对账。

        必须实现：基类默认返回空列表，不覆写的话对账会永远报告
        「活动委托 0 笔」—— 实盘重启时看不到已挂的单，可能重复下单。
        """
        if not self._trader:
            return []

        orders = []
        for o in self._trader.query_stock_orders(self._account) or []:
            try:
                orders.append(self._convert_order(o))
            except Exception:
                logger.exception("解析委托失败: order_id=%s",
                                 getattr(o, "order_id", "?"))
        return orders

    def query_trades(self) -> list[TradeData]:
        """查询当日全部成交，供对账使用"""
        if not self._trader:
            return []

        trades = []
        for t in self._trader.query_stock_trades(self._account) or []:
            try:
                trades.append(self._convert_trade(t))
            except Exception:
                logger.exception("解析成交失败: traded_id=%s",
                                 getattr(t, "traded_id", "?"))
        return trades

    def _sync_orders(self) -> None:
        """启动/重连时把当日委托全部推入事件引擎，重建本地订单簿"""
        for order in self.query_orders():
            self.on_order(order)

    def _convert_order(self, o) -> OrderData:
        symbol, exchange = split_vt_symbol(from_xt_symbol(o.stock_code))
        # order_remark 里放的是本地 orderid，取不到就退回券商单号
        orderid = getattr(o, "order_remark", "") or str(o.order_id)
        self._orderid_map.setdefault(orderid, o.order_id)
        return OrderData(
            symbol=symbol, exchange=exchange, orderid=orderid,
            direction=DIRECTION_XT2VT.get(o.order_type, Direction.LONG),
            price=o.price, volume=o.order_volume, traded=o.traded_volume,
            status=STATUS_XT2VT.get(o.order_status, Status.SUBMITTING),
            datetime=datetime.fromtimestamp(o.order_time) if o.order_time else datetime.now(),
            reference=getattr(o, "strategy_name", ""),
            gateway_name=self.gateway_name,
        )

    def _convert_trade(self, t) -> TradeData:
        symbol, exchange = split_vt_symbol(from_xt_symbol(t.stock_code))
        orderid = getattr(t, "order_remark", "") or str(t.order_id)
        return TradeData(
            symbol=symbol, exchange=exchange, orderid=orderid,
            tradeid=str(t.traded_id),
            direction=DIRECTION_XT2VT.get(t.order_type, Direction.LONG),
            price=t.traded_price, volume=t.traded_volume,
            datetime=datetime.fromtimestamp(t.traded_time) if t.traded_time else datetime.now(),
            reference=getattr(t, "strategy_name", ""),
            gateway_name=self.gateway_name,
        )


class _TraderCallback:
    """xtquant 异步回报回调。

    继承 XtQuantTraderCallback 的方法名由 SDK 约定，不可改。
    所有回调内必须捕获异常，否则会打断 SDK 的回调线程。
    """

    def __init__(self, gateway: MiniQmtGateway) -> None:
        self.gateway = gateway

    def on_disconnected(self) -> None:
        logger.error("miniQMT 连接断开")
        self.gateway.on_status(False, "连接断开")
        self.gateway._reconnect()

    def on_stock_order(self, order) -> None:
        try:
            self.gateway.on_order(self.gateway._convert_order(order))
        except Exception:
            logger.exception("处理委托回报异常")

    def on_stock_trade(self, trade) -> None:
        try:
            self.gateway.on_trade(self.gateway._convert_trade(trade))
        except Exception:
            logger.exception("处理成交回报异常")

    def on_order_error(self, err) -> None:
        try:
            orderid = getattr(err, "order_remark", "") or str(getattr(err, "order_id", ""))
            order = self.gateway._orders.get(orderid)
            if order:
                order.status = Status.REJECTED
                order.message = getattr(err, "error_msg", "报单错误")
                self.gateway.on_order(order)
            logger.error("报单错误: id=%s msg=%s", orderid, getattr(err, "error_msg", ""))
        except Exception:
            logger.exception("处理报单错误回报异常")

    def on_cancel_error(self, err) -> None:
        logger.error("撤单错误: %s", getattr(err, "error_msg", err))

    def on_account_status(self, status) -> None:
        logger.info("账户状态变更: %s", getattr(status, "status", status))
