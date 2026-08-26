"""交易网关抽象基类。

策略与引擎只依赖本接口，miniQMT / AMT / 模拟盘可自由替换。
新增网关只需实现 connect/send_order/cancel_order/query_* 五组方法。
"""
from abc import ABC, abstractmethod

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
from ..event.engine import (
    EVENT_ACCOUNT,
    EVENT_CONTRACT,
    EVENT_GATEWAY_STATUS,
    EVENT_ORDER,
    EVENT_POSITION,
    EVENT_TICK,
    EVENT_TRADE,
    Event,
    EventEngine,
)


class BaseGateway(ABC):
    """交易网关基类"""

    #: 连接所需配置字段说明，供上层校验
    default_setting: dict = {}

    def __init__(self, event_engine: EventEngine, gateway_name: str) -> None:
        self.event_engine = event_engine
        self.gateway_name = gateway_name
        self.connected: bool = False

    # ------------------------------------------------------------ 回调（子类调用）

    def _put(self, type_: str, data) -> None:
        data.gateway_name = self.gateway_name
        self.event_engine.put(Event(type_, data))

    def on_tick(self, tick: TickData) -> None:
        self._put(EVENT_TICK, tick)

    def on_order(self, order: OrderData) -> None:
        self._put(EVENT_ORDER, order)

    def on_trade(self, trade: TradeData) -> None:
        self._put(EVENT_TRADE, trade)

    def on_position(self, position: PositionData) -> None:
        self._put(EVENT_POSITION, position)

    def on_account(self, account: AccountData) -> None:
        self._put(EVENT_ACCOUNT, account)

    def on_contract(self, contract: ContractData) -> None:
        self._put(EVENT_CONTRACT, contract)

    def on_status(self, connected: bool, msg: str = "") -> None:
        """连接状态变化，上层据此决定是否暂停开仓"""
        self.connected = connected
        self.event_engine.put(
            Event(EVENT_GATEWAY_STATUS,
                  {"gateway": self.gateway_name, "connected": connected, "msg": msg})
        )

    # ------------------------------------------------------------ 抽象接口

    @abstractmethod
    def connect(self, setting: dict) -> bool:
        """建立连接，返回是否成功"""

    @abstractmethod
    def close(self) -> None:
        """断开连接并释放资源"""

    @abstractmethod
    def subscribe(self, req: SubscribeRequest) -> None:
        """订阅行情"""

    @abstractmethod
    def send_order(self, req: OrderRequest) -> str:
        """报单，返回 vt_orderid；失败返回空字符串"""

    @abstractmethod
    def cancel_order(self, req: CancelRequest) -> None:
        """撤单"""

    @abstractmethod
    def query_account(self) -> None:
        """查询资金，结果通过 on_account 推送"""

    @abstractmethod
    def query_position(self) -> None:
        """查询持仓，结果通过 on_position 推送"""

    def query_orders(self) -> list[OrderData]:
        """查询当日全部委托，用于断线重连后对账。默认不支持"""
        return []

    def query_trades(self) -> list[TradeData]:
        """查询当日全部成交，用于对账。默认不支持"""
        return []
