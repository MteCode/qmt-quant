"""领域对象定义。

所有跨模块传递的数据都用这里的 dataclass，禁止用裸 dict，
以便 IDE 补全与静态检查发现字段拼写错误。
"""
from dataclasses import dataclass, field
from datetime import datetime

from .constants import (
    ACTIVE_STATUSES,
    Direction,
    Exchange,
    Interval,
    Offset,
    OrderType,
    Product,
    Status,
)


@dataclass
class BaseData:
    """所有数据对象的基类，记录数据来源网关"""
    gateway_name: str = ""


@dataclass
class TickData(BaseData):
    """行情快照"""
    symbol: str = ""
    exchange: Exchange = Exchange.SZSE
    datetime: datetime = None

    name: str = ""
    volume: float = 0          # 累计成交量
    turnover: float = 0        # 累计成交额
    last_price: float = 0
    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    pre_close: float = 0

    limit_up: float = 0        # 涨停价
    limit_down: float = 0      # 跌停价

    bid_price_1: float = 0
    bid_volume_1: float = 0
    ask_price_1: float = 0
    ask_volume_1: float = 0
    # 完整五档，索引 0 对应一档
    bid_prices: list = field(default_factory=list)
    bid_volumes: list = field(default_factory=list)
    ask_prices: list = field(default_factory=list)
    ask_volumes: list = field(default_factory=list)

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"


@dataclass
class BarData(BaseData):
    """K 线"""
    symbol: str = ""
    exchange: Exchange = Exchange.SZSE
    datetime: datetime = None
    interval: Interval = Interval.DAILY

    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    close_price: float = 0
    volume: float = 0
    turnover: float = 0
    #: 是否停牌（停牌 Bar 不参与撮合）
    suspended: bool = False

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"


@dataclass
class OrderRequest:
    """下单请求（策略 → 引擎 → 风控 → 网关）"""
    symbol: str = ""
    exchange: Exchange = Exchange.SZSE
    direction: Direction = Direction.LONG
    order_type: OrderType = OrderType.LIMIT
    volume: float = 0
    price: float = 0
    offset: Offset = Offset.NONE
    reference: str = ""        # 归属策略名，用于回报路由

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"


@dataclass
class CancelRequest:
    """撤单请求"""
    orderid: str = ""
    symbol: str = ""
    exchange: Exchange = Exchange.SZSE

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"


@dataclass
class SubscribeRequest:
    """行情订阅请求"""
    symbol: str = ""
    exchange: Exchange = Exchange.SZSE

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"


@dataclass
class OrderData(BaseData):
    """委托回报"""
    symbol: str = ""
    exchange: Exchange = Exchange.SZSE
    orderid: str = ""
    direction: Direction = Direction.LONG
    order_type: OrderType = OrderType.LIMIT
    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    traded: float = 0
    status: Status = Status.SUBMITTING
    datetime: datetime = None
    reference: str = ""
    #: 拒单/撤单原因，便于排查
    message: str = ""

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"

    @property
    def vt_orderid(self) -> str:
        return f"{self.gateway_name}.{self.orderid}"

    def is_active(self) -> bool:
        """订单是否仍处于活动状态"""
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self) -> CancelRequest:
        return CancelRequest(
            orderid=self.orderid, symbol=self.symbol, exchange=self.exchange
        )


@dataclass
class TradeData(BaseData):
    """成交回报"""
    symbol: str = ""
    exchange: Exchange = Exchange.SZSE
    orderid: str = ""
    tradeid: str = ""
    direction: Direction = Direction.LONG
    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    commission: float = 0      # 本笔实际费用（佣金+印花税+过户费）
    datetime: datetime = None
    reference: str = ""

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"

    @property
    def vt_orderid(self) -> str:
        return f"{self.gateway_name}.{self.orderid}"

    @property
    def vt_tradeid(self) -> str:
        return f"{self.gateway_name}.{self.tradeid}"


@dataclass
class PositionData(BaseData):
    """持仓"""
    symbol: str = ""
    exchange: Exchange = Exchange.SZSE
    volume: float = 0          # 总持仓
    frozen: float = 0          # 冻结数量（T+1 当日买入 + 挂单卖出）
    price: float = 0           # 成本价
    pnl: float = 0             # 浮动盈亏
    yd_volume: float = 0       # 昨仓（可卖）

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"

    @property
    def available(self) -> float:
        """可卖数量"""
        return max(self.volume - self.frozen, 0)


@dataclass
class AccountData(BaseData):
    """账户资金"""
    accountid: str = ""
    balance: float = 0         # 总资产
    available: float = 0       # 可用资金
    market_value: float = 0    # 持仓市值
    frozen: float = 0          # 冻结资金

    @property
    def vt_accountid(self) -> str:
        return f"{self.gateway_name}.{self.accountid}"


@dataclass
class ContractData(BaseData):
    """合约/标的信息"""
    symbol: str = ""
    exchange: Exchange = Exchange.SZSE
    name: str = ""
    product: Product = Product.EQUITY
    pricetick: float = 0.01
    min_volume: float = 100    # 最小买入数量（A 股 100 股）
    is_st: bool = False

    @property
    def vt_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange.value}"


@dataclass
class LogData(BaseData):
    """日志事件"""
    msg: str = ""
    level: str = "INFO"
    datetime: datetime = field(default_factory=datetime.now)
