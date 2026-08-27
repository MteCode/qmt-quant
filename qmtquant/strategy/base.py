"""策略基类。

同一份策略代码在回测与实盘下行为完全一致 —— 关键在于策略只调用
`self.buy/sell/cancel`，由注入的 engine 决定这些调用落到模拟撮合还是真实券商。
"""
import logging
from abc import ABC
from typing import Any

from ..core.constants import Direction, OrderType
from ..core.objects import BarData, OrderData, TickData, TradeData

logger = logging.getLogger(__name__)


class StrategyBase(ABC):
    """CTA/选股策略基类"""

    #: 子类覆写：可优化的参数名列表，用于参数寻优与配置注入
    parameters: list[str] = []
    #: 子类覆写：需要持久化的运行时变量名列表
    variables: list[str] = ["inited", "trading", "pos"]

    def __init__(self, engine: Any, strategy_name: str,
                 vt_symbols: list[str], setting: dict | None = None) -> None:
        self.engine = engine
        self.strategy_name = strategy_name
        self.vt_symbols = vt_symbols

        self.inited: bool = False
        self.trading: bool = False
        #: 各标的持仓数量（策略视角，非账户视角）
        self.pos: dict[str, float] = {s: 0 for s in vt_symbols}

        self.update_setting(setting or {})

    # ------------------------------------------------------------ 参数

    def update_setting(self, setting: dict) -> None:
        """从配置注入参数，只接受 `parameters` 中声明过的字段"""
        for name in self.parameters:
            if name in setting:
                setattr(self, name, setting[name])

    def get_parameters(self) -> dict:
        return {n: getattr(self, n, None) for n in self.parameters}

    def get_variables(self) -> dict:
        return {n: getattr(self, n, None) for n in self.variables}

    def restore_variables(self, data: dict) -> None:
        """从持久化状态恢复运行时变量。

        只恢复 `variables` 中声明过的字段，且**不恢复 inited/trading** ——
        这两个是生命周期标志，必须由引擎按当前实际情况设置，
        从磁盘读一个 trading=True 会让策略在未初始化时就开始交易。
        """
        skip = {"inited", "trading"}
        for name in self.variables:
            if name in skip or name not in data:
                continue
            setattr(self, name, data[name])
        self.write_log(f"已恢复运行时状态：{sorted(set(self.variables) - skip)}")

    # ------------------------------------------------------------ 生命周期回调（子类覆写）

    def on_init(self) -> None:
        """初始化：加载历史数据、预热指标"""

    def on_start(self) -> None:
        """开始交易"""

    def on_stop(self) -> None:
        """停止交易"""

    def on_tick(self, tick: TickData) -> None:
        """行情快照推送"""

    def on_bar(self, bar: BarData) -> None:
        """单标的 K 线推送"""

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """多标的同一时间截面的 K 线推送（选股型策略用）"""

    def on_order(self, order: OrderData) -> None:
        """委托状态更新"""

    def on_trade(self, trade: TradeData) -> None:
        """成交回报。基类已更新 self.pos，子类覆写时记得调用 super()。"""
        delta = trade.volume if trade.direction == Direction.LONG else -trade.volume
        self.pos[trade.vt_symbol] = self.pos.get(trade.vt_symbol, 0) + delta

    # ------------------------------------------------------------ 交易接口

    def buy(self, vt_symbol: str, price: float, volume: float,
            order_type: OrderType = OrderType.LIMIT) -> str:
        return self._send_order(vt_symbol, Direction.LONG, price, volume, order_type)

    def sell(self, vt_symbol: str, price: float, volume: float,
             order_type: OrderType = OrderType.LIMIT) -> str:
        return self._send_order(vt_symbol, Direction.SHORT, price, volume, order_type)

    def _send_order(self, vt_symbol: str, direction: Direction, price: float,
                    volume: float, order_type: OrderType) -> str:
        if not self.trading:
            return ""
        return self.engine.send_order(
            self.strategy_name, vt_symbol, direction, price, volume, order_type
        )

    def cancel_order(self, vt_orderid: str) -> None:
        if self.trading:
            self.engine.cancel_order(vt_orderid)

    def cancel_all(self) -> None:
        """撤销本策略全部活动委托"""
        if self.trading:
            self.engine.cancel_all(self.strategy_name)

    # ------------------------------------------------------------ 查询辅助

    def get_pos(self, vt_symbol: str) -> float:
        return self.pos.get(vt_symbol, 0)

    def get_cash(self) -> float:
        return self.engine.get_cash()

    def load_bars(self, days: int, interval: str = "1d") -> None:
        """加载历史 K 线预热，回测与实盘由各自 engine 实现"""
        self.engine.load_bars(self, days, interval)

    def write_log(self, msg: str) -> None:
        logger.info("[%s] %s", self.strategy_name, msg)
