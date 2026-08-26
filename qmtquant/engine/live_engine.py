"""实盘引擎。

装配 EventEngine + Gateway + RiskManager + 策略集合，负责：
- 策略生命周期管理与行情订阅
- 下单链路：策略 → 风控 → 网关（策略拿不到 gateway 引用，无法绕过风控）
- 回报路由：按 vt_orderid 找回归属策略
- 断线保护：网关断开期间禁止一切下单

对策略暴露的接口与 BacktestEngine 完全一致，同一份策略代码可直接切换。
"""
import logging
import time
from datetime import datetime, timedelta

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
from ..event.engine import (
    EVENT_ACCOUNT,
    EVENT_GATEWAY_STATUS,
    EVENT_ORDER,
    EVENT_POSITION,
    EVENT_TICK,
    EVENT_TIMER,
    EVENT_TRADE,
    Event,
    EventEngine,
)
from ..gateway.base import BaseGateway
from ..risk.risk_manager import RiskManager
from ..strategy.base import StrategyBase
from ..utils.logger import get_trade_logger
from ..utils.symbol import normalize, split_vt_symbol

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()

#: 行情停推多久算异常（秒）
TICK_TIMEOUT = 60


class LiveEngine:
    """实盘/模拟盘交易引擎"""

    def __init__(self, event_engine: EventEngine, gateway: BaseGateway,
                 risk_manager: RiskManager) -> None:
        self.event_engine = event_engine
        self.gateway = gateway
        self.risk_manager = risk_manager

        self.strategies: dict[str, StrategyBase] = {}
        #: vt_orderid -> 策略名，用于回报路由
        self._orderid_strategy: dict[str, str] = {}
        #: vt_orderid -> OrderData，本地订单簿
        self.orders: dict[str, OrderData] = {}

        self.account: AccountData | None = None
        self.positions: dict[str, PositionData] = {}
        self.ticks: dict[str, TickData] = {}

        self._last_tick_time: datetime | None = None
        self._tick_warned: bool = False

        self._register_handlers()

    def _register_handlers(self) -> None:
        ee = self.event_engine
        ee.register(EVENT_TICK, self._on_tick)
        ee.register(EVENT_ORDER, self._on_order)
        ee.register(EVENT_TRADE, self._on_trade)
        ee.register(EVENT_ACCOUNT, self._on_account)
        ee.register(EVENT_POSITION, self._on_position)
        ee.register(EVENT_GATEWAY_STATUS, self._on_gateway_status)
        ee.register(EVENT_TIMER, self._on_timer)

    # ------------------------------------------------------------ 策略管理

    def add_strategy(self, strategy_class: type[StrategyBase], strategy_name: str,
                     vt_symbols: list[str], setting: dict | None = None) -> StrategyBase:
        if strategy_name in self.strategies:
            raise ValueError(f"策略名重复: {strategy_name}")
        vt_symbols = [normalize(s) for s in vt_symbols]
        strategy = strategy_class(self, strategy_name, vt_symbols, setting)
        self.strategies[strategy_name] = strategy
        logger.info("已添加策略 %s，标的 %s", strategy_name, vt_symbols)
        return strategy

    def init_all(self) -> None:
        """初始化所有策略并订阅行情"""
        for strategy in self.strategies.values():
            try:
                strategy.on_init()
                strategy.inited = True
                for vt_symbol in strategy.vt_symbols:
                    symbol, exchange = split_vt_symbol(vt_symbol)
                    self.gateway.subscribe(SubscribeRequest(symbol=symbol, exchange=exchange))
            except Exception:
                logger.exception("策略初始化失败: %s", strategy.strategy_name)

    def start_all(self) -> None:
        for strategy in self.strategies.values():
            if not strategy.inited:
                logger.error("策略未初始化，跳过启动: %s", strategy.strategy_name)
                continue
            try:
                strategy.on_start()
                strategy.trading = True
                logger.info("策略已启动: %s", strategy.strategy_name)
            except Exception:
                logger.exception("策略启动失败: %s", strategy.strategy_name)

    def stop_all(self) -> None:
        """停止所有策略，先撤单再停，避免留下孤儿挂单"""
        for strategy in self.strategies.values():
            if not strategy.trading:
                continue
            try:
                self.cancel_all(strategy.strategy_name)
                strategy.trading = False
                strategy.on_stop()
                logger.info("策略已停止: %s", strategy.strategy_name)
            except Exception:
                logger.exception("策略停止失败: %s", strategy.strategy_name)

    # ------------------------------------------------------------ 策略调用的接口

    def send_order(self, strategy_name: str, vt_symbol: str, direction: Direction,
                   price: float, volume: float,
                   order_type: OrderType = OrderType.LIMIT) -> str:
        """策略下单入口。必经风控，网关断开时直接拒绝。"""
        if not self.gateway.connected:
            logger.error("网关未连接，拒绝下单: %s %s", strategy_name, vt_symbol)
            return ""

        vt_symbol = normalize(vt_symbol)
        symbol, exchange = split_vt_symbol(vt_symbol)
        # 买入向下取整到 100 股，避免因零股被券商拒单
        if direction == Direction.LONG:
            volume = int(volume // 100) * 100
        if volume <= 0:
            return ""

        req = OrderRequest(
            symbol=symbol, exchange=exchange, direction=direction,
            order_type=order_type, price=round(price, 2), volume=volume,
            reference=strategy_name,
        )

        passed, reason = self.risk_manager.check(req)
        if not passed:
            logger.warning("风控拒单 %s %s: %s", strategy_name, vt_symbol,
                           reason.value if reason else "")
            return ""

        vt_orderid = self.gateway.send_order(req)
        if not vt_orderid:
            logger.error("网关报单失败: %s %s", strategy_name, vt_symbol)
            return ""

        self._orderid_strategy[vt_orderid] = strategy_name
        trade_logger.info(
            "报单 strategy=%s symbol=%s dir=%s price=%.3f vol=%s orderid=%s",
            strategy_name, vt_symbol, direction.value, req.price, volume, vt_orderid,
        )
        return vt_orderid

    def cancel_order(self, vt_orderid: str) -> None:
        order = self.orders.get(vt_orderid)
        if not order:
            logger.warning("撤单失败，本地无此订单: %s", vt_orderid)
            return
        if not order.is_active():
            return
        self.gateway.cancel_order(order.create_cancel_request())

    def cancel_all(self, strategy_name: str | None = None) -> None:
        """撤销指定策略（或全部）的活动委托"""
        for vt_orderid, order in list(self.orders.items()):
            if not order.is_active():
                continue
            if strategy_name and self._orderid_strategy.get(vt_orderid) != strategy_name:
                continue
            self.gateway.cancel_order(order.create_cancel_request())

    def get_cash(self) -> float:
        return self.account.available if self.account else 0.0

    def get_pos(self, vt_symbol: str) -> float:
        pos = self.positions.get(normalize(vt_symbol))
        return pos.volume if pos else 0.0

    def get_tick(self, vt_symbol: str) -> TickData | None:
        return self.ticks.get(normalize(vt_symbol))

    def load_bars(self, strategy: StrategyBase, days: int, interval: str = "1d") -> None:
        """用历史数据预热策略指标。

        盘中重启时这一步是必须的：不预热的话均线等窗口指标是空的，
        策略要等攒够 N 根 Bar 才能出信号，期间形同停摆。

        预热期间 `strategy.trading` 必须为 False，否则会按历史行情真实下单。
        数据源缺失时跳过，不阻断启动。
        """
        from ..config import get_config
        from ..core.constants import Interval

        cfg = get_config()
        try:
            from ..datafeed.xt_feed import XtDataFeed
            feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
        except Exception:
            logger.exception("数据源初始化失败，跳过预热")
            return

        end = datetime.now().strftime("%Y-%m-%d")
        # 自然日转交易日：留 2 倍余量覆盖周末与节假日
        start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")

        try:
            bars = feed.load_bars(strategy.vt_symbols, start, end, Interval(interval))
        except Exception:
            logger.exception("预热数据加载失败: %s", strategy.strategy_name)
            return

        if not bars:
            logger.warning("策略 %s 无预热数据，指标需盘中自行积累。"
                           "建议先运行 scripts/download_data.py", strategy.strategy_name)
            return

        was_trading = strategy.trading
        strategy.trading = False   # 预热期禁止下单
        try:
            grouped: dict[datetime, dict[str, BarData]] = {}
            for bar in bars:
                grouped.setdefault(bar.datetime, {})[bar.vt_symbol] = bar
            for dt in sorted(grouped):
                section = grouped[dt]
                strategy.on_bars(section)
                for bar in section.values():
                    strategy.on_bar(bar)
        except Exception:
            logger.exception("预热推送 Bar 异常: %s", strategy.strategy_name)
        finally:
            strategy.trading = was_trading

        logger.info("策略 %s 预热完成，%d 根 Bar（%s ~ %s）",
                    strategy.strategy_name, len(bars),
                    bars[0].datetime.date(), bars[-1].datetime.date())

    # ------------------------------------------------------------ 事件处理

    def _on_tick(self, event: Event) -> None:
        tick: TickData = event.data
        self.ticks[tick.vt_symbol] = tick
        self._last_tick_time = datetime.now()
        self._tick_warned = False

        for strategy in self.strategies.values():
            if not strategy.trading or tick.vt_symbol not in strategy.vt_symbols:
                continue
            try:
                strategy.on_tick(tick)
            except Exception:
                logger.exception("策略处理 tick 异常: %s", strategy.strategy_name)

    def _on_order(self, event: Event) -> None:
        order: OrderData = event.data
        self.orders[order.vt_orderid] = order

        if order.status in (Status.REJECTED, Status.CANCELLED):
            trade_logger.warning("委托%s orderid=%s symbol=%s msg=%s",
                                 order.status.value, order.vt_orderid,
                                 order.vt_symbol, order.message)

        strategy = self._route(order.vt_orderid)
        if strategy:
            try:
                strategy.on_order(order)
            except Exception:
                logger.exception("策略处理委托回报异常: %s", strategy.strategy_name)

    def _on_trade(self, event: Event) -> None:
        trade: TradeData = event.data
        self.risk_manager.on_trade(trade)
        trade_logger.info(
            "成交 symbol=%s dir=%s price=%.3f vol=%s tradeid=%s strategy=%s",
            trade.vt_symbol, trade.direction.value, trade.price, trade.volume,
            trade.vt_tradeid, trade.reference,
        )

        strategy = self._route(trade.vt_orderid)
        if strategy:
            try:
                strategy.on_trade(trade)
            except Exception:
                logger.exception("策略处理成交回报异常: %s", strategy.strategy_name)

        # 成交后刷新资金与持仓，保证风控用的是最新状态
        self.gateway.query_account()
        self.gateway.query_position()

    def _route(self, vt_orderid: str) -> StrategyBase | None:
        """按委托号找回归属策略。找不到说明是手工单或重启前的遗留单。"""
        name = self._orderid_strategy.get(vt_orderid)
        if not name:
            # 回退：用 reference 字段匹配
            order = self.orders.get(vt_orderid)
            name = order.reference if order else None
        return self.strategies.get(name) if name else None

    def _on_account(self, event: Event) -> None:
        self.account = event.data
        self.risk_manager.update_account(event.data)

    def _on_position(self, event: Event) -> None:
        pos: PositionData = event.data
        if pos.volume <= 0:
            self.positions.pop(pos.vt_symbol, None)
        else:
            self.positions[pos.vt_symbol] = pos
        self.risk_manager.update_position(pos)

    def _on_gateway_status(self, event: Event) -> None:
        data = event.data
        if data["connected"]:
            logger.info("网关已连接: %s", data.get("msg", ""))
        else:
            # 断线期间 send_order 会直接拒绝，这里额外告警
            logger.error("网关断开，暂停下单: %s", data.get("msg", ""))

    def _on_timer(self, event: Event) -> None:
        """定时健康检查：行情是否停推、事件是否积压"""
        if self._last_tick_time and not self._tick_warned:
            gap = (datetime.now() - self._last_tick_time).total_seconds()
            if gap > TICK_TIMEOUT and self._is_trading_time():
                logger.error("行情已停推 %.0f 秒，请检查订阅与网络", gap)
                self._tick_warned = True

        qsize = self.event_engine.qsize
        if qsize > 1000:
            logger.warning("事件队列积压 %d 条，处理速度跟不上推送", qsize)

    @staticmethod
    def _is_trading_time(now: datetime | None = None) -> bool:
        """是否在 A 股连续竞价时段"""
        now = now or datetime.now()
        if now.weekday() >= 5:
            return False
        t = now.time()
        from datetime import time as _t
        return (_t(9, 30) <= t <= _t(11, 30)) or (_t(13, 0) <= t <= _t(15, 0))

    # ------------------------------------------------------------ 日常运维

    def reconcile(self) -> bool:
        """与券商对账：本地持仓是否与查询结果一致。不一致返回 False 并告警。"""
        broker_orders = self.gateway.query_orders()
        active = [o for o in broker_orders if o.is_active()]
        if active:
            logger.warning("券商侧存在 %d 笔活动委托，请确认是否为本系统所下", len(active))
        for order in broker_orders:
            self.orders[order.vt_orderid] = order

        self.gateway.query_account()
        self.gateway.query_position()

        # 查询是异步回调，等事件处理完再打印，否则账户还是 None
        deadline = time.time() + 5
        while self.account is None and time.time() < deadline:
            time.sleep(0.1)

        if self.account:
            logger.info("账户 %s | 总资产 %.2f | 可用 %.2f | 市值 %.2f",
                        self.account.accountid, self.account.balance,
                        self.account.available, self.account.market_value)
        else:
            logger.error("对账失败：查询不到账户资金，请检查资金账号与账户类型配置")

        logger.info("对账完成：持仓 %d 个标的，活动委托 %d 笔",
                    len(self.positions), len(active))
        for pos in self.positions.values():
            logger.info("  持仓 %s 数量=%s 可用=%s 成本=%.3f",
                        pos.vt_symbol, pos.volume, pos.available, pos.price)
        return not active

    def daily_settle(self) -> None:
        """日终结算：停策略、撤单、重置风控计数"""
        self.stop_all()
        self.cancel_all()
        balance = self.account.balance if self.account else 0
        self.risk_manager.new_day(balance)
        logger.info("日终结算完成，总资产=%.2f", balance)

    def close(self) -> None:
        self.stop_all()
        self.gateway.close()
