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
from datetime import date as _date
from datetime import datetime, timedelta
from datetime import time as _dtime

from ..core.constants import Direction, OrderType, Status
from ..core.objects import (
    AccountData,
    BarData,
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

#: 策略状态定时落库间隔（秒）。
#: 只在成交与优雅退出时保存是不够的 —— 进程被强杀（SIGTERM/断电）时
#: finally 不会执行，自上次成交以来的状态全部丢失。定时保存把损失
#: 上限压到这个间隔内。
STATE_SAVE_INTERVAL = 60

#: 收盘时间。过了这个点撤掉未成交挂单 —— 留到隔夜没有意义：
#: 券商当日撤销后本地仍认为它活着，第二天的调仓会基于错误的在途状态计算。
MARKET_CLOSE = _dtime(15, 0)

#: 委托挂多久没成交就撤（秒）。
#:
#: 不撤的后果不是「单还挂着」这么轻描淡写：限价单挂一天不成交，
#: 目标持仓永远达不到，而系统不会告诉你 —— 调仓差异下次还这么算，
#: 风控的在途预留也一直占着额度。撤掉至少让状态回到确定：
#: 要么成交要么没有，下一轮重新决策。
ORDER_TIMEOUT = 300


class LiveEngine:
    """实盘/模拟盘交易引擎"""

    def __init__(self, event_engine: EventEngine, gateway: BaseGateway,
                 risk_manager: RiskManager, store=None,
                 order_timeout: int = ORDER_TIMEOUT) -> None:
        """
        :param store: StateStore。传入后策略状态、回撤记忆、成交流水会落库，
            重启可恢复。**持仓与资金不从这里恢复** —— 券商是唯一真相来源，
            信本地会导致重复下单或以为持有已被卖出的仓位。
        :param order_timeout: 委托挂单超时（秒），0 表示不撤。
        """
        self.order_timeout = order_timeout
        self.event_engine = event_engine
        self.gateway = gateway
        self.risk_manager = risk_manager
        self.store = store

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
        self._last_state_save: float = time.time()

        #: 引擎认为的当前交易日。与系统日期不符即触发日切。
        self._engine_date: _date = datetime.now().date()
        #: 今日收盘处理是否已执行（撤单 + 落库），避免重复触发
        self._closed_today: bool = False
        #: vt_orderid -> 报单时刻，用于超时撤单
        self._order_submit_time: dict[str, float] = {}
        #: 已因超时发过撤单指令的委托，避免反复发撤单
        self._timeout_cancelled: set[str] = set()

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

        if self.store is not None:
            saved = self.store.load_state(strategy_name)
            if saved:
                ts = self.store.state_updated_at(strategy_name)
                logger.info("发现 %s 的持久化状态（保存于 %s），正在恢复",
                            strategy_name, ts)
                try:
                    strategy.restore_variables(saved)
                except Exception:
                    logger.exception("恢复策略状态失败，将以全新状态启动: %s",
                                     strategy_name)
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

    def save_all_states(self) -> None:
        """把所有策略的运行时状态落库。

        在成交后与停止时调用 —— 成交会改变持仓，此时不存的话
        崩溃重启就会丢掉这次变化。
        """
        if self.store is None:
            return
        for name, strategy in self.strategies.items():
            try:
                self.store.save_state(name, strategy.get_variables())
            except Exception:
                logger.exception("保存策略状态失败: %s", name)

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
        self.save_all_states()

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
            # 风控已经为这笔单预留了额度，单没发出去就要还回去，
            # 否则额度被一笔不存在的委托永久占着
            self.risk_manager.release_reservation(req)
            logger.error("网关报单失败: %s %s", strategy_name, vt_symbol)
            return ""

        self._orderid_strategy[vt_orderid] = strategy_name
        self._order_submit_time[vt_orderid] = time.time()
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
        self._persist(lambda s: s.save_order(order), "委托")

        # 撤单/废单要把未成交部分的额度还给风控，否则预留会一直占着，
        # 当天后续调仓会被误拒
        self.risk_manager.on_order(order)

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

        # 成交后立刻落库：成交改变了持仓，此时不存的话崩溃重启会丢掉这次变化
        self._persist(lambda s: s.save_trade(trade), "成交")
        self.save_all_states()

        # 成交后刷新资金与持仓，保证风控用的是最新状态
        self.gateway.query_account()
        self.gateway.query_position()

    def _persist(self, fn, what: str) -> None:
        """落库失败不能影响交易 —— 持久化是辅助能力，
        为了记账而中断实盘是本末倒置。失败只记日志。
        """
        if self.store is None:
            return
        try:
            fn(self.store)
        except Exception:
            logger.exception("持久化%s失败", what)

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
        self._persist(
            lambda s: s.save_equity(event.data.balance, event.data.available,
                                    event.data.market_value), "净值")

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
        """定时健康检查：日切、超时撤单、行情停推、事件积压"""
        self._check_day_rollover()
        self._check_order_timeout()

        if self._last_tick_time and not self._tick_warned:
            gap = (datetime.now() - self._last_tick_time).total_seconds()
            if gap > TICK_TIMEOUT and self._is_trading_time():
                logger.error("行情已停推 %.0f 秒，请检查订阅与网络", gap)
                self._tick_warned = True

        qsize = self.event_engine.qsize
        if qsize > 1000:
            logger.warning("事件队列积压 %d 条，处理速度跟不上推送", qsize)

        # 定时落库：进程被强杀时 finally 不会执行，靠这个兜底
        if (self.store is not None
                and time.time() - self._last_state_save >= STATE_SAVE_INTERVAL):
            self._last_state_save = time.time()
            self.save_all_states()

    def _check_order_timeout(self, now_ts: float | None = None) -> None:
        """撤掉挂太久没成交的委托。

        部分成交的单同样撤 —— 撤掉的是未成交余量，已成交部分不受影响。
        撤单回报到达时 risk_manager.on_order 会把余量的预留还回去。

        这里只发撤单指令，不假设它一定成功：券商可能已经成交或已撤。
        `_timeout_cancelled` 记录发过指令的单，避免每次定时器都重发。
        """
        if self.order_timeout <= 0:
            return
        now_ts = now_ts or time.time()

        for vt_orderid, order in list(self.orders.items()):
            if not order.is_active():
                self._order_submit_time.pop(vt_orderid, None)
                self._timeout_cancelled.discard(vt_orderid)
                continue
            if vt_orderid in self._timeout_cancelled:
                continue
            submit = self._order_submit_time.get(vt_orderid)
            if submit is None:
                # 重启后从券商同步回来的单没有本地报单时刻。
                # 按「此刻开始计时」处理，给它完整的超时窗口 ——
                # 当成 0 会在重启瞬间把所有在途单全撤掉。
                self._order_submit_time[vt_orderid] = now_ts
                continue
            waited = now_ts - submit
            if waited < self.order_timeout:
                continue

            self._timeout_cancelled.add(vt_orderid)
            untraded = (order.volume or 0) - (order.traded or 0)
            logger.warning(
                "委托挂单超时 %.0f 秒未成交，撤单: %s %s 未成交 %s 股",
                waited, vt_orderid, order.vt_symbol, untraded)
            try:
                self.cancel_order(vt_orderid)
            except Exception:
                logger.exception("超时撤单失败: %s", vt_orderid)

    def _check_day_rollover(self, now: datetime | None = None) -> None:
        """跨日重置与收盘撤单。

        ## 不做会怎样

        `risk_manager.new_day()` 此前全仓没有调用者，后果是跨日累加：
        `_order_count` 跑到第三天就撞上 max_order_count_per_day 全天拒单；
        `_day_start_balance` 停在进程启动那天，日亏 3% 线拿今天和三天前比；
        `close_only` 一旦触发永不复位，之后每个交易日都只平不开。

        引擎要跨日常驻，所以这里做的是**滚动日切**而非停机结算：
        收盘撤单、次日零点重置计数，策略不停。停机式结算见 `daily_settle()`。
        """
        now = now or datetime.now()
        today = now.date()

        # ---- 收盘：撤掉未成交挂单并落库 ----
        if (not self._closed_today and today.weekday() < 5
                and now.time() >= MARKET_CLOSE):
            self._closed_today = True
            active = [o for o in self.orders.values() if o.is_active()]
            if active:
                logger.info("收盘撤单：%d 笔未成交委托", len(active))
                self.cancel_all()
            self.save_all_states()
            # 同步节流计时，否则紧接着的定时落库会再存一次
            self._last_state_save = time.time()

        # ---- 跨日：重置风控当日额度 ----
        if today != self._engine_date:
            self._engine_date = today
            self._closed_today = False
            balance = self.account.balance if self.account else 0
            self.risk_manager.new_day(balance)
            logger.info("交易日切换至 %s，风控当日额度已重置（日初总资产=%.2f）",
                        today, balance)

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
