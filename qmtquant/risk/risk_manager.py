"""风控管理器。

设计原则：风控是**硬约束**，策略无法绕过。
所有下单请求必须经过 `check()`，返回 (是否放行, 拒绝原因)。
"""
import logging
from datetime import date

from ..config import RiskConfig
from ..core.constants import Direction, RejectReason
from ..core.objects import AccountData, OrderRequest, PositionData, TradeData
from ..event.engine import EVENT_RISK_REJECT, Event, EventEngine
from ..utils.logger import get_trade_logger
from .drawdown import DrawdownConfig, DrawdownController

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class RiskManager:
    """下单前置风控 + 全局急停"""

    def __init__(self, config: RiskConfig, event_engine: EventEngine | None = None) -> None:
        self.config = config
        self.event_engine = event_engine

        # 回撤控制：覆盖「连续阴跌」盲区 —— 每天亏 1% 连亏 20 天累计 18%，
        # 却一次都不会触及 3% 的日亏线
        self.drawdown = DrawdownController(DrawdownConfig(
            enabled=config.drawdown_enabled,
            close_only_threshold=config.drawdown_close_only,
            reduce_threshold=config.drawdown_reduce,
            reduce_keep_ratio=config.drawdown_reduce_keep,
            flat_threshold=config.drawdown_flat,
            recovery_ratio=config.drawdown_recovery_ratio,
            min_observations=config.drawdown_min_observations,
            max_freeze_observations=config.drawdown_max_freeze,
        ))

        #: 全局急停：True 时拒绝一切下单
        self.kill_switch: bool = False
        #: 半开状态：只允许卖出（日内亏损触线时进入）
        self.close_only: bool = False

        self._trade_date: date = date.today()
        self._order_count: int = 0
        self._turnover: float = 0.0

        self.account: AccountData | None = None
        self.positions: dict[str, PositionData] = {}
        #: 每日开盘时记录的总资产，用于计算当日盈亏
        self._day_start_balance: float = 0.0

        # ---- 在途预留（见 check() 的说明）----
        #: 已放行但尚未落地的买单金额，按标的累计
        self._pending_buy: dict[str, float] = {}
        #: 已放行但尚未落地的卖单数量，按标的累计
        self._pending_sell: dict[str, float] = {}

    # ------------------------------------------------------------ 状态更新

    def update_account(self, account: AccountData) -> None:
        self.account = account
        if not self._day_start_balance:
            self._day_start_balance = account.balance
        self._check_daily_loss()
        self.drawdown.update(account.balance)

    def update_position(self, position: PositionData) -> None:
        if position.volume <= 0:
            self.positions.pop(position.vt_symbol, None)
        else:
            self.positions[position.vt_symbol] = position

    def on_trade(self, trade: TradeData) -> None:
        self._turnover += trade.price * trade.volume
        # 成交即落地，释放对应预留
        if trade.direction == Direction.LONG:
            self._release_buy(trade.vt_symbol, trade.price * trade.volume)
        else:
            self._release_sell(trade.vt_symbol, trade.volume)

    def new_day(self, balance: float) -> None:
        """交易日切换时重置当日额度"""
        self._trade_date = date.today()
        self._order_count = 0
        self._turnover = 0.0
        self._day_start_balance = balance
        self.close_only = False
        # 隔夜挂单已被券商清掉，在途预留不能带到第二天
        self.reset_reservations()
        logger.info("风控计数已重置，日初总资产=%.2f", balance)

    def _check_daily_loss(self) -> None:
        """当日亏损触线则进入只平不开"""
        if not self.account or not self._day_start_balance:
            return
        loss_ratio = (self._day_start_balance - self.account.balance) / self._day_start_balance
        if loss_ratio >= self.config.daily_loss_limit_ratio and not self.close_only:
            self.close_only = True
            logger.error("当日亏损 %.2f%% 触及阈值 %.2f%%，进入只平不开",
                         loss_ratio * 100, self.config.daily_loss_limit_ratio * 100)

    # ------------------------------------------------------------ 急停

    def activate_kill_switch(self, reason: str = "") -> None:
        self.kill_switch = True
        logger.error("【急停已开启】%s", reason)

    def release_kill_switch(self) -> None:
        self.kill_switch = False
        logger.warning("急停已解除")

    # ------------------------------------------------------------ 核心校验

    def check(self, req: OrderRequest) -> tuple[bool, RejectReason | None]:
        """下单前置校验。任一项不通过即拒单。

        ## 为什么放行后要预留额度

        校验读的是 `self.account` 这份快照，而账户快照要等券商推送才会变。
        批量调仓时在循环里连发 N 笔买单，如果放行不留痕，每一笔都拿同一份
        没变过的快照去比 —— 50 万可用资金，10 笔各 45 万的买单会**全部放行**，
        因为每笔单独看都「不超过可用资金」。单票占比、总仓位同理。

        所以放行即记入 `_pending_*`，后续校验扣掉在途量。预留在成交
        （`on_trade`）或委托终态（`on_order`）时释放，`new_day` 时清空。

        账户推送刷新时**不清预留**：券商冻结资金与本地预留会短暂重复计算，
        方向是偏保守（宁可拒单），这正是风控该犯的错误方向。
        """
        reason = self._do_check(req)
        if reason is not None:
            trade_logger.warning(
                "风控拒单 symbol=%s dir=%s price=%.3f vol=%s reason=%s ref=%s",
                req.vt_symbol, req.direction.value, req.price, req.volume,
                reason.value, req.reference,
            )
            if self.event_engine:
                self.event_engine.put(Event(EVENT_RISK_REJECT,
                                            {"request": req, "reason": reason}))
            return False, reason
        self._order_count += 1
        self._reserve(req)
        return True, None

    # ------------------------------------------------------------ 在途预留

    def _reserve(self, req: OrderRequest) -> None:
        sym = req.vt_symbol
        if req.direction == Direction.LONG:
            self._pending_buy[sym] = (self._pending_buy.get(sym, 0.0)
                                      + req.price * req.volume)
        else:
            self._pending_sell[sym] = (self._pending_sell.get(sym, 0.0)
                                       + req.volume)

    def _release_buy(self, vt_symbol: str, value: float) -> None:
        if value <= 0:
            return
        left = self._pending_buy.get(vt_symbol, 0.0) - value
        if left > 1e-6:
            self._pending_buy[vt_symbol] = left
        else:
            self._pending_buy.pop(vt_symbol, None)

    def _release_sell(self, vt_symbol: str, volume: float) -> None:
        if volume <= 0:
            return
        left = self._pending_sell.get(vt_symbol, 0.0) - volume
        if left > 1e-6:
            self._pending_sell[vt_symbol] = left
        else:
            self._pending_sell.pop(vt_symbol, None)

    @property
    def pending_buy_value(self) -> float:
        """在途买单总金额"""
        return sum(self._pending_buy.values())

    def release_reservation(self, req: OrderRequest) -> None:
        """撤销一笔已放行但最终没发出去的预留。

        风控放行后网关仍可能报单失败（未连接、参数错、券商拒收）。
        这时预留必须还回去，否则额度被一笔根本不存在的委托永久占着，
        当天后续下单会被逐渐挤死。
        """
        if req.direction == Direction.LONG:
            self._release_buy(req.vt_symbol, req.price * req.volume)
        else:
            self._release_sell(req.vt_symbol, req.volume)

    def reset_reservations(self) -> None:
        """清空在途预留。

        用于调用方确知没有在途委托的场景（如脚本启动时）。
        盘中滥用会让批量下单重新击穿额度，别拿它当「拒单了就清一下」的开关。
        """
        self._pending_buy.clear()
        self._pending_sell.clear()

    def on_order(self, order) -> None:
        """委托状态更新。终态时释放未成交部分的预留。"""
        from ..core.constants import Status
        if order.status not in (Status.CANCELLED, Status.REJECTED):
            return
        untraded = max(0.0, (order.volume or 0) - (order.traded or 0))
        if untraded <= 0:
            return
        if order.direction == Direction.LONG:
            self._release_buy(order.vt_symbol, untraded * (order.price or 0))
        else:
            self._release_sell(order.vt_symbol, untraded)

    def _do_check(self, req: OrderRequest) -> RejectReason | None:
        cfg = self.config
        is_buy = req.direction == Direction.LONG

        if self.kill_switch:
            return RejectReason.KILL_SWITCH
        if self.close_only and is_buy:
            return RejectReason.DAILY_LOSS_LIMIT
        # 回撤达到任一档位即停止开新仓；卖出始终放行，否则无法减仓自救
        if is_buy and not self.drawdown.allow_open():
            return RejectReason.DRAWDOWN_LIMIT

        # --- 数量合法性：买入必须 100 股整数倍，卖出允许零股（清仓场景）
        if req.volume <= 0:
            return RejectReason.INVALID_VOLUME
        if is_buy and req.volume % 100 != 0:
            return RejectReason.INVALID_VOLUME

        # --- 黑名单
        if req.vt_symbol in cfg.blacklist and is_buy:
            return RejectReason.BLACKLIST

        # --- 当日额度
        if self._order_count >= cfg.max_order_count_per_day:
            return RejectReason.ORDER_COUNT_LIMIT
        if self._turnover >= cfg.max_turnover_per_day:
            return RejectReason.TURNOVER_LIMIT

        order_value = req.price * req.volume
        if order_value > cfg.max_order_value:
            return RejectReason.ORDER_VALUE_LIMIT

        if is_buy:
            if not self.account:
                # 拿不到账户就不放行，宁可漏单也不越权
                return RejectReason.INSUFFICIENT_CASH
            # 扣掉在途买单，否则循环下单时每笔都拿同一份快照比，额度会被击穿
            pending_all = self.pending_buy_value
            if order_value > self.account.available - pending_all:
                return RejectReason.INSUFFICIENT_CASH

            balance = self.account.balance or 1
            # 单票占比：已有市值 + 该票在途买单 + 本次委托金额
            pos = self.positions.get(req.vt_symbol)
            held_value = pos.volume * req.price if pos else 0
            pending_sym = self._pending_buy.get(req.vt_symbol, 0.0)
            if ((held_value + pending_sym + order_value) / balance
                    > cfg.max_position_ratio):
                return RejectReason.POSITION_RATIO_LIMIT

            total_after = (self.account.market_value + pending_all
                           + order_value)
            if total_after / balance > cfg.max_total_position_ratio:
                return RejectReason.TOTAL_POSITION_LIMIT
        else:
            pos = self.positions.get(req.vt_symbol)
            pending_v = self._pending_sell.get(req.vt_symbol, 0.0)
            if not pos or pos.available - pending_v < req.volume:
                return RejectReason.INSUFFICIENT_POSITION

        return None

    # ------------------------------------------------------------ 观测

    def stats(self) -> dict:
        """当日风控使用情况，供监控展示"""
        return {
            "date": str(self._trade_date),
            "order_count": self._order_count,
            "order_count_limit": self.config.max_order_count_per_day,
            "turnover": round(self._turnover, 2),
            "turnover_limit": self.config.max_turnover_per_day,
            "kill_switch": self.kill_switch,
            "close_only": self.close_only,
            "drawdown": round(self.drawdown.drawdown, 4),
            "drawdown_level": self.drawdown.level.label,
            "target_position_ratio": self.drawdown.target_position_ratio(),
        }
