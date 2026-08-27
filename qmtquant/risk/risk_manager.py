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

    def new_day(self, balance: float) -> None:
        """交易日切换时重置当日额度"""
        self._trade_date = date.today()
        self._order_count = 0
        self._turnover = 0.0
        self._day_start_balance = balance
        self.close_only = False
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
        """下单前置校验。任一项不通过即拒单。"""
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
        return True, None

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
            if order_value > self.account.available:
                return RejectReason.INSUFFICIENT_CASH

            balance = self.account.balance or 1
            # 单票占比：已有市值 + 本次委托金额
            pos = self.positions.get(req.vt_symbol)
            held_value = pos.volume * req.price if pos else 0
            if (held_value + order_value) / balance > cfg.max_position_ratio:
                return RejectReason.POSITION_RATIO_LIMIT

            total_after = self.account.market_value + order_value
            if total_after / balance > cfg.max_total_position_ratio:
                return RejectReason.TOTAL_POSITION_LIMIT
        else:
            pos = self.positions.get(req.vt_symbol)
            if not pos or pos.available < req.volume:
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
