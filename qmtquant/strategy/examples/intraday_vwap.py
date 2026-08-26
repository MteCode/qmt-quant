"""日内分时均价线（KMCP / VWAP）策略。

## A 股的 T+1 约束决定了这个策略能做什么

**当天买入的股票当天卖不掉。** 所以「日内低买高卖赚差价」在 A 股做不了多头。
本策略的实际用法有两种，用 `role` 参数切换：

| role | 用法 | 说明 |
|------|------|------|
| `entry_timing` | 买入择时 | 日线策略给出「今天要买」，由本策略决定当天哪一分钟下单 |
| `t0_rotation` | T+0 回转 | 有底仓时：先卖昨仓、再低位买回，净持仓不变但摊低成本 |

`t0_rotation` 必须**先卖后买**，因为当日买入的份额被冻结、卖不掉。
顺序反了就会变成单纯加仓，且资金占用翻倍。

## 信号方向

| mode | 逻辑 |
|------|------|
| `trend` | 上穿均价线买、下穿卖（跟随日内动能） |
| `reversion` | 下穿均价线买、上穿卖（日内超跌反弹） |

⚠ 示例策略，用于验证框架链路，**不构成投资建议**。
分钟线只有约 1 年历史（券商限制），样本外窗口很短，过拟合风险高。
"""
import logging

from ..base import StrategyBase
from ..indicators import CrossDetector, IntradayVwap

logger = logging.getLogger(__name__)

MODE_TREND = "trend"
MODE_REVERSION = "reversion"

ROLE_ENTRY = "entry_timing"
ROLE_T0 = "t0_rotation"


class IntradayVwapStrategy(StrategyBase):
    """分时均价线策略"""

    parameters = ["mode", "role", "trade_ratio", "min_bars",
                  "start_minute", "stop_minute"]
    variables = ["inited", "trading", "pos", "vwap_value"]

    mode: str = MODE_REVERSION
    role: str = ROLE_ENTRY
    #: 单次交易占用比例（entry_timing 用可用资金，t0_rotation 用底仓数量）
    trade_ratio: float = 0.95
    #: 开盘后至少积累多少根分钟 Bar 才允许交易。
    #: 开盘前几分钟均价线极不稳定，一两笔大单就能带偏。
    min_bars: int = 15
    #: 允许交易的时间窗口（HHMM）。尾盘不开新仓，避免收盘前来不及处理
    start_minute: int = 945
    stop_minute: int = 1450

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)

        if self.mode not in (MODE_TREND, MODE_REVERSION):
            raise ValueError(f"未知 mode: {self.mode}")
        if self.role not in (ROLE_ENTRY, ROLE_T0):
            raise ValueError(f"未知 role: {self.role}")

        self.vwap = {s: IntradayVwap() for s in vt_symbols}
        self.cross = {s: CrossDetector() for s in vt_symbols}
        self.vwap_value: dict[str, float] = {}

        self._bar_count: dict[str, int] = {}
        self._day: dict[str, object] = {}
        #: 当日是否已完成动作，避免同一天反复进出
        self._done_today: dict[str, bool] = {}
        #: t0_rotation：当日已卖出待买回的数量
        self._pending_buyback: dict[str, float] = {}

    def on_init(self) -> None:
        self.write_log(f"初始化 mode={self.mode} role={self.role} "
                       f"窗口={self.start_minute}~{self.stop_minute}")

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        self.write_log("策略停止")

    # ------------------------------------------------------------ 主逻辑

    def on_bar(self, bar) -> None:
        if bar.suspended or bar.close_price <= 0:
            return

        vt_symbol = bar.vt_symbol
        vwap_calc = self.vwap.get(vt_symbol)
        if vwap_calc is None:
            return

        # 新交易日：重置日内状态。不重置会把昨天的成交混进均价线
        day = bar.datetime.date()
        if self._day.get(vt_symbol) != day:
            self._day[vt_symbol] = day
            self._bar_count[vt_symbol] = 0
            self._done_today[vt_symbol] = False
            self._pending_buyback[vt_symbol] = 0.0
            self.cross[vt_symbol].reset()

        vwap = vwap_calc.update(bar.datetime, bar.turnover, bar.volume,
                                fallback_price=bar.close_price)
        self._bar_count[vt_symbol] += 1
        if vwap is None:
            return
        self.vwap_value[vt_symbol] = vwap

        signal = self.cross[vt_symbol].update(bar.close_price, vwap)

        if not self.trading or self._done_today.get(vt_symbol):
            return
        if self._bar_count[vt_symbol] < self.min_bars:
            return
        if not self._in_window(bar.datetime):
            return
        if not signal:
            return

        if self.role == ROLE_ENTRY:
            self._handle_entry_timing(vt_symbol, bar, signal)
        else:
            self._handle_t0(vt_symbol, bar, signal)

    def _in_window(self, dt) -> bool:
        hhmm = dt.hour * 100 + dt.minute
        return self.start_minute <= hhmm <= self.stop_minute

    def _buy_signal(self, signal: str) -> bool:
        return signal == ("up" if self.mode == MODE_TREND else "down")

    def _sell_signal(self, signal: str) -> bool:
        return signal == ("down" if self.mode == MODE_TREND else "up")

    # ------------------------------------------------------------ 两种角色

    def _handle_entry_timing(self, vt_symbol: str, bar, signal: str) -> None:
        """买入择时：只在空仓时择机买入，卖出交给日线策略或风控"""
        if not self._buy_signal(signal):
            return
        if self.get_pos(vt_symbol) > 0:
            return

        cash = self.get_cash() * self.trade_ratio
        volume = int(cash / bar.close_price // 100) * 100
        if volume <= 0:
            return
        if self.buy(vt_symbol, bar.close_price * 1.005, volume):
            self._done_today[vt_symbol] = True
            self.write_log(f"日内择时买入 {vt_symbol} {volume}股 "
                           f"@{bar.close_price:.2f} 均价={self.vwap_value[vt_symbol]:.2f}")

    def _handle_t0(self, vt_symbol: str, bar, signal: str) -> None:
        """T+0 回转：先卖昨仓，再低位买回。

        必须先卖后买 —— 当日买入的份额被冻结，卖不掉。
        顺序反了会变成单纯加仓，资金占用翻倍且无法回转。
        """
        pending = self._pending_buyback.get(vt_symbol, 0.0)

        # 阶段一：高位卖出昨仓
        if pending <= 0 and self._sell_signal(signal):
            available = self.get_pos(vt_symbol)
            volume = int(available * self.trade_ratio // 100) * 100
            if volume <= 0:
                return
            if self.sell(vt_symbol, bar.close_price * 0.995, volume):
                self._pending_buyback[vt_symbol] = volume
                self.write_log(f"T0 卖出 {vt_symbol} {volume}股 "
                               f"@{bar.close_price:.2f} 均价={self.vwap_value[vt_symbol]:.2f}")
            return

        # 阶段二：低位买回，恢复底仓
        if pending > 0 and self._buy_signal(signal):
            if self.buy(vt_symbol, bar.close_price * 1.005, pending):
                self._pending_buyback[vt_symbol] = 0.0
                self._done_today[vt_symbol] = True
                self.write_log(f"T0 买回 {vt_symbol} {pending}股 "
                               f"@{bar.close_price:.2f} 均价={self.vwap_value[vt_symbol]:.2f}")

    def on_stop_day(self, vt_symbol: str) -> None:
        """收盘检查：T0 卖出后没买回，底仓就少了一块，必须告警"""
        pending = self._pending_buyback.get(vt_symbol, 0.0)
        if pending > 0:
            logger.warning("[%s] %s 当日 T0 卖出 %s 股未买回，底仓已减少",
                           self.strategy_name, vt_symbol, pending)
