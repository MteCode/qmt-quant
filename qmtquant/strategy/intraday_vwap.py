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

## ⚠ 数据约束：样本外窗口极短

分钟线只有约 1 年历史（券商限制），而日线有 20 年。
这意味着**本策略无法做像样的样本外检验** ——
1 年数据切成训练/测试后，每段只剩半年，任何结论都不稳健。

这是数据问题不是代码问题，加再多机制也解决不了。
把它工程化的意义在于：`entry_timing` 角色本身不需要独立盈利，
它只是给日线策略提供一个更好的下单时点，收益归因也应该这么算。

⚠ 本策略不构成投资建议，过拟合风险显著高于日线策略。
"""
import logging

from .base import StrategyBase
from .indicators import CrossDetector, IntradayVwap

logger = logging.getLogger(__name__)

MODE_TREND = "trend"
MODE_REVERSION = "reversion"

ROLE_ENTRY = "entry_timing"
ROLE_T0 = "t0_rotation"


class IntradayVwapStrategy(StrategyBase):
    """分时均价线策略"""

    parameters = ["mode", "role", "trade_ratio", "min_bars",
                  "start_minute", "stop_minute", "price_buffer"]
    variables = ["inited", "trading", "pos", "vwap_value", "trade_count"]

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
    #: 限价缓冲。日内成交在下一根分钟 Bar，跳空幅度远小于日线，
    #: 所以 0.5% 足够 —— 日线策略需要 3% 是因为隔夜跳空
    price_buffer: float = 0.005

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._coerce_types()
        self._validate()

        self.vwap = {s: IntradayVwap() for s in vt_symbols}
        self.cross = {s: CrossDetector() for s in vt_symbols}
        self.vwap_value: dict[str, float] = {}

        self._bar_count: dict[str, int] = {}
        self._day: dict[str, object] = {}
        #: 当日是否已完成动作，避免同一天反复进出
        self._done_today: dict[str, bool] = {}
        #: t0_rotation：当日已卖出待买回的数量
        self._pending_buyback: dict[str, float] = {}
        #: 买卖次数
        self.trade_count: int = 0
        #: T0 卖出后未能买回的次数 —— 每一次都意味着底仓永久少了一块
        self.failed_buyback_count: int = 0

    def _coerce_types(self) -> None:
        """整数参数强制转型。参数寻优会从 DataFrame 传入 float，
        与 hhmm 做整数比较时会出现 945.0 != 945 之类的意外"""
        for name in ("min_bars", "start_minute", "stop_minute"):
            setattr(self, name, int(getattr(self, name)))
        for name in ("trade_ratio", "price_buffer"):
            setattr(self, name, float(getattr(self, name)))

    def _validate(self) -> None:
        if self.mode not in (MODE_TREND, MODE_REVERSION):
            raise ValueError(f"未知 mode: {self.mode}，可选 trend / reversion")
        if self.role not in (ROLE_ENTRY, ROLE_T0):
            raise ValueError(
                f"未知 role: {self.role}，可选 entry_timing / t0_rotation")
        if not 0 < self.trade_ratio <= 1:
            raise ValueError("trade_ratio 必须在 (0, 1] 之间")
        if self.min_bars < 1:
            raise ValueError("min_bars 至少为 1")
        if self.stop_minute <= self.start_minute:
            raise ValueError(
                f"stop_minute 必须晚于 start_minute，"
                f"实际 {self.stop_minute} <= {self.start_minute}")
        # 尾盘不留出时间，T0 卖出后可能来不及买回，底仓就少了一块
        if self.role == ROLE_T0 and self.stop_minute > 1450:
            raise ValueError(
                f"T0 回转的 stop_minute 不应晚于 1450，实际 {self.stop_minute}"
                "：留出买回时间，否则卖出后来不及买回，底仓会永久减少")

    def on_init(self) -> None:
        self.write_log(f"初始化 mode={self.mode} role={self.role} "
                       f"窗口={self.start_minute}~{self.stop_minute}")

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        msg = f"策略停止。累计买卖 {self.trade_count} 次"
        if self.failed_buyback_count:
            msg += f"，⚠ T0 未买回 {self.failed_buyback_count} 次"
        self.write_log(msg)

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

        # 必须传后复权的收盘价而非 bar.turnover：成交额字段是未复权的，
        # 两者混用会让均价线落在另一个价格空间，穿越永不触发（静默零成交）
        vwap = vwap_calc.update(bar.datetime, bar.close_price, bar.volume)
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
        if self.buy(vt_symbol, bar.close_price * (1 + self.price_buffer), volume):
            self._done_today[vt_symbol] = True
            self.trade_count += 1
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
            if self.sell(vt_symbol, bar.close_price * (1 - self.price_buffer), volume):
                self._pending_buyback[vt_symbol] = volume
                self.trade_count += 1
                self.write_log(f"T0 卖出 {vt_symbol} {volume}股 "
                               f"@{bar.close_price:.2f} 均价={self.vwap_value[vt_symbol]:.2f}")
            return

        # 阶段二：低位买回，恢复底仓
        if pending > 0 and self._buy_signal(signal):
            if self.buy(vt_symbol, bar.close_price * (1 + self.price_buffer), pending):
                self._pending_buyback[vt_symbol] = 0.0
                self._done_today[vt_symbol] = True
                self.trade_count += 1
                self.write_log(f"T0 买回 {vt_symbol} {pending}股 "
                               f"@{bar.close_price:.2f} 均价={self.vwap_value[vt_symbol]:.2f}")

    def on_stop_day(self, vt_symbol: str) -> None:
        """收盘检查：T0 卖出后没买回，底仓就少了一块，必须告警"""
        pending = self._pending_buyback.get(vt_symbol, 0.0)
        if pending > 0:
            self.failed_buyback_count += 1
            logger.warning("[%s] %s 当日 T0 卖出 %s 股未买回，底仓已减少",
                           self.strategy_name, vt_symbol, pending)
