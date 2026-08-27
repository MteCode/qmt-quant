"""均线趋势策略（日线）。

用两条均线：
- **年线（MA250）** 定趋势方向 —— 价格在年线上方视为多头格局
- **周线（MA5）** 定买卖时机 —— 穿越周线触发动作

两种模式，用 `mode` 参数切换：

| mode | 逻辑 | 什么行情下有效 |
|------|------|--------------|
| `trend` | 站上买、跌破卖 | 单边趋势市。能避开大跌，震荡市反复打脸 |
| `reversion` | 跌破买、站上卖 | 震荡市。遇到单边下跌会越买越套 |

## 与 IndexTimingStrategy 的分工

两者都是均线择时，但适用对象不同：

| | TrendMaStrategy（本模块） | IndexTimingStrategy |
|---|---|---|
| 标的 | 多只个股，各自独立判断 | 单一指数 ETF |
| 信号 | 双均线（周线穿越 + 年线过滤） | 单均线 + 缓冲带 + 确认天数 |
| 仓位 | 每只满仓进出 | 全市场在场/离场 |
| 防打脸 | 靠年线过滤 | 三道机制（band/confirm/min_hold） |

个股上做均线择时的问题是**资金分配**：多只同时发出买入信号时，
先到的吃掉全部资金。所以本策略更适合单标的或少数几只。

⚠ 本策略不构成投资建议。两种模式都请自行做样本外验证再考虑实盘。
"""
import logging

from .base import StrategyBase
from .indicators import MA_WEEK, MA_YEAR, CrossDetector, MovingAverage

logger = logging.getLogger(__name__)

MODE_TREND = "trend"
MODE_REVERSION = "reversion"


class TrendMaStrategy(StrategyBase):
    """周线/年线均线策略"""

    parameters = ["fast_window", "slow_window", "mode", "position_ratio",
                  "use_trend_filter", "price_buffer", "exit_price_buffer"]
    variables = ["inited", "trading", "pos", "fast_ma_value", "slow_ma_value",
                 "trade_count"]

    #: 周线 = 5 日均线（A 股习惯：一周 5 个交易日）
    fast_window: int = MA_WEEK
    #: 年线 = 250 日均线
    slow_window: int = MA_YEAR
    #: trend 跟随趋势 / reversion 均值回归
    mode: str = MODE_TREND
    #: 单次买入占用可用资金比例
    position_ratio: float = 0.95
    #: 是否要求「年线之上才允许买入」。关掉后只看周线穿越
    use_trend_filter: bool = True
    #: 限价单相对收盘价的缓冲比例。
    #: 信号在 T 日收盘产生，成交在 T+1 开盘 —— 若次日跳空幅度超过缓冲，
    #: 限价单就会失效（买单低于开盘价、卖单高于开盘价）。
    #: 调大提高成交率但滑点变差；调小滑点小但容易踏空。
    price_buffer: float = 0.03
    #: **卖出限价缓冲，必须不小于买入侧**。
    #:
    #: 实测踩出来的：买卖都用 2% 时，遇到 -3%/日的连续跳空下跌，
    #: 卖单限价（收盘×0.98）高于次日开盘（收盘×0.97），委托被拒 ——
    #: 恰恰在最需要离场时离不掉。买卖缓冲天然不对称：
    #: 错过买入只是少赚，错过卖出是实亏。
    #: 撮合价取开盘价而非限价，放宽缓冲不恶化成交价，只提高成交概率。
    exit_price_buffer: float = 0.08

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._coerce_types()
        self._validate()

        self.fast_ma = {s: MovingAverage(self.fast_window) for s in vt_symbols}
        self.slow_ma = {s: MovingAverage(self.slow_window) for s in vt_symbols}
        self.cross = {s: CrossDetector() for s in vt_symbols}
        # 供外部观测的最新值
        self.fast_ma_value: dict[str, float] = {}
        self.slow_ma_value: dict[str, float] = {}
        #: 买卖次数，换手过高说明均线窗口太短
        self.trade_count: int = 0

    def _coerce_types(self) -> None:
        """整数参数强制转型。参数寻优会从 DataFrame 传入 float，
        用作切片下标或 range() 时会 TypeError，且常被外层吞掉"""
        for name in ("fast_window", "slow_window"):
            setattr(self, name, int(getattr(self, name)))
        for name in ("position_ratio", "price_buffer", "exit_price_buffer"):
            setattr(self, name, float(getattr(self, name)))
        self.use_trend_filter = bool(self.use_trend_filter)

    def _validate(self) -> None:
        if self.mode not in (MODE_TREND, MODE_REVERSION):
            raise ValueError(f"未知 mode: {self.mode}，可选 trend / reversion")
        if self.fast_window < 2:
            raise ValueError(f"fast_window 至少为 2，实际 {self.fast_window}")
        if self.slow_window <= self.fast_window:
            raise ValueError(
                f"slow_window 必须大于 fast_window，"
                f"实际 {self.slow_window} <= {self.fast_window}")
        if not 0 < self.position_ratio <= 1:
            raise ValueError("position_ratio 必须在 (0, 1] 之间")
        if self.exit_price_buffer < self.price_buffer:
            raise ValueError(
                f"卖出缓冲必须不小于买入缓冲（错过卖出是实亏，错过买入只是少赚），"
                f"实际 {self.exit_price_buffer} < {self.price_buffer}")

    def on_init(self) -> None:
        self.write_log(f"初始化 mode={self.mode} 周线MA{self.fast_window} "
                       f"年线MA{self.slow_window} 趋势过滤={self.use_trend_filter}")
        # 预热：年线需要 250 根 Bar，不预热的话实盘要等一年才出信号
        self.load_bars(self.slow_window + self.fast_window + 20)

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        self.write_log(f"策略停止。累计买卖 {self.trade_count} 次")

    def on_bar(self, bar) -> None:
        if bar.suspended or bar.close_price <= 0:
            return

        vt_symbol = bar.vt_symbol
        fast_ma = self.fast_ma.get(vt_symbol)
        slow_ma = self.slow_ma.get(vt_symbol)
        if fast_ma is None or slow_ma is None:
            return

        fast = fast_ma.update(bar.close_price)
        slow = slow_ma.update(bar.close_price)
        if fast is None:
            return
        self.fast_ma_value[vt_symbol] = fast
        if slow is not None:
            self.slow_ma_value[vt_symbol] = slow

        # 穿越检测必须每根 Bar 都更新，否则会漏掉穿越点
        signal = self.cross[vt_symbol].update(bar.close_price, fast)

        # 年线未就绪时，若要求趋势过滤则不交易
        if self.use_trend_filter and slow is None:
            return

        want_buy, want_sell = self._resolve(signal, bar.close_price, slow)
        pos = self.get_pos(vt_symbol)

        if want_buy and pos == 0:
            self._open(vt_symbol, bar)
        elif want_sell and pos > 0:
            price = bar.close_price * (1 - self.exit_price_buffer)
            if self.sell(vt_symbol, price, pos):
                self.trade_count += 1
                self.write_log(f"卖出 {vt_symbol} {pos}股 @{bar.close_price:.2f} "
                               f"({self.mode}/{signal})")

    def _resolve(self, signal: str, close: float, slow: float | None) -> tuple[bool, bool]:
        """把穿越信号翻译成买卖意图"""
        if self.mode == MODE_TREND:
            buy, sell = signal == "up", signal == "down"
            # 趋势过滤：只在年线之上做多，年线之下一律离场
            if self.use_trend_filter and slow is not None:
                if close < slow:
                    return False, True
                buy = buy and close > slow
        else:
            # 均值回归：跌破周线视为超跌买入，回升到周线上方卖出
            buy, sell = signal == "down", signal == "up"
            if self.use_trend_filter and slow is not None:
                # 只在年线之上抄底，避免在单边下跌里越买越套
                buy = buy and close > slow
                if close < slow:
                    return False, True
        return buy, sell

    def _open(self, vt_symbol: str, bar) -> None:
        cash = self.get_cash() * self.position_ratio
        # 整手约束交给引擎，见 PortfolioStrategy.rebalance 中的说明
        volume = cash / bar.close_price
        if self.buy(vt_symbol, bar.close_price * (1 + self.price_buffer), volume):
            self.trade_count += 1
            self.write_log(f"买入 {vt_symbol} {volume}股 @{bar.close_price:.2f} "
                           f"({self.mode})")
