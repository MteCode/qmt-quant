"""均线趋势策略（日线）。

用两条均线：
- **年线（MA250）** 定趋势方向 —— 价格在年线上方视为多头格局
- **周线（MA5）** 定买卖时机 —— 穿越周线触发动作

两种模式，用 `mode` 参数切换：

| mode | 逻辑 | 什么行情下有效 |
|------|------|--------------|
| `trend` | 站上买、跌破卖 | 单边趋势市。能避开大跌，震荡市反复打脸 |
| `reversion` | 跌破买、站上卖 | 震荡市。遇到单边下跌会越买越套 |

⚠ 示例策略，用于验证框架链路，**不构成投资建议**。
两种模式都请自行做样本外验证再考虑实盘。
"""
from ..base import StrategyBase
from ..indicators import MA_WEEK, MA_YEAR, CrossDetector, MovingAverage

MODE_TREND = "trend"
MODE_REVERSION = "reversion"


class TrendMaStrategy(StrategyBase):
    """周线/年线均线策略"""

    parameters = ["fast_window", "slow_window", "mode",
                  "position_ratio", "use_trend_filter", "price_buffer"]
    variables = ["inited", "trading", "pos", "fast_ma_value", "slow_ma_value"]

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

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)

        if self.mode not in (MODE_TREND, MODE_REVERSION):
            raise ValueError(f"未知 mode: {self.mode}，可选 trend / reversion")

        self.fast_ma = {s: MovingAverage(self.fast_window) for s in vt_symbols}
        self.slow_ma = {s: MovingAverage(self.slow_window) for s in vt_symbols}
        self.cross = {s: CrossDetector() for s in vt_symbols}
        # 供外部观测的最新值
        self.fast_ma_value: dict[str, float] = {}
        self.slow_ma_value: dict[str, float] = {}

    def on_init(self) -> None:
        self.write_log(f"初始化 mode={self.mode} 周线MA{self.fast_window} "
                       f"年线MA{self.slow_window} 趋势过滤={self.use_trend_filter}")
        # 预热：年线需要 250 根 Bar，不预热的话实盘要等一年才出信号
        self.load_bars(self.slow_window + self.fast_window + 20)

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        self.write_log("策略停止")

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
            if self.sell(vt_symbol, bar.close_price * (1 - self.price_buffer), pos):
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
        volume = int(cash / bar.close_price // 100) * 100
        if volume <= 0:
            return
        if self.buy(vt_symbol, bar.close_price * (1 + self.price_buffer), volume):
            self.write_log(f"买入 {vt_symbol} {volume}股 @{bar.close_price:.2f} "
                           f"({self.mode})")
