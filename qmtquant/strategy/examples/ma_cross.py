"""双均线择时策略示例。

逻辑：快线上穿慢线全仓买入，下穿清仓。
仅用于验证框架链路，**不构成投资建议，实盘前请自行做样本外验证**。
"""
from ..base import StrategyBase


class MaCrossStrategy(StrategyBase):
    """双均线金叉死叉"""

    parameters = ["fast_window", "slow_window", "position_ratio", "price_buffer"]
    variables = ["inited", "trading", "pos", "fast_ma", "slow_ma"]

    fast_window: int = 5
    slow_window: int = 20
    #: 单次买入占用可用资金比例
    position_ratio: float = 0.95
    #: 限价单缓冲，同 TrendMaStrategy
    price_buffer: float = 0.03

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        # 每个标的独立维护收盘价窗口
        self.closes: dict[str, list[float]] = {s: [] for s in vt_symbols}
        self.fast_ma: dict[str, float] = {}
        self.slow_ma: dict[str, float] = {}
        self._prev_diff: dict[str, float] = {}

    def on_init(self) -> None:
        self.write_log(f"初始化：fast={self.fast_window} slow={self.slow_window}")
        # 预热均线窗口。不预热的话盘中重启后要等 slow_window 根 Bar 才能出信号。
        # 回测引擎的 load_bars 是空实现（数据已一次性装载），实盘才真正读历史。
        self.load_bars(self.slow_window + 10)

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        self.write_log("策略停止")

    def on_bar(self, bar) -> None:
        if bar.suspended:
            return

        vt_symbol = bar.vt_symbol
        closes = self.closes.setdefault(vt_symbol, [])
        closes.append(bar.close_price)
        # 只保留计算所需长度，避免长期运行内存增长
        if len(closes) > self.slow_window + 5:
            del closes[0]
        if len(closes) < self.slow_window:
            return

        fast = sum(closes[-self.fast_window:]) / self.fast_window
        slow = sum(closes[-self.slow_window:]) / self.slow_window
        self.fast_ma[vt_symbol] = fast
        self.slow_ma[vt_symbol] = slow

        diff = fast - slow
        prev_diff = self._prev_diff.get(vt_symbol)
        self._prev_diff[vt_symbol] = diff
        if prev_diff is None:
            return

        pos = self.get_pos(vt_symbol)

        # 金叉且空仓 → 买入
        if prev_diff <= 0 < diff and pos == 0:
            cash = self.get_cash() * self.position_ratio
            volume = int(cash / bar.close_price // 100) * 100
            if volume > 0:
                # 用略高于收盘的限价，提高次日开盘成交概率
                # 只在委托真正发出后才记日志：预热期 trading=False、
                # 或被风控拦截时都会返回空，此时打「已买入」会误导排查
                if self.buy(vt_symbol, bar.close_price * (1 + self.price_buffer), volume):
                    self.write_log(f"金叉买入 {vt_symbol} {volume}股 @{bar.close_price:.2f}")

        # 死叉且有仓 → 清仓
        elif prev_diff >= 0 > diff and pos > 0:
            if self.sell(vt_symbol, bar.close_price * (1 - self.price_buffer), pos):
                self.write_log(f"死叉卖出 {vt_symbol} {pos}股 @{bar.close_price:.2f}")
