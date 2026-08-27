"""双均线金叉死叉。

## 这是什么策略

快线上穿慢线（金叉）买入，下穿（死叉）清仓。技术分析里最基础的一个，
也是最容易被过拟合的一个 —— 两个整数参数，网格一扫总能找到「好看」的组合。

## 与 TrendMaStrategy / IndexTimingStrategy 的区别

三者都用均线，但判断依据不同：

| 策略 | 信号 |
|------|------|
| MaCrossStrategy（本模块） | 快线 vs **慢线**（两条均线互相穿越） |
| TrendMaStrategy | 价格 vs 周线，再用年线做方向过滤 |
| IndexTimingStrategy | 价格 vs 单均线，带缓冲带与确认天数 |

「均线穿均线」比「价格穿均线」滞后更多 —— 快线本身已经是平滑过的。
好处是假信号少，代价是转折点认得晚。

## 已知弱点：横盘震荡

均线策略的通病。价格在两线附近反复穿越时每次都产生一买一卖，
成本累积而收益为零。本模块**没有**缓冲带和确认天数机制
（那是 IndexTimingStrategy 的做法）—— 保持它的简单性，
需要防打脸时请直接用 IndexTimingStrategy。

⚠ 本策略不构成投资建议，参数极易过拟合，实盘前务必做参数平原检验。
"""
import logging

from ..core.objects import BarData
from .base import StrategyBase

logger = logging.getLogger(__name__)


class MaCrossStrategy(StrategyBase):
    """双均线金叉死叉"""

    parameters = ["fast_window", "slow_window", "position_ratio",
                  "price_buffer", "exit_price_buffer"]
    variables = ["inited", "trading", "pos", "fast_ma", "slow_ma",
                 "trade_count"]

    #: 快线窗口
    fast_window: int = 5
    #: 慢线窗口
    slow_window: int = 20
    #: 单次买入占用可用资金比例
    position_ratio: float = 0.95
    #: 买入限价缓冲。信号在 T 日收盘产生、T+1 开盘成交，需覆盖跳空
    price_buffer: float = 0.03
    #: **卖出限价缓冲，必须不小于买入侧**。
    #: 买卖缓冲天然不对称：错过买入只是少赚，错过卖出是实亏。
    #: 撮合价取开盘价而非限价，放宽缓冲不恶化成交价，只提高成交概率
    exit_price_buffer: float = 0.08

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._coerce_types()
        self._validate()

        # 每个标的独立维护收盘价窗口
        self.closes: dict[str, list[float]] = {s: [] for s in vt_symbols}
        self.fast_ma: dict[str, float] = {}
        self.slow_ma: dict[str, float] = {}
        self._prev_diff: dict[str, float] = {}
        #: 买卖次数。窗口越短越高，是成本的直接来源
        self.trade_count: int = 0

    def _coerce_types(self) -> None:
        """整数参数强制转型。参数寻优会从 DataFrame 传入 float，
        用作切片下标时会 TypeError，且常被外层 try/except 吞掉，
        表现为整个参数网格静默跑空"""
        self.fast_window = int(self.fast_window)
        self.slow_window = int(self.slow_window)
        for name in ("position_ratio", "price_buffer", "exit_price_buffer"):
            setattr(self, name, float(getattr(self, name)))

    def _validate(self) -> None:
        if self.fast_window < 1:
            raise ValueError(f"fast_window 至少为 1，实际 {self.fast_window}")
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

    # ------------------------------------------------------------ 生命周期

    def on_init(self) -> None:
        self.write_log(f"初始化：fast=MA{self.fast_window} "
                       f"slow=MA{self.slow_window}")
        # 预热均线窗口。不预热的话盘中重启后要等 slow_window 根 Bar 才有信号。
        # 回测引擎的 load_bars 是空实现（数据已一次性装载），实盘才真正读历史
        self.load_bars(self.slow_window + 10)

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        self.write_log(f"策略停止。累计买卖 {self.trade_count} 次")

    # ------------------------------------------------------------ 主循环

    def on_bar(self, bar: BarData) -> None:
        if bar.suspended or bar.close_price <= 0:
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
        # 首根有效 Bar 没有前值，无从判断穿越方向
        if prev_diff is None:
            return

        pos = self.get_pos(vt_symbol)

        # 金叉且空仓 → 买入
        if prev_diff <= 0 < diff and pos == 0:
            self._open(vt_symbol, bar)
        # 死叉且有仓 → 清仓
        elif prev_diff >= 0 > diff and pos > 0:
            self._close(vt_symbol, bar, pos)

    def _open(self, vt_symbol: str, bar: BarData) -> None:
        cash = self.get_cash() * self.position_ratio
        # 整手约束由引擎统一处理：策略自行取整会让「预算不足一手」
        # 的情况被静默丢弃，引擎统计不到，报告里也看不到
        volume = cash / bar.close_price
        if volume <= 0:
            return
        # 只在委托真正发出后才记日志：预热期 trading=False、
        # 或被风控拦截时都会返回空，此时打「已买入」会误导排查
        if self.buy(vt_symbol, bar.close_price * (1 + self.price_buffer), volume):
            self.trade_count += 1
            self.write_log(f"金叉买入 {vt_symbol} {volume}股 "
                           f"@{bar.close_price:.2f}")

    def _close(self, vt_symbol: str, bar: BarData, pos: float) -> None:
        price = bar.close_price * (1 - self.exit_price_buffer)
        if self.sell(vt_symbol, price, pos):
            self.trade_count += 1
            self.write_log(f"死叉卖出 {vt_symbol} {pos}股 "
                           f"@{bar.close_price:.2f}")
