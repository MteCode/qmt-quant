"""横截面均值回归策略。

## 核心逻辑

在一篮子股票中，找出相对**自身**近期均值跌得最多的（z-score 最低），
买入，等价格回归到均值附近卖出。

为什么用 z-score 而不是跌幅百分比：不同股票的波动率差好几倍。
一只年化波动 15% 的银行股跌 5% 是罕见事件，一只波动 60% 的题材股
跌 5% 是家常便饭。不做波动率标准化，选出来的永远是高波动股票，
本质上是在赌波动率而不是赌回归。

## 均值回归的四种典型死法，以及本策略的应对

1. **在单边下跌里越买越套** —— 最常见。「跌多了会反弹」在趋势下跌中不成立。
   → `trend_filter_window` 长期均线过滤，只在均线上方做多。

2. **不设止损，一只股票吃掉全部利润** —— 均值回归的收益分布是
   「大量小赚 + 偶尔巨亏」，没有止损时那个巨亏足以抹平几十次小赚。
   → `stop_z` 止损：z-score 继续恶化到该阈值就认输离场。

3. **买了不回归，仓位被无限期占用** —— 资金锁死在不动的票上，
   错过其他机会，且回撤持续扩大。
   → `max_holding_days` 持有期上限，超时无条件退出。

4. **在流动性差的票上被滑点吃穿** —— 回归策略单次预期收益本就不高，
   几个点的冲击成本足以让正期望变负期望。
   → `min_turnover` 成交额下限过滤。

## A 股特有约束

- **T+1**：当日买入次日才可卖，因此持有期至少 1 天，
  不存在「日内买入后立刻止损」的可能，止损最快也是次日执行。
- **涨跌停无法成交**：跌停时想买买不到、涨停时想卖卖不掉。
  信号阶段就跳过这些标的，避免产生注定无法执行的委托。
- **一手 100 股**：整手约束由引擎统一处理，策略不自行取整。

## 信号稀少是设计使然，不是 bug

趋势过滤与 z 阈值互相拉扯：跌得够深让 z 到 -2，往往同时也跌破了均线。
实测（涨 1%/日的上行趋势，MA20 趋势线，lookback=10）：

| 回调幅度 | z-score | 仍在趋势线上 |
|---------|---------|-------------|
| 5% | -0.43 | 是 |
| 8% | -1.40 | 是 |
| 10% | -1.85 | **否** |
| 15% | -2.46 | **否** |

所以 `entry_z=-2.0` 配 `trend_filter_window=120` 时信号会很少 ——
这正是想要的：宁可少做，也不在下跌趋势里接飞刀。
若觉得信号太少，应放宽 `entry_z`（如 -1.2），而不是关掉趋势过滤 ——
关掉过滤等于把死法一重新打开。

⚠ 本策略不构成投资建议。均值回归在 A 股的有效性随市场状态变化，
上线前请自行做样本外验证与参数稳健性检查。
"""
import logging
from collections import defaultdict, deque

from ..core.constants import get_price_limit
from ..core.objects import BarData
from .base import StrategyBase

logger = logging.getLogger(__name__)

#: 退出原因，用于事后归因 —— 分不清是「正常回归」还是「止损」的话，
#: 无法判断策略是靠回归赚钱还是靠运气躲过止损
EXIT_REVERT = "回归"
EXIT_STOP = "止损"
EXIT_TIMEOUT = "超时"
EXIT_TREND = "跌破趋势线"


class MeanReversionStrategy(StrategyBase):
    """横截面均值回归"""

    parameters = [
        "lookback", "entry_z", "exit_z", "stop_z",
        "max_holding_days", "trend_filter_window", "min_turnover",
        "max_holdings", "position_ratio", "price_buffer",
    ]
    variables = ["inited", "trading", "pos", "entry_info", "exit_stats"]

    #: 均值与标准差的计算窗口（交易日）
    lookback: int = 20
    #: 买入阈值：z-score 低于该值视为超跌。
    #: 取值越负越严格，信号越少但质量越高
    entry_z: float = -2.0
    #: 卖出阈值：z-score 回升到该值以上视为已回归。
    #: 不要设成 0 —— 等完全回到均值往往等不到，且错过了大部分利润
    exit_z: float = -0.5
    #: 止损阈值：z-score 继续跌破该值说明「回归」判断已失效
    stop_z: float = -4.0
    #: 最长持有天数，超时无条件退出
    max_holding_days: int = 20
    #: 趋势过滤窗口。价格需在该均线之上才允许买入，
    #: 也用作持仓的趋势止损（跌破即离场）。
    #: **设为 0 表示关闭过滤**（仅用于对照实验，等于打开死法一）。
    #: 不要用 1 来「关闭」—— 窗口为 1 时均线就是价格自身，
    #: `price > price` 恒为 False，会变成永远不许买入
    trend_filter_window: int = 120
    #: 日成交额下限（元），低于此值不参与 —— 滑点会吃掉回归策略微薄的预期收益
    min_turnover: float = 50_000_000
    #: 最多同时持有几只
    max_holdings: int = 10
    #: 总资金中用于建仓的比例，其余留作缓冲
    position_ratio: float = 0.90
    #: 限价缓冲，见 TrendMaStrategy 的说明
    price_buffer: float = 0.03

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._validate()

        window = max(self.lookback, self.trend_filter_window) + 5
        self.closes: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        #: vt_symbol -> {"bar_count": 建仓时的 Bar 序号, "entry_z": 建仓时 z}
        self.entry_info: dict[str, dict] = {}
        #: 退出原因计数，用于归因
        self.exit_stats: dict[str, int] = defaultdict(int)

        self._bar_count: int = 0
        #: 最近一次计算的 z-score，供报告与调试查看
        self.zscores: dict[str, float] = {}

    def _validate(self) -> None:
        if not (self.stop_z < self.entry_z < self.exit_z):
            raise ValueError(
                f"阈值必须满足 止损 < 买入 < 卖出，实际为 "
                f"{self.stop_z} / {self.entry_z} / {self.exit_z}")
        if self.lookback < 5:
            raise ValueError("lookback 至少 5，否则标准差不稳定")
        if self.trend_filter_window < 0:
            raise ValueError("trend_filter_window 不能为负（0 表示关闭过滤）")
        if self.max_holdings < 1:
            raise ValueError("max_holdings 至少为 1")

    # ------------------------------------------------------------ 生命周期

    def on_init(self) -> None:
        self.write_log(
            f"初始化 lookback={self.lookback} 入场z={self.entry_z} "
            f"出场z={self.exit_z} 止损z={self.stop_z} "
            f"趋势线MA{self.trend_filter_window} 最多持有{self.max_holdings}只")
        # 预热：需要 max(lookback, 趋势窗口) 根 Bar 才能出第一个信号
        self.load_bars(max(self.lookback, self.trend_filter_window) + 20)

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        if self.exit_stats:
            detail = "，".join(f"{k} {v} 次" for k, v in
                              sorted(self.exit_stats.items()))
            self.write_log(f"策略停止。退出原因分布：{detail}")
        else:
            self.write_log("策略停止")

    # ------------------------------------------------------------ 主循环

    def on_bars(self, bars: dict[str, BarData]) -> None:
        self._bar_count += 1
        self._update_indicators(bars)

        if not self.trading:
            return

        # 先处理退出：卖出释放资金后买单才有钱成交，
        # 引擎也会保证卖单优先撮合
        self._handle_exits(bars)
        self._handle_entries(bars)

    def _update_indicators(self, bars: dict[str, BarData]) -> None:
        for vt_symbol, bar in bars.items():
            # 停牌日价格无意义，纳入会污染均值与标准差
            if bar.suspended or bar.close_price <= 0:
                continue
            self.closes[vt_symbol].append(bar.close_price)

    # ------------------------------------------------------------ 指标

    def compute_zscore(self, vt_symbol: str) -> float | None:
        """当前收盘价相对近期均值的标准分。

        标准差为 0（连续多日同价，如长期停牌后复牌）时返回 None ——
        除零会得到 inf，让该标的永远排在信号首位。
        """
        closes = self.closes.get(vt_symbol)
        if closes is None or len(closes) < self.lookback:
            return None

        window = list(closes)[-self.lookback:]
        mean = sum(window) / self.lookback
        var = sum((x - mean) ** 2 for x in window) / self.lookback
        std = var ** 0.5
        if std <= 1e-9:
            return None
        return (window[-1] - mean) / std

    def above_trend(self, vt_symbol: str) -> bool | None:
        """价格是否在长期均线之上。数据不足返回 None（保守视为不满足）。

        窗口 <= 1 表示关闭过滤，直接返回 True。
        """
        if self.trend_filter_window <= 1:
            return True

        closes = self.closes.get(vt_symbol)
        if closes is None or len(closes) < self.trend_filter_window:
            return None
        window = list(closes)[-self.trend_filter_window:]
        return window[-1] > sum(window) / self.trend_filter_window

    # ------------------------------------------------------------ 可交易性

    def _tradable(self, bar: BarData, prev_close: float | None,
                  direction: str) -> bool:
        """该标的当前能否交易。

        涨跌停时无法成交，信号阶段就跳过，避免产生注定失败的委托
        —— 那些委托会污染拒单统计，掩盖真正的执行问题。
        """
        if bar.suspended or bar.close_price <= 0:
            return False
        if bar.turnover < self.min_turnover:
            return False

        if prev_close is None or prev_close <= 0:
            return True

        limit = get_price_limit(bar.vt_symbol)
        change = bar.close_price / prev_close - 1
        if direction == "buy" and change >= limit - 1e-6:
            return False      # 涨停买不到
        if direction == "sell" and change <= -limit + 1e-6:
            return False      # 跌停卖不掉
        return True

    def _prev_close(self, vt_symbol: str) -> float | None:
        closes = self.closes.get(vt_symbol)
        if closes is None or len(closes) < 2:
            return None
        return closes[-2]

    # ------------------------------------------------------------ 退出

    def _handle_exits(self, bars: dict[str, BarData]) -> None:
        for vt_symbol in list(self.entry_info):
            volume = self.get_pos(vt_symbol)
            if volume <= 0:
                self.entry_info.pop(vt_symbol, None)
                continue

            bar = bars.get(vt_symbol)
            if bar is None:
                continue

            reason = self._exit_reason(vt_symbol, bar)
            if reason is None:
                continue

            if not self._tradable(bar, self._prev_close(vt_symbol), "sell"):
                # 跌停或停牌卖不掉，下一根 Bar 再试。
                # 不清 entry_info，否则会忘记这笔仓位仍需退出
                continue

            price = bar.close_price * (1 - self.price_buffer)
            if self.sell(vt_symbol, price, volume):
                self.exit_stats[reason] += 1
                self.entry_info.pop(vt_symbol, None)
                self.write_log(
                    f"卖出 {vt_symbol} {volume:.0f}股 @{bar.close_price:.2f} "
                    f"原因={reason} z={self.zscores.get(vt_symbol, float('nan')):.2f}")

    def _exit_reason(self, vt_symbol: str, bar: BarData) -> str | None:
        """判断是否该退出，以及退出原因。顺序即优先级。"""
        z = self.compute_zscore(vt_symbol)
        if z is not None:
            self.zscores[vt_symbol] = z

        info = self.entry_info.get(vt_symbol, {})

        # 止损优先：判断已经失效，越早认输越好
        if z is not None and z <= self.stop_z:
            return EXIT_STOP

        # 跌破趋势线：买入前提已不成立
        if self.above_trend(vt_symbol) is False:
            return EXIT_TREND

        # 已回归
        if z is not None and z >= self.exit_z:
            return EXIT_REVERT

        # 超时：资金不能被无限期占用
        held = self._bar_count - info.get("bar_count", self._bar_count)
        if held >= self.max_holding_days:
            return EXIT_TIMEOUT

        return None

    # ------------------------------------------------------------ 入场

    def _handle_entries(self, bars: dict[str, BarData]) -> None:
        held = {s for s, v in self.pos.items() if v > 0}
        slots = self.max_holdings - len(held)
        if slots <= 0:
            return

        candidates = self._rank_candidates(bars, held)
        if not candidates:
            return

        # 按目标持仓数均分总资产，而非按剩余现金 ——
        # 后者会让仓位随每次建仓越滚越小
        budget = (self._total_value(bars) * self.position_ratio
                  / self.max_holdings)

        for z, vt_symbol in candidates[:slots]:
            bar = bars[vt_symbol]
            volume = budget / bar.close_price
            price = bar.close_price * (1 + self.price_buffer)
            if self.buy(vt_symbol, price, volume):
                self.entry_info[vt_symbol] = {
                    "bar_count": self._bar_count, "entry_z": z}
                self.write_log(f"买入 {vt_symbol} @{bar.close_price:.2f} "
                               f"z={z:.2f}")

    def _rank_candidates(self, bars: dict[str, BarData],
                         held: set[str]) -> list[tuple[float, str]]:
        """筛出符合条件的标的，按 z-score 升序（越超跌越靠前）"""
        universe = self.engine.get_universe()
        scored: list[tuple[float, str]] = []

        for vt_symbol in universe:
            if vt_symbol in held:
                continue
            bar = bars.get(vt_symbol)
            if bar is None:
                continue
            if not self._tradable(bar, self._prev_close(vt_symbol), "buy"):
                continue
            # 趋势过滤：数据不足时 above_trend 返回 None，一并排除
            if self.above_trend(vt_symbol) is not True:
                continue

            z = self.compute_zscore(vt_symbol)
            if z is None:
                continue
            self.zscores[vt_symbol] = z

            # 已跌破止损线的不接飞刀 —— 买进来下一根就要止损
            if z <= self.stop_z or z > self.entry_z:
                continue
            scored.append((z, vt_symbol))

        scored.sort()
        return scored

    def _total_value(self, bars: dict[str, BarData]) -> float:
        value = self.get_cash()
        for vt_symbol, volume in self.pos.items():
            if volume <= 0:
                continue
            bar = bars.get(vt_symbol)
            if bar is not None and bar.close_price > 0:
                value += volume * bar.close_price
        return value
