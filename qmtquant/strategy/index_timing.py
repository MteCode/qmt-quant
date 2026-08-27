"""时序趋势跟随（指数择时）。

## 这是什么策略

不选股，只判断**要不要持有市场**。趋势向上时满仓持有指数 ETF，
趋势向下时空仓。学界称时序动量（time-series momentum），
业内俗称均线择时。

与横截面动量的区别是根本性的：

| | 横截面动量 | 时序趋势（本策略） |
|---|---|---|
| 比什么 | 股票之间互相比 | 指数跟自己的历史比 |
| 决定 | 买哪些股票 | 要不要持有市场 |
| 空仓 | 永远满仓，只换标的 | 趋势向下时空仓 |
| 收益来源 | 赌个股跑赢同行 | 躲开系统性下跌 |

## 为什么它可能还有效

实测沪深300 的横截面动量 IC 为负（-0.058，t=-4.6），已被套利掉。
但时序趋势不同：**机构结构性做不了空仓**。基金经理踏空半年会被赎回，
所以「跌破均线就清仓」这个动作，制度上就轮不到他们做。
这是少数不容易被机构套利掉的边际。

## 为什么它不受成分股偏差影响

交易标的是沪深300 ETF，信号来自指数点位本身，与成分股是谁无关。
选股策略必须知道「2021 年哪 300 只在指数里」，本策略不需要 ——
这是它在当前数据条件下唯一可信的原因。

## 核心风险：来回打脸（whipsaw）

均线择时最大的敌人不是大跌，是横盘震荡。价格在均线上下反复穿越，
每次穿越都产生一次买卖，交易成本累积而收益为零。
实测沪深300 上 MA20 择时反而亏 29%（比买入持有还差），正是死于此。

三道防护：
1. `band` 缓冲带 —— 必须偏离均线超过一定幅度才触发，过滤贴线摩擦
2. `confirm_days` 确认天数 —— 连续 N 日满足条件才动手，过滤单日假突破
3. `min_holding_days` 最短持有 —— 刚建仓就反向的信号不予理会

三者都是用「反应慢一点」换「少交易」，代价是转折点会晚几天。

⚠ 本策略不构成投资建议。择时策略在不同市场状态下表现差异极大，
上线前请自行做样本外与参数稳健性验证。
"""
import logging
from collections import deque

from ..core.objects import BarData
from .base import StrategyBase

logger = logging.getLogger(__name__)

SIGNAL_LONG = "持有"
SIGNAL_FLAT = "空仓"


class IndexTimingStrategy(StrategyBase):
    """均线择时：站上均线持有，跌破空仓"""

    parameters = [
        "ma_window", "band", "confirm_days", "min_holding_days",
        "position_ratio", "price_buffer", "exit_price_buffer",
    ]
    variables = ["inited", "trading", "pos", "signal", "ma_value",
                 "switch_count"]

    #: 均线窗口（交易日）
    ma_window: int = 60
    #: 缓冲带。价格需高于均线 (1+band) 才买入、低于 (1-band) 才卖出。
    #: 0 表示不设缓冲，贴着均线来回穿越会产生大量无效交易
    band: float = 0.01
    #: 确认天数：连续满足条件多少天才动手，过滤单日假突破
    confirm_days: int = 2
    #: 最短持有天数：刚建仓就反向的信号不予理会
    min_holding_days: int = 5
    #: 建仓占用的资金比例。留一点缓冲，避免因价格跳动导致买单资金不足
    position_ratio: float = 0.95
    #: 买入限价缓冲。信号在收盘产生、次日开盘成交，需覆盖跳空。
    #: 错过一次买入只是少赚，所以可以给得紧一点
    price_buffer: float = 0.02
    #: **卖出限价缓冲，必须远大于买入侧**。
    #:
    #: 这是实测踩出来的：原先买卖都用 2%，遇到 -3%/日的崩盘时，
    #: 卖单限价（收盘×0.98）高于次日开盘价（收盘×0.97），委托被拒 ——
    #: 恰恰在最需要离场的时候离不掉，择时反而跑输买入持有。
    #:
    #: 买卖缓冲天然不对称：错过买入只是少赚，错过卖出是实亏。
    #: 注意撮合价取的是开盘价而非限价，放宽缓冲不会恶化成交价，
    #: 只是提高成交概率 —— 等价于「跌停价挂单」的实盘做法。
    exit_price_buffer: float = 0.08

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._coerce_types()
        self._validate()

        if len(vt_symbols) != 1:
            raise ValueError(
                f"择时策略只交易单一标的（ETF），收到 {len(vt_symbols)} 个")
        self.vt_symbol = vt_symbols[0]

        self.closes: deque = deque(maxlen=self.ma_window + 5)
        #: 当前信号，也是持久化后恢复的关键状态
        self.signal: str = SIGNAL_FLAT
        self.ma_value: float | None = None
        #: 信号切换次数，换手过高说明缓冲带/确认天数设得太松
        self.switch_count: int = 0

        self._bar_count: int = 0
        #: 连续满足买入/卖出条件的天数
        self._long_streak: int = 0
        self._flat_streak: int = 0
        #: 最近一次改变持仓的 Bar 序号，用于最短持有约束
        self._last_action_bar: int = -10_000

    def _coerce_types(self) -> None:
        """整数参数强制转型。参数寻优会从 DataFrame 传入 float，
        用作 deque(maxlen=) 或比较时会出问题。"""
        for name in ("ma_window", "confirm_days", "min_holding_days"):
            setattr(self, name, int(getattr(self, name)))
        for name in ("band", "position_ratio", "price_buffer",
                     "exit_price_buffer"):
            setattr(self, name, float(getattr(self, name)))

    def _validate(self) -> None:
        if self.ma_window < 5:
            raise ValueError("ma_window 至少 5")
        if not 0 <= self.band < 0.5:
            raise ValueError("band 必须在 [0, 0.5) 之间")
        if self.confirm_days < 1:
            raise ValueError("confirm_days 至少为 1")
        if self.min_holding_days < 0:
            raise ValueError("min_holding_days 不能为负")
        if self.exit_price_buffer < self.price_buffer:
            raise ValueError(
                f"卖出缓冲必须不小于买入缓冲（错过卖出是实亏，错过买入只是少赚），"
                f"实际为 {self.exit_price_buffer} < {self.price_buffer}")

    # ------------------------------------------------------------ 生命周期

    def on_init(self) -> None:
        self.write_log(
            f"初始化 标的={self.vt_symbol} MA{self.ma_window} "
            f"缓冲带={self.band:.1%} 确认{self.confirm_days}日 "
            f"最短持有{self.min_holding_days}日")
        # 预热：不预热的话实盘要等 ma_window 根 Bar 才有信号
        self.load_bars(self.ma_window + 20)

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        self.write_log(f"策略停止。信号切换 {self.switch_count} 次，"
                       f"当前信号={self.signal}")

    # ------------------------------------------------------------ 主循环

    def on_bar(self, bar: BarData) -> None:
        if bar.vt_symbol != self.vt_symbol:
            return
        if bar.suspended or bar.close_price <= 0:
            return

        self._bar_count += 1
        self.closes.append(bar.close_price)

        ma = self.compute_ma()
        if ma is None:
            return
        self.ma_value = ma

        self._update_streaks(bar.close_price, ma)

        if not self.trading:
            return

        target = self._resolve_signal()
        if target != self.signal:
            self._switch_to(target, bar)

    def compute_ma(self) -> float | None:
        """均线。数据不足返回 None。"""
        if len(self.closes) < self.ma_window:
            return None
        window = list(self.closes)[-self.ma_window:]
        return sum(window) / self.ma_window

    def _update_streaks(self, price: float, ma: float) -> None:
        """更新连续满足条件的天数。

        缓冲带在这里生效：价格落在 [ma*(1-band), ma*(1+band)] 之间时
        两个计数都清零 —— 既不算站上也不算跌破，避免贴线来回触发。
        """
        upper = ma * (1 + self.band)
        lower = ma * (1 - self.band)

        if price > upper:
            self._long_streak += 1
            self._flat_streak = 0
        elif price < lower:
            self._flat_streak += 1
            self._long_streak = 0
        else:
            self._long_streak = 0
            self._flat_streak = 0

    def _resolve_signal(self) -> str:
        """由连续天数决定目标信号，未达确认天数则维持现状"""
        if self._long_streak >= self.confirm_days:
            return SIGNAL_LONG
        if self._flat_streak >= self.confirm_days:
            return SIGNAL_FLAT
        return self.signal

    def _switch_to(self, target: str, bar: BarData) -> None:
        """切换持仓。最短持有期内的反向信号一律忽略。"""
        held = self._bar_count - self._last_action_bar
        if held < self.min_holding_days:
            logger.debug("%s 距上次动作仅 %d 日，忽略切换至 %s",
                         self.vt_symbol, held, target)
            return

        if target == SIGNAL_LONG:
            if self.get_pos(self.vt_symbol) > 0:
                self.signal = target
                return
            cash = self.get_cash() * self.position_ratio
            volume = cash / bar.close_price
            price = bar.close_price * (1 + self.price_buffer)
            if not self.buy(self.vt_symbol, price, volume):
                return
            self.write_log(f"买入 @{bar.close_price:.3f} "
                           f"MA{self.ma_window}={self.ma_value:.3f}")
        else:
            volume = self.get_pos(self.vt_symbol)
            if volume <= 0:
                self.signal = target
                return
            price = bar.close_price * (1 - self.exit_price_buffer)
            if not self.sell(self.vt_symbol, price, volume):
                return
            self.write_log(f"清仓 @{bar.close_price:.3f} "
                           f"MA{self.ma_window}={self.ma_value:.3f}")

        self.signal = target
        self.switch_count += 1
        self._last_action_bar = self._bar_count
