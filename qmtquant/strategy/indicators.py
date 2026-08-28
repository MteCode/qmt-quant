"""策略常用指标。

均线窗口按 A 股习惯命名：
- 周线 = MA5   （一周 5 个交易日）
- 月线 = MA20
- 季线 = MA60
- 半年线 = MA120
- 年线 = MA250
"""
from collections import deque

MA_WEEK = 5
MA_MONTH = 20
MA_QUARTER = 60
MA_HALF_YEAR = 120
MA_YEAR = 250


class MovingAverage:
    """滚动均线。用 deque + 增量求和，避免每根 Bar 重算整个窗口。"""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("均线窗口必须 >= 1")
        self.window = window
        self._values: deque[float] = deque(maxlen=window)
        self._sum: float = 0.0

    def update(self, value: float) -> float | None:
        """推入新值，返回当前均线；数据不足时返回 None"""
        if len(self._values) == self.window:
            self._sum -= self._values[0]
        self._values.append(value)
        self._sum += value
        return self.value

    @property
    def value(self) -> float | None:
        if len(self._values) < self.window:
            return None
        return self._sum / self.window

    @property
    def ready(self) -> bool:
        return len(self._values) == self.window

    def reset(self) -> None:
        self._values.clear()
        self._sum = 0.0


class CrossDetector:
    """上穿/下穿检测。

    单独抽出来是因为穿越判断极易写错：
    必须比较**前一根**的相对位置与**当前**的相对位置，
    只看当前值大于均线会导致持续出信号，而非只在穿越那一刻出。
    """

    def __init__(self) -> None:
        self._prev_diff: float | None = None

    def update(self, value: float, reference: float) -> str:
        """:return: 'up' 上穿 / 'down' 下穿 / '' 无穿越"""
        diff = value - reference
        prev = self._prev_diff
        self._prev_diff = diff

        if prev is None:
            return ""
        if prev <= 0 < diff:
            return "up"
        if prev >= 0 > diff:
            return "down"
        return ""

    @property
    def above(self) -> bool | None:
        """当前是否在参考线上方"""
        if self._prev_diff is None:
            return None
        return self._prev_diff > 0

    def reset(self) -> None:
        self._prev_diff = None


class IntradayVwap:
    """日内分时均价线（KMCP / VWAP）。

    含义与看盘软件的分时黄线一致：当日累计成交额 ÷ 当日累计成交量。
    **每个交易日开盘必须重置**，否则会把昨天的成交混进来。

    ## 为什么不直接用行情里的成交额字段

    踩过的坑：本地存的价格是**后复权**的，而成交额字段是**未复权**的原始值。
    用 ``turnover / volume`` 算出来的均价落在未复权价格空间，
    与后复权的收盘价根本不可比 —— 实测茅台收盘价 8137、
    ``turnover/volume`` 只有 1300，比值恒为 6.26（正是复权因子）。

    后果是收盘价永远"在均价线之上"，穿越永远不会发生，
    策略**静默零成交**：不报错、不告警，回测结果一片空白还看不出原因。

    因此本类改为累加 ``价格 × 成交量``，价格由调用方传入 ——
    传什么价格空间，算出来就是什么价格空间，与收盘价天然一致。
    分钟 Bar 内价格波动极小，用收盘价近似该 Bar 的成交均价误差可忽略。
    """

    def __init__(self) -> None:
        self._date = None
        self._amount: float = 0.0
        self._volume: float = 0.0

    def update(self, dt, price: float, volume: float) -> float | None:
        """推入一根 Bar。

        :param price: 该 Bar 的代表价格，**必须与后续比较用的价格同一复权口径**。
            通常传 ``bar.close_price``。
        :param volume: 该 Bar 成交量（股）
        """
        day = dt.date()
        if self._date != day:
            self._date = day
            self._amount = 0.0
            self._volume = 0.0

        if volume <= 0 or price <= 0:
            return self.value

        self._amount += price * volume
        self._volume += volume
        return self.value

    @property
    def value(self) -> float | None:
        if self._volume <= 0:
            return None
        return self._amount / self._volume

    @property
    def is_new_day(self) -> bool:
        return self._volume == 0


class AverageTrueRange:
    """ATR（平均真实波幅）。

    ## 为什么止损要用 ATR 而不是固定百分比

    「跌 5% 止损」对不同标的含义完全不同：某只票日常波动就有 4%，
    5% 止损等于每周被扫两次；另一只日波动 0.8%，5% 止损形同虚设。

    ATR 把止损距离统一到「几倍日常波动」这个尺度上，
    跨标的、跨时期才可比。海龟交易法则用的就是 2×ATR。

    ## 真实波幅而非当日振幅

    TR = max(今日最高−今日最低, |今日最高−昨收|, |今日最低−昨收|)

    后两项是为了把**跳空**算进去 —— 只看当日振幅的话，
    低开 5% 然后横盘一天，波幅会被算成接近 0，而实际风险极大。
    """

    def __init__(self, window: int = 14) -> None:
        if window < 1:
            raise ValueError(f"window 至少为 1，实际 {window}")
        self.window = int(window)
        self._trs: deque = deque(maxlen=self.window)
        self._prev_close: float | None = None

    def update(self, high: float, low: float, close: float) -> float | None:
        """推入一根 Bar，返回当前 ATR（数据不足时 None）"""
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low,
                     abs(high - self._prev_close),
                     abs(low - self._prev_close))
        self._prev_close = close
        self._trs.append(tr)
        return self.value

    @property
    def value(self) -> float | None:
        if len(self._trs) < self.window:
            return None
        return sum(self._trs) / self.window

    @property
    def ready(self) -> bool:
        return len(self._trs) >= self.window

    def reset(self) -> None:
        self._trs.clear()
        self._prev_close = None


class Donchian:
    """唐奇安通道：过去 N 根 Bar 的最高价与最低价。

    突破策略的入场信号来源。注意**不含当根 Bar** ——
    含了的话「突破 N 日新高」会变成恒真（今天的最高价当然
    是包含今天在内的最高价之一），信号永远触发。
    """

    def __init__(self, window: int = 20) -> None:
        if window < 2:
            raise ValueError(f"window 至少为 2，实际 {window}")
        self.window = int(window)
        self._highs: deque = deque(maxlen=self.window)
        self._lows: deque = deque(maxlen=self.window)

    def update(self, high: float, low: float) -> None:
        """推入一根 Bar。**先取值再 update**，否则通道会含入当根"""
        self._highs.append(high)
        self._lows.append(low)

    @property
    def upper(self) -> float | None:
        return max(self._highs) if len(self._highs) >= self.window else None

    @property
    def lower(self) -> float | None:
        return min(self._lows) if len(self._lows) >= self.window else None

    @property
    def ready(self) -> bool:
        return len(self._highs) >= self.window
