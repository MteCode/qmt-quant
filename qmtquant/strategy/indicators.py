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
