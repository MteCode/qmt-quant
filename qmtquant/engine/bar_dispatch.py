"""Bar 分发 —— 补上实盘引擎「只有 tick、没有 bar」的缺口。

## 缺口在哪

`LiveEngine` 只注册了 `EVENT_TICK`，实盘期间靠 `strategy.on_tick()` 驱动。
而 `strategy.on_bars()` **只在预热阶段被调用一次**，且那时 `trading=False`
不会下单。

后果是：所有 bar 驱动的策略 —— `PortfolioStrategy`、`SignalFileStrategy`、
`IntradayGBMStrategy` —— 它们的 `on_bars` 在实盘**永远不会被调用**。
表现为引擎正常启动、策略正常加载、对账正常通过，然后一笔单都没有，
而且日志里没有任何异常。

这个模块补两件事：

1. **BarAggregator**：把 tick 聚合成分钟 bar。实盘只有 tick 流，
   bar 得自己攒。
2. **SectionDispatcher**：把同一时刻各标的的 bar 拼成横截面再分发。
   选股类策略要的是「这一分钟全市场长什么样」，一根一根喂没有意义。

## 横截面何时算「完整」

理论上要等所有订阅标的的 bar 都到齐，但现实中总有停牌、无成交、
数据延迟的标的永远不到。所以用**双条件**触发：

- 收到了下一分钟的 bar（说明这一分钟已经过去）
- 或距该分钟结束超过 `section_timeout` 秒

先到先触发。宁可让横截面缺几只，也不能因为等一只停牌股而整体停摆 ——
后者会让策略在整个交易日里一次都不被调用。
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta

from ..core.constants import Interval
from ..core.objects import BarData, TickData

logger = logging.getLogger(__name__)

#: 横截面等待上限（秒）。超过就带着已有的标的先发出去
SECTION_TIMEOUT = 20


class BarAggregator:
    """把 tick 聚合成分钟 bar。

    每个标的独立维护当前分钟的 OHLCV。跨分钟时收口上一根并返回。
    """

    def __init__(self) -> None:
        self._bars: dict[str, BarData] = {}
        #: 上一根 bar 的累计成交量，用于算本根的增量
        self._last_volume: dict[str, float] = {}
        self._last_turnover: dict[str, float] = {}

    def update(self, tick: TickData) -> BarData | None:
        """喂一个 tick，返回刚收口的上一根 bar（若跨分钟）。"""
        if tick.datetime is None or tick.last_price <= 0:
            return None

        vt = tick.vt_symbol
        minute = tick.datetime.replace(second=0, microsecond=0)
        cur = self._bars.get(vt)

        finished = None
        if cur is not None and cur.datetime != minute:
            finished = cur
            cur = None

        if cur is None:
            cur = BarData(
                symbol=tick.symbol, exchange=tick.exchange,
                datetime=minute, interval=Interval.MINUTE,
                open_price=tick.last_price, high_price=tick.last_price,
                low_price=tick.last_price, close_price=tick.last_price,
                volume=0, turnover=0, gateway_name=tick.gateway_name,
            )
            self._bars[vt] = cur
        else:
            cur.high_price = max(cur.high_price, tick.last_price)
            cur.low_price = min(cur.low_price, tick.last_price)
            cur.close_price = tick.last_price

        # tick 里的 volume 是当日累计，bar 要的是本根增量。
        # 首个 tick 拿不到基准，记下当前值、本根成交量记 0 —— 这样
        # 只是第一根 bar 的量偏小，比用累计值当增量（会大出几个数量级）安全。
        vol, tov = tick.volume or 0, tick.turnover or 0
        if vt in self._last_volume:
            cur.volume = max(0.0, vol - self._last_volume[vt])
            cur.turnover = max(0.0, tov - self._last_turnover[vt])
        if finished is not None or vt not in self._last_volume:
            self._last_volume[vt] = vol
            self._last_turnover[vt] = tov

        return finished

    def flush(self) -> list[BarData]:
        """收口所有未完成的 bar。收盘或停机时调用。"""
        out = list(self._bars.values())
        self._bars.clear()
        return out


class SectionDispatcher:
    """把 bar 按时刻攒成横截面，齐了或超时就交给回调。

    :param on_section: 收到完整横截面时调用，签名 (datetime, {vt_symbol: BarData})
    :param expected: 预期标的数。为 0 表示不按数量判断，只靠跨分钟与超时
    :param timeout: 距该分钟结束多少秒后强制发出
    """

    def __init__(self, on_section, expected: int = 0,
                 timeout: int = SECTION_TIMEOUT) -> None:
        self.on_section = on_section
        self.expected = expected
        self.timeout = timeout
        self._pending: dict[datetime, dict[str, BarData]] = defaultdict(dict)
        #: 已经发出去的时刻，防止迟到的 bar 触发重复分发
        self._done: set[datetime] = set()

    def add(self, bar: BarData) -> None:
        if bar.datetime is None:
            return
        dt = bar.datetime.replace(second=0, microsecond=0)
        if dt in self._done:
            # 迟到的 bar：该时刻已分发过，再发一次会让策略重复决策
            return
        self._pending[dt][bar.vt_symbol] = bar

        # 收到更晚时刻的 bar，说明更早的那些已经不会再有了
        stale = [d for d in self._pending if d < dt]
        for d in sorted(stale):
            self._emit(d)

        if self.expected and len(self._pending[dt]) >= self.expected:
            self._emit(dt)

    def check_timeout(self, now: datetime | None = None) -> None:
        """定时调用。把等太久的横截面先发出去。

        没有这一步的话，最后一分钟的 bar 会一直攒着不发 ——
        收盘前那根往往正是要平仓的那根。
        """
        now = now or datetime.now()
        for dt in sorted(list(self._pending)):
            # 该分钟结束时刻 = dt + 1 分钟
            if (now - (dt + timedelta(minutes=1))).total_seconds() >= self.timeout:
                self._emit(dt)

    def _emit(self, dt: datetime) -> None:
        section = self._pending.pop(dt, None)
        if not section:
            return
        self._done.add(dt)
        # 只留最近的时刻，否则跑一整天会无限增长
        if len(self._done) > 600:
            for d in sorted(self._done)[:200]:
                self._done.discard(d)
        try:
            self.on_section(dt, section)
        except Exception:
            logger.exception("横截面分发异常 dt=%s，%d 只标的", dt, len(section))

    def flush(self) -> None:
        """把所有挂起的横截面立刻发出。停机时调用。"""
        for dt in sorted(list(self._pending)):
            self._emit(dt)
