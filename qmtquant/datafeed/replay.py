"""行情回放 —— 把本地历史 K 线按时序推入事件引擎。

## 为什么需要

`SimGateway` 只**消费** EVENT_BAR/EVENT_TICK 做撮合，不产生行情。
所以 `run_live.py --gateway sim` 起来之后，引擎、策略、对账全都正常，
然后一根 bar 都不会来 —— 策略一次都不被调用，一笔单都没有，
而且日志里没有任何异常。这是最难查的那种「正常」。

回放器补上这个源。相比等开盘用实时行情，它有几个实际好处：

- **收盘后也能跑**，不用等到第二天 09:30
- **可重复**：同一段行情跑两遍结果一致，改了策略能直接对比
- **可加速**：一天的行情几秒钟跑完，不用真等 4 小时

## 与真实行情的差别（必须知道）

回放推的是**已收口的 K 线**，没有盘口、没有逐笔。所以：

- tick 级策略在回放下不会被触发（它们等的是 `on_tick`）
- 撮合用 bar 的收盘价，没有滑点与排队 —— 成交率会比实盘乐观
- 涨跌停、停牌只能靠 bar 数据自身反映

回放验证的是**链路通不通、策略逻辑对不对**，不是绩效。
拿回放的收益当预期是自欺欺人 —— 那和回测的乐观偏差是同一个来源。
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from ..core.constants import Exchange, Interval
from ..core.objects import BarData
from ..event.engine import EVENT_BAR, Event, EventEngine

logger = logging.getLogger(__name__)


class BarReplayFeeder:
    """按时序回放本地 1m/1d K 线。

    :param speed: 加速倍数。0 表示不等待（尽快跑完）；
                  1 表示与真实时间同速；60 表示 1 分钟行情用 1 秒
    """

    def __init__(self, event_engine: EventEngine, store_dir: str | Path,
                 vt_symbols: list[str], interval: str = "1m",
                 start: str | None = None, end: str | None = None,
                 speed: float = 0.0) -> None:
        self.event_engine = event_engine
        self.store_dir = Path(store_dir)
        self.vt_symbols = list(vt_symbols)
        self.interval = interval
        self.start = start
        self.end = end
        self.speed = speed

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sections: list[tuple] = []
        self.n_bars = 0
        self.n_sections = 0

    # ------------------------------------------------------------ 装载

    def _bar_file(self, vt_symbol: str) -> Path | None:
        """标的的 parquet 路径。优先清洗层。"""
        try:
            code, ex = vt_symbol.split(".")
        except ValueError:
            return None
        for base in (self.store_dir / "clean" / self.interval,
                     self.store_dir / self.interval):
            p = base / ex / f"{code}.parquet"
            if p.exists():
                return p
        return None

    def load(self) -> int:
        """读取并按时刻组织成横截面。返回横截面数。"""
        import pandas as pd

        by_time: dict[datetime, dict[str, BarData]] = {}
        missing = []

        for vt in self.vt_symbols:
            p = self._bar_file(vt)
            if p is None:
                missing.append(vt)
                continue
            try:
                df = pd.read_parquet(p)
            except (OSError, ValueError) as e:
                logger.warning("读取 %s 失败: %s", vt, e)
                continue

            if self.start:
                df = df[df.index >= self.start]
            if self.end:
                df = df[df.index <= pd.Timestamp(self.end)
                        + pd.Timedelta(days=1)]
            if df.empty:
                continue

            code, ex = vt.split(".")
            try:
                exchange = Exchange[ex]
            except KeyError:
                logger.warning("未知交易所 %s，跳过 %s", ex, vt)
                continue
            iv = Interval.MINUTE if self.interval == "1m" else Interval.DAILY

            for dt, row in df.iterrows():
                bar = BarData(
                    symbol=code, exchange=exchange,
                    datetime=dt.to_pydatetime(), interval=iv,
                    open_price=float(row.get("open", 0)),
                    high_price=float(row.get("high", 0)),
                    low_price=float(row.get("low", 0)),
                    close_price=float(row.get("close", 0)),
                    volume=float(row.get("volume", 0)),
                    turnover=float(row.get("amount", 0)),
                    gateway_name="REPLAY",
                )
                by_time.setdefault(bar.datetime, {})[vt] = bar
                self.n_bars += 1

        if missing:
            logger.warning("%d 只标的没有本地 %s 数据（如 %s），"
                           "回放中它们不会出现",
                           len(missing), self.interval, missing[:3])

        self._sections = sorted(by_time.items())
        self.n_sections = len(self._sections)
        logger.info("回放装载完成：%d 只标的、%d 根 bar、%d 个时刻",
                    len(self.vt_symbols) - len(missing),
                    self.n_bars, self.n_sections)
        return self.n_sections

    # ------------------------------------------------------------ 回放

    def start_replay(self) -> None:
        if self._thread is not None:
            return
        if not self._sections:
            logger.error("没有可回放的数据，先调用 load()")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="BarReplay",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        first_dt = self._sections[0][0]
        last_dt = self._sections[-1][0]
        logger.info("开始回放 %s ~ %s（%d 个时刻，speed=%s）",
                    first_dt, last_dt, self.n_sections,
                    "尽快" if self.speed <= 0 else f"{self.speed}x")

        prev_dt = None
        for i, (dt, section) in enumerate(self._sections, 1):
            if self._stop.is_set():
                logger.info("回放已中止（%d/%d）", i, self.n_sections)
                return

            # 按真实时间间隔等待。speed<=0 时不等，尽快跑完
            if self.speed > 0 and prev_dt is not None:
                gap = (dt - prev_dt).total_seconds() / self.speed
                if gap > 0:
                    # 用 wait 而非 sleep，停止时能立刻响应
                    if self._stop.wait(min(gap, 60)):
                        return
            prev_dt = dt

            for bar in section.values():
                self.event_engine.put(Event(EVENT_BAR, bar))

            if i % 500 == 0 or i == self.n_sections:
                logger.info("回放进度 %d/%d（%s）", i, self.n_sections, dt)

        logger.info("回放结束：共 %d 个时刻、%d 根 bar",
                    self.n_sections, self.n_bars)
