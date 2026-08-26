"""事件引擎。

单线程事件循环 + 独立定时器线程。所有跨模块通信都经过这里，
模块之间只依赖事件类型字符串，不互相 import。
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Thread
from typing import Any, Callable

logger = logging.getLogger(__name__)

EVENT_TICK = "eTick"
EVENT_BAR = "eBar"
EVENT_ORDER = "eOrder"
EVENT_TRADE = "eTrade"
EVENT_POSITION = "ePosition"
EVENT_ACCOUNT = "eAccount"
EVENT_CONTRACT = "eContract"
EVENT_LOG = "eLog"
EVENT_TIMER = "eTimer"
EVENT_RISK_REJECT = "eRiskReject"
EVENT_GATEWAY_STATUS = "eGatewayStatus"


@dataclass
class Event:
    """事件对象"""
    type: str
    data: Any = None


HandlerType = Callable[[Event], None]


class EventEngine:
    """事件驱动引擎"""

    def __init__(self, timer_interval: float = 1.0) -> None:
        """
        :param timer_interval: 定时事件间隔（秒）
        """
        self._queue: Queue = Queue()
        self._active: bool = False
        self._thread: Thread = Thread(target=self._run, name="EventEngine", daemon=True)
        self._timer: Thread = Thread(target=self._run_timer, name="EventTimer", daemon=True)
        self._timer_interval: float = timer_interval

        self._handlers: dict[str, list[HandlerType]] = defaultdict(list)
        self._general_handlers: list[HandlerType] = []

    # ---------------------------------------------------------------- 生命周期

    def start(self) -> None:
        if self._active:
            return
        self._active = True
        self._thread.start()
        self._timer.start()
        logger.info("事件引擎已启动")

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        self._timer.join(timeout=self._timer_interval + 1)
        self._thread.join(timeout=5)
        logger.info("事件引擎已停止")

    # ---------------------------------------------------------------- 内部循环

    def _run(self) -> None:
        while self._active:
            try:
                event = self._queue.get(block=True, timeout=1)
            except Empty:
                continue
            self._process(event)

    def _process(self, event: Event) -> None:
        # handler 抛异常绝不能终止事件循环，否则整个系统失聪
        for handler in self._handlers[event.type]:
            try:
                handler(event)
            except Exception:
                logger.exception("事件处理异常: type=%s handler=%s", event.type, handler)

        for handler in self._general_handlers:
            try:
                handler(event)
            except Exception:
                logger.exception("通用事件处理异常: type=%s", event.type)

    def _run_timer(self) -> None:
        import time

        while self._active:
            time.sleep(self._timer_interval)
            if self._active:
                self.put(Event(EVENT_TIMER))

    # ---------------------------------------------------------------- 对外接口

    def put(self, event: Event) -> None:
        self._queue.put(event)

    def register(self, type_: str, handler: HandlerType) -> None:
        """注册指定类型的事件处理函数（重复注册会被忽略）"""
        if handler not in self._handlers[type_]:
            self._handlers[type_].append(handler)

    def unregister(self, type_: str, handler: HandlerType) -> None:
        handlers = self._handlers[type_]
        if handler in handlers:
            handlers.remove(handler)
        if not handlers:
            self._handlers.pop(type_, None)

    def register_general(self, handler: HandlerType) -> None:
        """注册接收所有事件的处理函数（日志/录制用）"""
        if handler not in self._general_handlers:
            self._general_handlers.append(handler)

    def unregister_general(self, handler: HandlerType) -> None:
        if handler in self._general_handlers:
            self._general_handlers.remove(handler)

    @property
    def qsize(self) -> int:
        """当前积压事件数，用于监控是否处理不过来"""
        return self._queue.qsize()
