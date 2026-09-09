"""告警通道 —— 把 ERROR 级日志推到企业微信 / 钉钉。

## 为什么要有

`NotifyConfig` 在 config.py 里定义了、从 yaml 加载了，但**任何地方都
没读过**。它承诺了一个不存在的功能：配好 webhook 也不会收到任何东西，
而且不会有任何提示。

对一个跑实盘的系统这很危险。系统里已经有二十来处 logger.error，
覆盖的都是真出事的场景：网关断开、报单被拒、行情停推、重连达上限
进入只读、对账查不到资金。这些事发生在无人值守的时段，
日志文件里静静躺着没人看。

## 为什么做成 logging handler 而不是逐处调用

逐个调用点插一行 notify() 要改二十几处，且以后新增的错误会漏掉。
挂成 handler 则自动覆盖全部 ERROR/CRITICAL，包括还没写的那些。

## 三个必须处理的点

**不能阻塞。** 事件引擎只有一个处理线程，在日志路径里同步发 HTTP，
一次网络超时就冻住整个交易系统。所以走后台队列，投递即返回。

**要去重限流。** 网关断开时 reconnect 循环会连续打同一条 error，
不限流会瞬间刷屏并触发 webhook 的频控，之后真正重要的告警反而发不出去。

**自身失败不能影响主流程。** 告警发不出去是小事，因为发不出去而让
交易系统崩掉是大事。所有异常吞掉，只在本地日志留痕
—— 且用 handleError 而不是 logger.error，否则会递归触发自己。
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

__all__ = ["WebhookNotifier", "NotifyHandler", "attach_notifier"]

logger = logging.getLogger(__name__)

#: 发送超时（秒）。宁可漏一条告警，也不能让队列线程卡住。
SEND_TIMEOUT = 5

#: 同一条消息在这个窗口内只发一次
DEDUP_WINDOW = 300

#: 每分钟最多发几条，超出的攒起来合并成一条
RATE_LIMIT = 10


@dataclass
class _Payload:
    title: str
    text: str


class WebhookNotifier:
    """企业微信 / 钉钉群机器人。

    两家的 JSON 结构不同，但都接受 markdown。这里只做格式适配，
    不处理签名 —— 加签的 webhook 请在 URL 里带好 sign 参数。
    """

    def __init__(self, webhook: str, channel: str = "wecom",
                 timeout: int = SEND_TIMEOUT,
                 opener=None) -> None:
        self.webhook = webhook
        self.channel = (channel or "wecom").lower()
        self.timeout = timeout
        #: 注入点，测试用假的传输，不真发网络请求
        self._opener = opener or urllib.request.urlopen

    def _body(self, p: _Payload) -> dict:
        text = f"**{p.title}**\n{p.text}" if p.title else p.text
        if self.channel == "dingtalk":
            return {"msgtype": "markdown",
                    "markdown": {"title": p.title or "告警", "text": text}}
        # 企业微信
        return {"msgtype": "markdown", "markdown": {"content": text}}

    def send(self, title: str, text: str) -> bool:
        """发一条。失败返回 False，不抛。"""
        if not self.webhook:
            return False
        data = json.dumps(self._body(_Payload(title, text)),
                          ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.webhook, data=data,
            headers={"Content-Type": "application/json"})
        try:
            with self._opener(req, timeout=self.timeout) as resp:
                return 200 <= getattr(resp, "status", 200) < 300
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return False


class NotifyHandler(logging.Handler):
    """把 ERROR 及以上的日志推给 notifier。

    投递是异步的：emit() 只往队列里放，由后台线程发送。
    队列满时丢弃最旧的 —— 告警积压时，新的比旧的重要。
    """

    def __init__(self, notifier: WebhookNotifier,
                 level: int = logging.ERROR,
                 dedup_window: int = DEDUP_WINDOW,
                 rate_limit: int = RATE_LIMIT,
                 queue_size: int = 200,
                 clock=time.monotonic) -> None:
        super().__init__(level)
        self.notifier = notifier
        self.dedup_window = dedup_window
        self.rate_limit = rate_limit
        self._clock = clock
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._seen: dict[str, float] = {}
        self._window_start = clock()
        self._window_count = 0
        self._suppressed = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="notify-worker", daemon=True)
        self._worker.start()

    # ---------------------------------------------------------- 过滤

    def _should_send(self, key: str) -> bool:
        """去重 + 限流。两者都在锁内判断，避免多线程同时放行。"""
        now = self._clock()
        with self._lock:
            last = self._seen.get(key)
            if last is not None and now - last < self.dedup_window:
                self._suppressed += 1
                return False

            if now - self._window_start >= 60:
                self._window_start = now
                self._window_count = 0
            if self._window_count >= self.rate_limit:
                self._suppressed += 1
                return False

            self._seen[key] = now
            self._window_count += 1
            # 清掉过期的去重记录，别让字典无限长
            if len(self._seen) > 1000:
                cutoff = now - self.dedup_window
                self._seen = {k: v for k, v in self._seen.items()
                              if v >= cutoff}
            return True

    # ---------------------------------------------------------- 日志接口

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # 去重按「未格式化的模板 + 名字」算，而不是最终文本 ——
            # 「重连失败 3/10」「重连失败 4/10」是同一件事的连续汇报，
            # 按最终文本去重等于不去重。
            key = f"{record.name}:{record.levelno}:{record.msg}"
            if not self._should_send(key):
                return
            title = f"[{record.levelname}] {record.name}"
            text = self.format(record)
            try:
                self._q.put_nowait(_Payload(title, text))
            except queue.Full:
                # 队列满说明发不出去且在积压，新的比旧的重要
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(_Payload(title, text))
                except (queue.Empty, queue.Full):
                    pass
        except Exception:                       # noqa: BLE001
            # 告警自身出错绝不能影响业务线程；也不能用 logger.error
            # 上报（会递归触发本 handler）
            self.handleError(record)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                p = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.notifier.send(p.title, p.text)
            except Exception:                   # noqa: BLE001
                pass

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2)
        super().close()

    @property
    def suppressed(self) -> int:
        """被去重或限流挡掉的条数。用于自检：一直为 0 说明没在工作。"""
        return self._suppressed


def attach_notifier(cfg, level: int = logging.ERROR) -> NotifyHandler | None:
    """按配置把告警挂到根日志。

    默认不启用（NotifyConfig.enabled=False）。只有用户在 config.yaml 里
    显式打开并填了 webhook 才会真的往外发 —— 外发是有副作用的动作，
    不该靠默认值触发。
    """
    notify = getattr(cfg, "notify", None)
    if not notify or not getattr(notify, "enabled", False):
        return None
    webhook = getattr(notify, "webhook", "")
    if not webhook:
        logger.warning("告警已启用但没有配 webhook，不会发出任何告警")
        return None

    h = NotifyHandler(
        WebhookNotifier(webhook, getattr(notify, "channel", "wecom")),
        level=level)
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s\n%(message)s"))
    logging.getLogger().addHandler(h)
    logger.info("告警通道已启用: %s", getattr(notify, "channel", "wecom"))
    return h
