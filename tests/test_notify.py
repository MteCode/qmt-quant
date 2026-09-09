"""告警通道测试。

## 背景

NotifyConfig 在 config.py 里定义了、从 yaml 加载了，但任何地方都没读过
—— 配好 webhook 也收不到任何东西，且没有任何提示。

一个跑实盘的系统里，「网关断开」「重连达上限进入只读」「行情停推」
这类事往往发生在无人值守时段。日志文件里静静躺着没人看，
和没记录区别不大。

## 这些测试不发真实网络请求

WebhookNotifier 接受注入的 opener，测试传一个假的传输记录调用。
真发请求的测试会在离线环境失败，也会把测试变成一个外发动作。
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error

import pytest

from qmtquant.config import AppConfig, NotifyConfig
from qmtquant.utils.notify import (
    NotifyHandler,
    WebhookNotifier,
    attach_notifier,
)


class FakeResponse:
    def __init__(self, status: int = 200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    """记录调用，不发网络。"""

    def __init__(self, status: int = 200, raises: Exception | None = None):
        self.calls: list[dict] = []
        self.status = status
        self.raises = raises

    def __call__(self, req, timeout=None):
        self.calls.append({
            "url": req.full_url,
            "body": json.loads(req.data.decode("utf-8")),
            "timeout": timeout,
        })
        if self.raises:
            raise self.raises
        return FakeResponse(self.status)


class TestWebhookFormats:
    def test_wecom_payload(self):
        op = FakeOpener()
        n = WebhookNotifier("https://example.invalid/hook", "wecom", opener=op)
        assert n.send("标题", "正文") is True
        body = op.calls[0]["body"]
        assert body["msgtype"] == "markdown"
        assert "标题" in body["markdown"]["content"]
        assert "正文" in body["markdown"]["content"]

    def test_dingtalk_payload(self):
        op = FakeOpener()
        n = WebhookNotifier("https://example.invalid/hook", "dingtalk",
                            opener=op)
        n.send("标题", "正文")
        body = op.calls[0]["body"]
        assert body["markdown"]["title"] == "标题"
        assert "正文" in body["markdown"]["text"]

    def test_unknown_channel_falls_back_to_wecom(self):
        op = FakeOpener()
        WebhookNotifier("https://x.invalid", "телеграм", opener=op).send("a", "b")
        assert "content" in op.calls[0]["body"]["markdown"]


class TestSendNeverRaises:
    """告警发不出去是小事，因为发不出去而让交易系统崩掉是大事。"""

    @pytest.mark.parametrize("exc", [
        urllib.error.URLError("no route"),
        OSError("socket closed"),
        TimeoutError("timed out"),
        ValueError("bad url"),
    ])
    def test_transport_errors_return_false(self, exc):
        n = WebhookNotifier("https://x.invalid", opener=FakeOpener(raises=exc))
        assert n.send("t", "x") is False

    def test_empty_webhook_is_noop(self):
        op = FakeOpener()
        assert WebhookNotifier("", opener=op).send("t", "x") is False
        assert op.calls == []

    def test_non_2xx_is_false(self):
        n = WebhookNotifier("https://x.invalid", opener=FakeOpener(status=500))
        assert n.send("t", "x") is False


class _Clock:
    """可控时钟，避免用 sleep 测限流 —— 那样测试又慢又不稳。"""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s: float):
        self.t += s


def _drain(h: NotifyHandler, timeout: float = 2.0):
    """等后台线程把队列发完。"""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if h._q.empty():
            time.sleep(0.05)
            return
        time.sleep(0.01)


def _record(msg: str, name: str = "qmtquant.gateway",
            level: int = logging.ERROR, args=()) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, args, None)


class TestDedup:
    """网关断开时 reconnect 循环会连续打同一条 error。
    不去重会瞬间刷屏并触发 webhook 频控，之后真正重要的告警反而发不出去。
    """

    def test_same_message_sent_once(self):
        op, clk = FakeOpener(), _Clock()
        h = NotifyHandler(WebhookNotifier("https://x.invalid", opener=op),
                          clock=clk)
        try:
            for _ in range(5):
                h.emit(_record("miniQMT 连接断开"))
            _drain(h)
            assert len(op.calls) == 1
            assert h.suppressed == 4
        finally:
            h.close()

    def test_dedup_expires(self):
        op, clk = FakeOpener(), _Clock()
        h = NotifyHandler(WebhookNotifier("https://x.invalid", opener=op),
                          dedup_window=300, clock=clk)
        try:
            h.emit(_record("断开"))
            clk.advance(301)
            h.emit(_record("断开"))
            _drain(h)
            assert len(op.calls) == 2
        finally:
            h.close()

    def test_dedup_by_template_not_final_text(self):
        """「重连失败 3/10」「4/10」是同一件事的连续汇报。

        按最终文本去重等于不去重 —— 每条都不一样，全都会发出去。
        """
        op, clk = FakeOpener(), _Clock()
        h = NotifyHandler(WebhookNotifier("https://x.invalid", opener=op),
                          clock=clk)
        try:
            for i in range(1, 6):
                h.emit(_record("重连失败 %d/10", args=(i,)))
            _drain(h)
            assert len(op.calls) == 1, \
                "同一个模板的连续汇报应当只发一条"
        finally:
            h.close()

    def test_different_messages_both_sent(self):
        op, clk = FakeOpener(), _Clock()
        h = NotifyHandler(WebhookNotifier("https://x.invalid", opener=op),
                          clock=clk)
        try:
            h.emit(_record("网关断开"))
            h.emit(_record("报单被拒"))
            _drain(h)
            assert len(op.calls) == 2
        finally:
            h.close()


class TestRateLimit:
    def test_caps_per_minute(self):
        op, clk = FakeOpener(), _Clock()
        h = NotifyHandler(WebhookNotifier("https://x.invalid", opener=op),
                          rate_limit=3, clock=clk)
        try:
            for i in range(10):
                h.emit(_record(f"错误{i}"))
            _drain(h)
            assert len(op.calls) == 3
            assert h.suppressed == 7
        finally:
            h.close()

    def test_window_resets(self):
        op, clk = FakeOpener(), _Clock()
        h = NotifyHandler(WebhookNotifier("https://x.invalid", opener=op),
                          rate_limit=2, clock=clk)
        try:
            for i in range(5):
                h.emit(_record(f"a{i}"))
            _drain(h)
            assert len(op.calls) == 2
            clk.advance(61)
            for i in range(5):
                h.emit(_record(f"b{i}"))
            _drain(h)
            assert len(op.calls) == 4
        finally:
            h.close()


class TestLevelFiltering:
    def test_warning_not_sent(self):
        op, clk = FakeOpener(), _Clock()
        h = NotifyHandler(WebhookNotifier("https://x.invalid", opener=op),
                          clock=clk)
        h.setFormatter(logging.Formatter("%(message)s"))
        log = logging.getLogger("test.notify.level")
        log.setLevel(logging.DEBUG)
        log.addHandler(h)
        try:
            log.warning("只是警告")
            log.error("真的出事了")
            _drain(h)
            assert len(op.calls) == 1
            assert "真的出事了" in op.calls[0]["body"]["markdown"]["content"]
        finally:
            log.removeHandler(h)
            h.close()


class TestAttachRespectsConfig:
    """外发是有副作用的动作，不该靠默认值触发。"""

    def test_disabled_by_default(self):
        cfg = AppConfig()
        assert cfg.notify.enabled is False
        assert attach_notifier(cfg) is None

    def test_enabled_without_webhook_is_noop(self):
        cfg = AppConfig(notify=NotifyConfig(enabled=True, webhook=""))
        assert attach_notifier(cfg) is None

    def test_enabled_with_webhook_attaches(self):
        cfg = AppConfig(notify=NotifyConfig(
            enabled=True, webhook="https://example.invalid/hook",
            channel="dingtalk"))
        h = attach_notifier(cfg)
        try:
            assert h is not None
            assert h in logging.getLogger().handlers
            assert h.notifier.channel == "dingtalk"
        finally:
            if h:
                logging.getLogger().removeHandler(h)
                h.close()


class TestNeverBlocksCaller:
    def test_emit_returns_before_send_completes(self):
        """事件引擎只有一个处理线程，日志路径里同步发 HTTP，
        一次网络超时就冻住整个交易系统。"""
        started = []

        class SlowOpener(FakeOpener):
            def __call__(self, req, timeout=None):
                started.append(1)
                time.sleep(0.5)
                return super().__call__(req, timeout)

        h = NotifyHandler(
            WebhookNotifier("https://x.invalid", opener=SlowOpener()),
            clock=_Clock())
        try:
            t0 = time.monotonic()
            h.emit(_record("慢速发送"))
            elapsed = time.monotonic() - t0
            assert elapsed < 0.1, \
                f"emit 应当立即返回，实际耗时 {elapsed:.3f}s"
        finally:
            h.close()

    def test_full_queue_drops_oldest_not_caller(self):
        h = NotifyHandler(
            WebhookNotifier("https://x.invalid",
                            opener=FakeOpener(raises=OSError("down"))),
            queue_size=2, rate_limit=1000, clock=_Clock())
        try:
            for i in range(50):
                h.emit(_record(f"错误{i}"))   # 不应抛，也不应卡住
        finally:
            h.close()
