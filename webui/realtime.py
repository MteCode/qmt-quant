"""WebSocket 实时推送层。

将 MainEngine 的事件流通过 Flask-SocketIO 推送到浏览器。

事件流::

    Gateway → EventEngine → MainEngine._on_event
                               ↓ register_broadcast
                           _broadcast()
                               ↓ socketio.emit
                           浏览器 WebSocket 客户端

客户端订阅::

    // 连接后自动收到引擎状态
    socket.on('engine_status', data => { ... });
    // 订阅特定事件
    socket.on('eTick', data => { ... });
    socket.on('eOrder', data => { ... });
    socket.on('eTrade', data => { ... });
    socket.on('eAccount', data => { ... });
    socket.on('eStrategyUpdate', data => { ... });
"""
from __future__ import annotations

import logging
import time
from threading import Lock

from flask import Flask

logger = logging.getLogger(__name__)

# MainEngine 实例（由 init_realtime 设置）
_engine = None

# SocketIO 实例
socketio = None

# 限流：同一事件类型每秒最多广播 N 次
_RATE_LIMITS: dict[str, int] = {
    "eTick": 5,         # tick 最高 5 次/秒
    "eBar": 2,
    "ePosition": 2,
}
_last_emit: dict[str, float] = {}
_emit_lock = Lock()


def _should_emit(event_type: str) -> bool:
    """限流检查。不在限流表里的事件类型不限。"""
    limit = _RATE_LIMITS.get(event_type)
    if limit is None:
        return True
    now = time.monotonic()
    interval = 1.0 / limit
    with _emit_lock:
        last = _last_emit.get(event_type, 0)
        if now - last < interval:
            return False
        _last_emit[event_type] = now
    return True


def _broadcast(event_type: str, data: dict) -> None:
    """MainEngine 事件回调 → SocketIO 广播。"""
    if socketio is None:
        return
    if not _should_emit(event_type):
        return
    try:
        socketio.emit(event_type, data, namespace="/ws")
    except Exception:
        logger.debug("WebSocket 广播异常", exc_info=True)


def init_realtime(app: Flask, engine=None) -> None:
    """初始化 WebSocket 层。

    :param app: Flask 应用实例
    :param engine: MainEngine 实例（可选，后续通过 set_engine 设置）
    """
    global socketio, _engine

    try:
        from flask_socketio import SocketIO
    except ImportError:
        logger.warning("flask-socketio 未安装，实时推送不可用。"
                       "安装: pip install flask-socketio simple-websocket")
        return

    socketio = SocketIO(
        app,
        async_mode="threading",
        cors_allowed_origins="*",
        logger=False,
        engineio_logger=False,
    )

    if engine is not None:
        set_engine(engine)

    _register_handlers()
    logger.info("WebSocket 实时推送已初始化")


def set_engine(engine) -> None:
    """设置 MainEngine 并注册广播回调。"""
    global _engine
    if _engine is not None:
        _engine.unregister_broadcast(_broadcast)
    _engine = engine
    if engine is not None:
        engine.register_broadcast(_broadcast)


def _register_handlers():
    """注册 SocketIO 事件处理器。"""
    if socketio is None:
        return

    from flask_socketio import emit

    @socketio.on("connect", namespace="/ws")
    def on_connect():
        """客户端连接时推送当前引擎状态。"""
        if _engine is not None:
            emit("engine_status", _engine.get_engine_status())
            emit("eStrategyUpdate",
                 {"strategies": _engine.get_all_strategies()})

    @socketio.on("get_status", namespace="/ws")
    def on_get_status():
        """客户端主动请求状态刷新。"""
        if _engine is not None:
            emit("engine_status", _engine.get_engine_status())

    @socketio.on("get_strategies", namespace="/ws")
    def on_get_strategies():
        if _engine is not None:
            emit("eStrategyUpdate",
                 {"strategies": _engine.get_all_strategies()})

    @socketio.on("get_positions", namespace="/ws")
    def on_get_positions():
        if _engine is not None:
            emit("positions", _engine.get_all_positions())

    @socketio.on("get_orders", namespace="/ws")
    def on_get_orders():
        if _engine is not None:
            emit("orders", _engine.get_all_orders())

    # ---- 命令：引擎控制 ----

    @socketio.on("cmd_connect_gateway", namespace="/ws")
    def on_cmd_connect(data):
        if _engine is None:
            emit("cmd_result", {"ok": False, "error": "引擎未初始化"})
            return
        gateway = data.get("gateway", "sim")
        ok = _engine.connect_gateway(gateway)
        emit("cmd_result", {"ok": ok, "action": "connect_gateway"})
        if ok:
            emit("engine_status", _engine.get_engine_status())

    @socketio.on("cmd_load_strategies", namespace="/ws")
    def on_cmd_load():
        if _engine is None:
            emit("cmd_result", {"ok": False, "error": "引擎未初始化"})
            return
        count = _engine.load_strategies_from_config()
        emit("cmd_result", {"ok": True, "action": "load_strategies",
                            "count": count})

    @socketio.on("cmd_init_all", namespace="/ws")
    def on_cmd_init_all():
        if _engine is not None:
            _engine.init_all_strategies()
            emit("cmd_result", {"ok": True, "action": "init_all"})

    @socketio.on("cmd_start_all", namespace="/ws")
    def on_cmd_start_all():
        if _engine is not None:
            _engine.start_all_strategies()
            emit("cmd_result", {"ok": True, "action": "start_all"})

    @socketio.on("cmd_stop_all", namespace="/ws")
    def on_cmd_stop_all():
        if _engine is not None:
            _engine.stop_all_strategies()
            emit("cmd_result", {"ok": True, "action": "stop_all"})

    @socketio.on("cmd_start_strategy", namespace="/ws")
    def on_cmd_start_strategy(data):
        if _engine is not None:
            ok = _engine.start_strategy(data.get("name", ""))
            emit("cmd_result", {"ok": ok, "action": "start_strategy"})

    @socketio.on("cmd_stop_strategy", namespace="/ws")
    def on_cmd_stop_strategy(data):
        if _engine is not None:
            ok = _engine.stop_strategy(data.get("name", ""))
            emit("cmd_result", {"ok": ok, "action": "stop_strategy"})

    @socketio.on("cmd_close", namespace="/ws")
    def on_cmd_close():
        if _engine is not None:
            _engine.close()
            emit("engine_status", _engine.get_engine_status())

    @socketio.on("cmd_kill_switch", namespace="/ws")
    def on_cmd_kill(data):
        if _engine is not None:
            if data.get("activate"):
                _engine.activate_kill_switch(data.get("reason", "手动急停"))
            else:
                _engine.deactivate_kill_switch()
            emit("cmd_result", {"ok": True, "action": "kill_switch"})


def run_socketio(app: Flask, **kwargs) -> None:
    """用 SocketIO 替代 app.run() 启动 Flask。"""
    if socketio is not None:
        socketio.run(app, **kwargs)
    else:
        kwargs.pop("allow_unsafe_werkzeug", None)
        app.run(**kwargs)
