"""系统主引擎 —— 统管事件、网关、策略的中央协调器。

参照 vnpy MainEngine 设计，所有组件的生命周期由这里控制::

    MainEngine
    ├── EventEngine（事件总线）
    ├── Gateway（券商通道）
    ├── RiskManager（风控）
    ├── LiveEngine（实盘撮合）
    └── 策略实例集合

策略和 UI 不直接操作内部组件，一律通过 MainEngine 的方法。
UI 通过 ``register_broadcast()`` 注册回调，接收实时事件推送。

用法（在 Flask 进程中）::

    engine = MainEngine()
    engine.start()                           # 启动事件引擎
    engine.connect_gateway("sim")            # 连接网关
    engine.load_strategies_from_config()     # 从 config.yaml 加载策略
    engine.init_all_strategies()
    engine.start_all_strategies()
    ...
    engine.close()

也可以单独当脚本跑（替代 run_live.py）::

    python -m qmtquant.engine.main_engine --gateway sim
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..config import AppConfig, get_config
from ..core.constants import Direction, Status
from ..core.objects import (
    AccountData, BarData, OrderData, PositionData, TickData, TradeData,
)
from ..event.engine import (
    EVENT_ACCOUNT, EVENT_BAR, EVENT_GATEWAY_STATUS, EVENT_LOG,
    EVENT_ORDER, EVENT_POSITION, EVENT_TICK, EVENT_TIMER, EVENT_TRADE,
    Event, EventEngine,
)
from ..gateway.base import BaseGateway
from ..risk.risk_manager import RiskManager
from ..strategy.base import StrategyBase
from .live_engine import LiveEngine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# 不需要广播给 UI 的事件类型（太频繁或无意义）
_SKIP_BROADCAST = {EVENT_TIMER}


# ------------------------------------------------------------------ 序列化

def serialize_event(obj: Any) -> dict | None:
    """将事件数据转为 JSON-safe dict。Enum→字符串，datetime→ISO。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in fields(obj):
            v = getattr(obj, f.name)
            out[f.name] = _convert(v)
        return out
    return {"value": _convert(obj)}


def _convert(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, datetime):
        return v.isoformat()
    if is_dataclass(v) and not isinstance(v, type):
        return serialize_event(v)
    if isinstance(v, (list, tuple)):
        return [_convert(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _convert(val) for k, val in v.items()}
    return v


# ------------------------------------------------------------------ 网关工厂

def build_gateway(name: str, event_engine: EventEngine,
                  cfg: AppConfig) -> BaseGateway:
    """按名称构造网关实例。"""
    if name == "sim":
        from ..gateway.sim_gateway import SimGateway
        return SimGateway(event_engine, "SIM",
                          initial_capital=cfg.backtest.initial_capital,
                          cost=cfg.cost)
    if name == "miniqmt":
        from ..gateway.miniqmt_gateway import MiniQmtGateway
        return MiniQmtGateway(event_engine)
    raise ValueError(f"未知网关: {name}（可选 sim / miniqmt）")


def _load_class(dotted: str) -> type:
    """``a.b.module.ClassName`` → 类对象"""
    module_path, _, class_name = dotted.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ------------------------------------------------------------------ 主引擎


class MainEngine:
    """系统主引擎。

    线程模型：EventEngine 有自己的处理线程和定时器线程，
    Gateway 在自己的线程里推事件。MainEngine 本身不开线程，
    只在事件回调（EventEngine 线程）里做广播。
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.event_engine = EventEngine()

        self.gateway: BaseGateway | None = None
        self.gateway_name: str = ""
        self.risk_manager: RiskManager | None = None
        self.live_engine: LiveEngine | None = None
        self.store = None

        # 已注册的策略类：name → class
        self._strategy_classes: dict[str, type[StrategyBase]] = {}

        # 事件广播回调（供 WebSocket / Qt 等 UI 层注册）
        self._broadcast_handlers: list[Callable[[str, dict], None]] = []

        # 状态
        self._started = False
        self._gateway_connected = False

        # 实时快照（供 UI 查询当前状态，不必等事件）
        self.account: AccountData | None = None
        self.positions: dict[str, PositionData] = {}
        self.latest_ticks: dict[str, dict] = {}
        self._strategy_pnl: dict[str, float] = {}

        # 注册事件转发
        self.event_engine.register_general(self._on_event)

    # ------------------------------------------------------------ 广播

    def register_broadcast(self,
                           handler: Callable[[str, dict], None]) -> None:
        """注册事件广播回调。handler(event_type, data_dict)。
        在 EventEngine 线程中调用，handler 应尽快返回。
        """
        self._broadcast_handlers.append(handler)

    def unregister_broadcast(self,
                             handler: Callable[[str, dict], None]) -> None:
        if handler in self._broadcast_handlers:
            self._broadcast_handlers.remove(handler)

    def _on_event(self, event: Event) -> None:
        """EventEngine 通用处理器：更新快照 + 广播到 UI。"""
        # 更新本地快照
        self._update_snapshot(event)

        # 广播到 UI
        if event.type in _SKIP_BROADCAST:
            return
        if not self._broadcast_handlers:
            return

        data = serialize_event(event.data)
        if data is None:
            return

        for handler in self._broadcast_handlers:
            try:
                handler(event.type, data)
            except Exception:
                logger.debug("广播处理异常", exc_info=True)

    def _update_snapshot(self, event: Event) -> None:
        """维护本地快照供同步查询。"""
        if event.type == EVENT_ACCOUNT and isinstance(event.data, AccountData):
            self.account = event.data
        elif event.type == EVENT_POSITION and isinstance(event.data, PositionData):
            pos = event.data
            if pos.volume <= 0:
                self.positions.pop(pos.vt_symbol, None)
            else:
                self.positions[pos.vt_symbol] = pos
        elif event.type == EVENT_TICK and isinstance(event.data, TickData):
            tick = event.data
            self.latest_ticks[tick.vt_symbol] = {
                "symbol": tick.symbol,
                "vt_symbol": tick.vt_symbol,
                "last_price": tick.last_price,
                "volume": tick.volume,
                "datetime": tick.datetime.isoformat() if tick.datetime else "",
            }

    # ------------------------------------------------------------ 生命周期

    @property
    def started(self) -> bool:
        return self._started

    @property
    def gateway_connected(self) -> bool:
        return self._gateway_connected and self.gateway is not None and self.gateway.connected

    def start(self) -> None:
        """启动事件引擎。不连接网关、不加载策略，只启动事件循环。"""
        if self._started:
            return
        self.event_engine.start()
        self._started = True
        logger.info("MainEngine 事件引擎已启动")

    def connect_gateway(self, gateway_name: str | None = None,
                        setting: dict | None = None) -> bool:
        """连接券商网关。"""
        if not self._started:
            self.start()

        name = gateway_name or self.config.gateway.name
        try:
            self.gateway = build_gateway(name, self.event_engine, self.config)
        except ValueError as e:
            logger.error("创建网关失败: %s", e)
            return False

        self.gateway_name = name

        if setting is None:
            cfg = self.config.gateway
            setting = {
                "qmt_path": cfg.qmt_path,
                "account_id": cfg.account_id,
                "account_type": cfg.account_type,
                "reconnect_max_retry": cfg.reconnect_max_retry,
                "reconnect_base_delay": cfg.reconnect_base_delay,
                "request_timeout": cfg.request_timeout,
            }

        if not self.gateway.connect(setting):
            logger.error("网关连接失败: %s", name)
            return False

        self._gateway_connected = True

        # 初始化风控
        self.risk_manager = RiskManager(self.config.risk, self.event_engine)

        # 初始化状态存储
        try:
            from ..config import DATA_DIR
            from ..store.database import StateStore
            self.store = StateStore(DATA_DIR / "state.db")
        except Exception:
            logger.warning("状态存储初始化失败，策略状态不会持久化", exc_info=True)

        # 创建 LiveEngine
        self.live_engine = LiveEngine(
            self.event_engine, self.gateway, self.risk_manager,
            store=self.store,
        )

        logger.info("网关已连接: %s", name)

        # 对账
        try:
            self.live_engine.reconcile()
        except Exception:
            logger.warning("对账失败", exc_info=True)

        return True

    def close(self) -> None:
        """停止所有策略、断开网关、停止事件引擎。"""
        if self.live_engine:
            try:
                self.live_engine.close()
            except Exception:
                logger.exception("LiveEngine 关闭异常")
            self.live_engine = None

        if self.gateway:
            self._gateway_connected = False
            self.gateway = None

        if self._started:
            self.event_engine.stop()
            self._started = False

        self.risk_manager = None
        self.store = None
        logger.info("MainEngine 已关闭")

    # ------------------------------------------------------------ 策略类注册

    def register_strategy_class(self, name: str,
                                cls: type[StrategyBase]) -> None:
        """注册一个策略类，之后可通过 name 创建实例。"""
        self._strategy_classes[name] = cls

    def register_strategy_class_by_path(self, name: str,
                                        dotted: str) -> bool:
        """通过点分路径注册策略类。"""
        try:
            cls = _load_class(dotted)
        except (ImportError, AttributeError) as e:
            logger.error("加载策略类 %s 失败: %s", dotted, e)
            return False
        if not (isinstance(cls, type) and issubclass(cls, StrategyBase)):
            logger.error("%s 不是 StrategyBase 的子类", dotted)
            return False
        self._strategy_classes[name] = cls
        return True

    def load_strategy_classes_from_config(self) -> int:
        """从 config.yaml 的 strategies 列表预注册所有策略类。"""
        count = 0
        for item in self.config.strategies:
            dotted = item.get("class", "")
            name = item.get("name", "")
            if dotted and name:
                if self.register_strategy_class_by_path(name, dotted):
                    count += 1
        return count

    @property
    def strategy_class_names(self) -> list[str]:
        return list(self._strategy_classes.keys())

    # ------------------------------------------------------------ 策略实例管理

    def add_strategy(self, class_name: str, strategy_name: str,
                     vt_symbols: list[str],
                     setting: dict | None = None) -> bool:
        """添加策略实例（未初始化、未启动）。"""
        if self.live_engine is None:
            logger.error("引擎未就绪，请先 connect_gateway()")
            return False

        cls = self._strategy_classes.get(class_name)
        if cls is None:
            logger.error("策略类未注册: %s", class_name)
            return False

        if strategy_name in self.live_engine.strategies:
            logger.error("策略名已存在: %s", strategy_name)
            return False

        try:
            self.live_engine.add_strategy(cls, strategy_name, vt_symbols,
                                          setting)
            logger.info("已添加策略: %s (class=%s, symbols=%d)",
                        strategy_name, class_name, len(vt_symbols))
            self._broadcast_strategy_update()
            return True
        except Exception:
            logger.exception("添加策略失败: %s", strategy_name)
            return False

    def init_strategy(self, strategy_name: str) -> bool:
        """初始化单个策略（加载历史数据预热）。"""
        if self.live_engine is None:
            return False
        strategy = self.live_engine.strategies.get(strategy_name)
        if not strategy:
            return False
        try:
            strategy.on_init()
            strategy.inited = True
            # 订阅行情
            from ..utils.symbol import split_vt_symbol
            from ..core.objects import SubscribeRequest
            for vt_sym in strategy.vt_symbols:
                sym, exc = split_vt_symbol(vt_sym)
                self.gateway.subscribe(SubscribeRequest(symbol=sym, exchange=exc))
            self._broadcast_strategy_update()
            return True
        except Exception:
            logger.exception("初始化策略失败: %s", strategy_name)
            return False

    def start_strategy(self, strategy_name: str) -> bool:
        """启动单个策略。"""
        if self.live_engine is None:
            return False
        strategy = self.live_engine.strategies.get(strategy_name)
        if not strategy or not strategy.inited:
            return False
        try:
            strategy.on_start()
            strategy.trading = True
            logger.info("策略已启动: %s", strategy_name)
            self._broadcast_strategy_update()
            return True
        except Exception:
            logger.exception("启动策略失败: %s", strategy_name)
            return False

    def stop_strategy(self, strategy_name: str) -> bool:
        """停止单个策略。"""
        if self.live_engine is None:
            return False
        strategy = self.live_engine.strategies.get(strategy_name)
        if not strategy or not strategy.trading:
            return False
        try:
            self.live_engine.cancel_all(strategy_name)
            strategy.trading = False
            strategy.on_stop()
            self.live_engine.save_all_states()
            logger.info("策略已停止: %s", strategy_name)
            self._broadcast_strategy_update()
            return True
        except Exception:
            logger.exception("停止策略失败: %s", strategy_name)
            return False

    def remove_strategy(self, strategy_name: str) -> bool:
        """移除策略实例（必须先停止）。"""
        if self.live_engine is None:
            return False
        strategy = self.live_engine.strategies.get(strategy_name)
        if not strategy:
            return False
        if strategy.trading:
            logger.error("策略仍在运行，请先停止: %s", strategy_name)
            return False
        del self.live_engine.strategies[strategy_name]
        logger.info("已移除策略: %s", strategy_name)
        self._broadcast_strategy_update()
        return True

    def init_all_strategies(self) -> None:
        """初始化所有策略。"""
        if self.live_engine:
            self.live_engine.init_all()
            self._broadcast_strategy_update()

    def start_all_strategies(self) -> None:
        """启动所有策略。"""
        if self.live_engine:
            self.live_engine.start_all()
            self._broadcast_strategy_update()

    def stop_all_strategies(self) -> None:
        """停止所有策略。"""
        if self.live_engine:
            self.live_engine.stop_all()
            self._broadcast_strategy_update()

    def load_strategies_from_config(self) -> int:
        """从 config.yaml 加载并添加所有策略。"""
        self.load_strategy_classes_from_config()
        count = 0
        for item in self.config.strategies:
            name = item.get("name", "")
            vt_symbols = item.get("vt_symbols", [])
            setting = item.get("setting", {})
            if name and name in self._strategy_classes:
                if isinstance(vt_symbols, str) and vt_symbols == "from_signal":
                    vt_symbols = _resolve_signal_symbols(item, ROOT)
                if self.add_strategy(name, name, vt_symbols, setting):
                    count += 1
        return count

    # ------------------------------------------------------------ 查询接口

    def get_engine_status(self) -> dict:
        """返回引擎整体状态。"""
        return {
            "started": self._started,
            "gateway_name": self.gateway_name,
            "gateway_connected": self.gateway_connected,
            "strategy_count": len(self.live_engine.strategies) if self.live_engine else 0,
            "running_count": sum(
                1 for s in (self.live_engine.strategies.values() if self.live_engine else [])
                if s.trading
            ),
            "account": serialize_event(self.account) if self.account else None,
            "event_queue_size": self.event_engine.qsize if self._started else 0,
        }

    def get_all_strategies(self) -> list[dict]:
        """返回所有策略实例的状态。"""
        if not self.live_engine:
            return []
        out = []
        for name, strategy in self.live_engine.strategies.items():
            out.append({
                "name": name,
                "class": type(strategy).__name__,
                "inited": strategy.inited,
                "trading": strategy.trading,
                "vt_symbols": strategy.vt_symbols,
                "parameters": strategy.get_parameters(),
                "variables": strategy.get_variables(),
            })
        return out

    def get_all_positions(self) -> list[dict]:
        """返回网关报回的实时持仓。"""
        return [serialize_event(p) for p in self.positions.values()]

    def get_all_orders(self) -> list[dict]:
        """返回本地订单簿。"""
        if not self.live_engine:
            return []
        return [serialize_event(o) for o in self.live_engine.orders.values()]

    def get_active_orders(self) -> list[dict]:
        """返回活动委托。"""
        if not self.live_engine:
            return []
        return [serialize_event(o) for o in self.live_engine.orders.values()
                if o.is_active()]

    # ------------------------------------------------------------ 辅助

    def _broadcast_strategy_update(self) -> None:
        """广播策略状态变化。"""
        data = {"strategies": self.get_all_strategies()}
        for handler in self._broadcast_handlers:
            try:
                handler("eStrategyUpdate", data)
            except Exception:
                pass

    def activate_kill_switch(self, reason: str = "") -> None:
        """急停：禁止一切下单。"""
        if self.risk_manager:
            self.risk_manager.activate_kill_switch(reason)
            logger.warning("急停已激活: %s", reason)

    def deactivate_kill_switch(self) -> None:
        """解除急停。"""
        if self.risk_manager:
            self.risk_manager.deactivate_kill_switch()
            logger.info("急停已解除")


def _resolve_signal_symbols(item: dict, root: Path) -> list[str]:
    """从信号文件和持仓文件解析标的列表。"""
    import csv as _csv
    import json

    syms: set[str] = set()
    sig = (item.get("setting") or {}).get("signal_file", "")
    if sig:
        p = root / sig
        if p.exists():
            try:
                with p.open(encoding="utf-8-sig") as f:
                    syms.update(
                        r["vt_symbol"].strip()
                        for r in _csv.DictReader(f)
                        if r.get("vt_symbol"))
            except (OSError, KeyError):
                pass

    pos_file = root / "strategies" / "alstm_ppo_csi1000" / "state" / "positions.json"
    if pos_file.exists():
        try:
            snap = json.loads(pos_file.read_text(encoding="utf-8"))
            syms.update(
                h["vt_symbol"] for h in snap.get("holdings", [])
                if h.get("vt_symbol"))
        except (OSError, ValueError, KeyError):
            pass

    raw = item.get("vt_symbols", [])
    if isinstance(raw, list):
        syms.update(s for s in raw if s != "from_signal")
    return sorted(syms)
