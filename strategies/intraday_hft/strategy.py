"""日内高频剥头皮策略（Tick 级别 VWAP 回归 + 动量突破）。

原理：
  1. 实时计算滚动 VWAP（volume-weighted average price）
  2. 价格低于 VWAP 超过阈值 → 买入（均值回归做多）
  3. 价格高于 VWAP 超过阈值 → 卖出（获利了结）
  4. 持仓超过最大时间 → 强制平仓（避免隔夜风险）
  5. 止损：亏损超过阈值立即平仓

适用场景：SIM 模拟盘演示 / 日内高频策略框架验证
"""
import logging
import math
import random
import threading
import time
from collections import deque
from datetime import datetime, timedelta

from qmtquant.core.constants import Direction, Exchange, OrderType
from qmtquant.core.objects import TickData
from qmtquant.event.engine import EVENT_TICK, Event
from qmtquant.strategy.base import StrategyBase

logger = logging.getLogger(__name__)


class IntradayHftStrategy(StrategyBase):
    """日内高频 VWAP 回归 + 动量突破"""

    parameters = [
        "vwap_window",       # VWAP 滚动窗口（tick 数）
        "entry_threshold",   # 偏离 VWAP 多少比例触发买入
        "exit_threshold",    # 偏离 VWAP 多少比例触发卖出
        "stop_loss",         # 止损比例
        "lot_size",          # 每次下单手数（100的倍数）
        "max_hold_seconds",  # 持仓最长时间（秒）
        "max_daily_trades",  # 每日最大交易笔数
        "sim_tick_interval", # 模拟 tick 推送间隔（秒）
    ]
    variables = StrategyBase.variables + [
        "daily_trades", "entry_price", "total_pnl",
    ]

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        self.vwap_window: int = 200
        self.entry_threshold: float = 0.003
        self.exit_threshold: float = 0.002
        self.stop_loss: float = 0.005
        self.lot_size: int = 100
        self.max_hold_seconds: int = 300
        self.max_daily_trades: int = 50
        self.sim_tick_interval: float = 0.5

        self.daily_trades: int = 0
        self.entry_price: float = 0.0
        self.entry_time: datetime | None = None
        self.total_pnl: float = 0.0

        self._tick_buf: deque = deque(maxlen=500)
        self._vwap: float = 0.0
        self._momentum: float = 0.0
        self._sim_thread: threading.Thread | None = None
        self._sim_stop = threading.Event()

        super().__init__(engine, strategy_name, vt_symbols, setting)

    def on_init(self):
        self.write_log(
            f"日内高频策略初始化: 标的={self.vt_symbols}, "
            f"VWAP窗口={self.vwap_window}, "
            f"入场偏离={self.entry_threshold:.1%}, "
            f"止损={self.stop_loss:.1%}"
        )

    def on_start(self):
        self.daily_trades = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.write_log("日内高频策略启动")
        self._start_sim_ticks()

    def on_stop(self):
        self._stop_sim_ticks()
        self.write_log(
            f"日内高频策略停止, 今日交易 {self.daily_trades} 笔, "
            f"累计盈亏 {self.total_pnl:.2f}"
        )

    def on_tick(self, tick: TickData):
        if tick.vt_symbol not in self.vt_symbols:
            return
        if tick.last_price <= 0:
            return

        self._tick_buf.append({
            "price": tick.last_price,
            "volume": max(tick.volume, 1),
            "time": tick.datetime or datetime.now(),
        })

        self._update_vwap()
        self._check_signal(tick)

    def _update_vwap(self):
        buf = list(self._tick_buf)
        if len(buf) < 10:
            return
        window = buf[-self.vwap_window:]
        total_pv = sum(t["price"] * t["volume"] for t in window)
        total_v = sum(t["volume"] for t in window)
        if total_v > 0:
            self._vwap = total_pv / total_v

        if len(buf) >= 5:
            recent = [t["price"] for t in buf[-5:]]
            self._momentum = (recent[-1] - recent[0]) / recent[0] if recent[0] else 0

    def _check_signal(self, tick: TickData):
        if not self.trading or self._vwap <= 0:
            return
        if self.daily_trades >= self.max_daily_trades:
            return

        price = tick.last_price
        vt_symbol = tick.vt_symbol
        deviation = (price - self._vwap) / self._vwap
        current_pos = self.get_pos(vt_symbol)
        now = tick.datetime or datetime.now()

        if current_pos > 0:
            hold_time = (now - self.entry_time).total_seconds() if self.entry_time else 0
            pnl_pct = (price - self.entry_price) / self.entry_price if self.entry_price else 0

            should_exit = False
            reason = ""

            if pnl_pct <= -self.stop_loss:
                should_exit = True
                reason = f"止损 {pnl_pct:.2%}"
            elif deviation >= self.exit_threshold:
                should_exit = True
                reason = f"获利 VWAP偏离+{deviation:.2%}"
            elif hold_time >= self.max_hold_seconds:
                should_exit = True
                reason = f"超时平仓 {hold_time:.0f}s"
            elif pnl_pct >= self.exit_threshold * 2:
                should_exit = True
                reason = f"目标止盈 {pnl_pct:.2%}"

            if should_exit:
                self.sell(vt_symbol, price * 0.995, current_pos)
                pnl = (price - self.entry_price) * current_pos
                self.total_pnl += pnl
                self.daily_trades += 1
                self.write_log(
                    f"平仓 {vt_symbol} {current_pos}股 @ {price:.2f} "
                    f"[{reason}] 盈亏={pnl:+.2f}"
                )
                self.entry_price = 0.0
                self.entry_time = None

        elif current_pos == 0:
            should_buy = False

            if deviation <= -self.entry_threshold and self._momentum > -0.001:
                should_buy = True
            elif self._momentum >= self.entry_threshold * 1.5:
                should_buy = True

            if should_buy:
                volume = self.lot_size
                cost = price * volume
                cash = self.get_cash()
                if cost > cash * 0.3:
                    return

                self.buy(vt_symbol, price * 1.005, volume)
                self.entry_price = price
                self.entry_time = now
                self.daily_trades += 1
                self.write_log(
                    f"开仓 {vt_symbol} {volume}股 @ {price:.2f} "
                    f"VWAP={self._vwap:.2f} 偏离={deviation:+.2%} "
                    f"动量={self._momentum:+.4f}"
                )

    # ---- Tick 模拟器（SIM 网关专用） ----

    def _start_sim_ticks(self):
        """启动后台线程模拟 tick 推送，让 SIM 网关有行情可撮合。"""
        self._sim_stop.clear()
        self._sim_thread = threading.Thread(
            target=self._sim_tick_loop, daemon=True,
            name=f"sim-tick-{self.strategy_name}",
        )
        self._sim_thread.start()
        self.write_log("Tick 模拟器已启动")

    def _stop_sim_ticks(self):
        self._sim_stop.set()
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=3)
        self._sim_thread = None
        self.write_log("Tick 模拟器已停止")

    def _sim_tick_loop(self):
        """生成带微观结构的模拟 tick 流。

        价格模型：几何布朗运动 + 均值回复 + 随机跳跃
        """
        prices = {}
        for sym in self.vt_symbols:
            prices[sym] = {
                "price": 10.0 + random.random() * 20,
                "mu": 0.0,
                "vol_acc": 0,
            }
            prices[sym]["open"] = prices[sym]["price"]
            prices[sym]["high"] = prices[sym]["price"]
            prices[sym]["low"] = prices[sym]["price"]

        tick_count = 0
        while not self._sim_stop.is_set():
            for vt_symbol in self.vt_symbols:
                state = prices[vt_symbol]
                p = state["price"]

                drift = -0.0001 * (p - state["open"]) / state["open"]
                shock = random.gauss(0, 1) * 0.001
                jump = 0.0
                if random.random() < 0.02:
                    jump = random.gauss(0, 1) * 0.003

                ret = drift + shock + jump
                p = p * (1 + ret)
                p = round(p, 2)
                if p <= 0.1:
                    p = 0.1

                state["price"] = p
                state["high"] = max(state["high"], p)
                state["low"] = min(state["low"], p)

                vol_tick = random.randint(100, 5000)
                state["vol_acc"] += vol_tick

                spread = max(0.01, p * 0.0005)
                parts = vt_symbol.split(".")
                symbol = parts[0]
                exc_str = parts[1] if len(parts) > 1 else "SSE"
                try:
                    exc = Exchange(exc_str)
                except ValueError:
                    exc = Exchange.SSE

                tick = TickData(
                    symbol=symbol,
                    exchange=exc,
                    datetime=datetime.now(),
                    last_price=p,
                    open_price=state["open"],
                    high_price=state["high"],
                    low_price=state["low"],
                    pre_close=state["open"],
                    volume=state["vol_acc"],
                    bid_price_1=round(p - spread / 2, 2),
                    bid_volume_1=random.randint(100, 10000),
                    ask_price_1=round(p + spread / 2, 2),
                    ask_volume_1=random.randint(100, 10000),
                    limit_up=round(state["open"] * 1.1, 2),
                    limit_down=round(state["open"] * 0.9, 2),
                    gateway_name="SIM",
                )

                try:
                    self.engine.event_engine.put(Event(EVENT_TICK, tick))
                except Exception:
                    pass

            tick_count += 1
            self._sim_stop.wait(self.sim_tick_interval)
