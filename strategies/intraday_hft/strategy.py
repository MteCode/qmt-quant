"""日内高频剥头皮策略（Tick 级别 VWAP 回归 + 动量突破）。

原理：
  1. 实时计算滚动 VWAP（volume-weighted average price）
  2. 价格低于 VWAP 超过阈值 → 买入（均值回归做多）
  3. 价格高于 VWAP 超过阈值 → 卖出（获利了结）
  4. 持仓超过最大时间 → 强制平仓（避免隔夜风险）
  5. 止损：亏损超过阈值立即平仓

运行模式：
  - miniQMT 模拟盘：接收真实行情 tick，策略自动交易
  - SIM 网关：sim_ticks=true 时自带 tick 生成器（仅供离线调试）
"""
import logging
import random
import threading
from collections import deque
from datetime import datetime

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
        "take_profit",       # 止盈比例
        "lot_size",          # 每次下单手数（100的倍数）
        "max_hold_seconds",  # 持仓最长时间（秒）
        "max_daily_trades",  # 每日最大交易笔数
        "close_time",        # 尾盘平仓时间
        "sim_ticks",         # 是否启动假 tick 生成器（仅 SIM 网关调试用）
        "sim_tick_interval", # 假 tick 间隔（秒）
    ]
    variables = StrategyBase.variables + [
        "daily_trades", "entry_price", "total_pnl",
    ]

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        self.vwap_window: int = 200
        self.entry_threshold: float = 0.003
        self.exit_threshold: float = 0.002
        self.stop_loss: float = 0.005
        self.take_profit: float = 0.006
        self.lot_size: int = 100
        self.max_hold_seconds: int = 300
        self.max_daily_trades: int = 50
        self.close_time: str = "14:50"
        self.sim_ticks: bool = False
        self.sim_tick_interval: float = 0.5

        self.daily_trades: int = 0
        self.entry_price: float = 0.0
        self.entry_time: datetime | None = None
        self.total_pnl: float = 0.0

        # 每个标的独立的 tick 缓冲和 VWAP
        self._tick_bufs: dict[str, deque] = {}
        self._vwaps: dict[str, float] = {}
        self._momentums: dict[str, float] = {}
        self._entry_prices: dict[str, float] = {}
        self._entry_times: dict[str, datetime | None] = {}

        self._sim_thread: threading.Thread | None = None
        self._sim_stop = threading.Event()

        super().__init__(engine, strategy_name, vt_symbols, setting)

        for sym in self.vt_symbols:
            self._tick_bufs[sym] = deque(maxlen=500)
            self._vwaps[sym] = 0.0
            self._momentums[sym] = 0.0
            self._entry_prices[sym] = 0.0
            self._entry_times[sym] = None

    def on_init(self):
        self.write_log(
            f"日内高频策略初始化: 标的={self.vt_symbols}, "
            f"VWAP窗口={self.vwap_window}, "
            f"入场偏离={self.entry_threshold:.1%}, "
            f"止损={self.stop_loss:.1%}, "
            f"sim_ticks={self.sim_ticks}"
        )

    def on_start(self):
        self.daily_trades = 0
        self.total_pnl = 0.0
        for sym in self.vt_symbols:
            self._entry_prices[sym] = 0.0
            self._entry_times[sym] = None
        self.write_log("日内高频策略启动")
        if self.sim_ticks:
            self._start_sim_ticks()

    def on_stop(self):
        self._stop_sim_ticks()
        self.write_log(
            f"日内高频策略停止, 今日交易 {self.daily_trades} 笔, "
            f"累计盈亏 {self.total_pnl:.2f}"
        )

    # ---- Tick 处理 ----

    def on_tick(self, tick: TickData):
        if tick.vt_symbol not in self.vt_symbols:
            return
        if tick.last_price <= 0:
            return

        sym = tick.vt_symbol
        self._tick_bufs[sym].append({
            "price": tick.last_price,
            "volume": max(tick.volume, 1),
            "time": tick.datetime or datetime.now(),
        })

        self._update_indicators(sym)
        self._check_close_time(tick)
        self._check_signal(tick)

    def _update_indicators(self, sym: str):
        buf = list(self._tick_bufs[sym])
        if len(buf) < 10:
            return
        window = buf[-self.vwap_window:]
        total_pv = sum(t["price"] * t["volume"] for t in window)
        total_v = sum(t["volume"] for t in window)
        if total_v > 0:
            self._vwaps[sym] = total_pv / total_v

        if len(buf) >= 5:
            recent = [t["price"] for t in buf[-5:]]
            self._momentums[sym] = (recent[-1] - recent[0]) / recent[0] if recent[0] else 0

    def _check_close_time(self, tick: TickData):
        """尾盘强制平仓 —— 日内策略不隔夜。"""
        now = tick.datetime or datetime.now()
        close_h, close_m = map(int, self.close_time.split(":"))
        if now.hour > close_h or (now.hour == close_h and now.minute >= close_m):
            sym = tick.vt_symbol
            current_pos = self.get_pos(sym)
            if current_pos > 0:
                self.sell(sym, tick.last_price * 0.995, current_pos)
                pnl = (tick.last_price - self._entry_prices.get(sym, tick.last_price)) * current_pos
                self.total_pnl += pnl
                self.daily_trades += 1
                self.write_log(f"尾盘平仓 {sym} {current_pos}股 @ {tick.last_price:.2f} 盈亏={pnl:+.2f}")
                self._entry_prices[sym] = 0.0
                self._entry_times[sym] = None

    def _check_signal(self, tick: TickData):
        if not self.trading:
            return
        if self.daily_trades >= self.max_daily_trades:
            return

        sym = tick.vt_symbol
        price = tick.last_price
        vwap = self._vwaps.get(sym, 0)
        momentum = self._momentums.get(sym, 0)
        if vwap <= 0:
            return

        deviation = (price - vwap) / vwap
        current_pos = self.get_pos(sym)
        now = tick.datetime or datetime.now()

        # ---- 持仓中：检查出场 ----
        if current_pos > 0:
            entry_p = self._entry_prices.get(sym, price)
            entry_t = self._entry_times.get(sym)
            hold_time = (now - entry_t).total_seconds() if entry_t else 0
            pnl_pct = (price - entry_p) / entry_p if entry_p else 0

            should_exit = False
            reason = ""

            if pnl_pct <= -self.stop_loss:
                should_exit = True
                reason = f"止损 {pnl_pct:.2%}"
            elif pnl_pct >= self.take_profit:
                should_exit = True
                reason = f"止盈 {pnl_pct:.2%}"
            elif deviation >= self.exit_threshold:
                should_exit = True
                reason = f"VWAP偏离回归 +{deviation:.2%}"
            elif hold_time >= self.max_hold_seconds:
                should_exit = True
                reason = f"超时 {hold_time:.0f}s"

            if should_exit:
                self.sell(sym, price * 0.995, current_pos)
                pnl = (price - entry_p) * current_pos
                self.total_pnl += pnl
                self.daily_trades += 1
                self.write_log(
                    f"平仓 {sym} {current_pos}股 @ {price:.2f} [{reason}] 盈亏={pnl:+.2f}"
                )
                self._entry_prices[sym] = 0.0
                self._entry_times[sym] = None

        # ---- 空仓：检查入场 ----
        elif current_pos == 0:
            should_buy = False

            if deviation <= -self.entry_threshold and momentum > -0.001:
                should_buy = True
            elif momentum >= self.entry_threshold * 1.5:
                should_buy = True

            if should_buy:
                volume = self.lot_size
                cost = price * volume
                cash = self.get_cash()
                if cost > cash * 0.3:
                    return

                self.buy(sym, price * 1.005, volume)
                self._entry_prices[sym] = price
                self._entry_times[sym] = now
                self.daily_trades += 1
                self.write_log(
                    f"开仓 {sym} {volume}股 @ {price:.2f} "
                    f"VWAP={vwap:.2f} 偏离={deviation:+.2%} 动量={momentum:+.4f}"
                )

    def on_trade(self, trade):
        super().on_trade(trade)
        self.write_log(
            f"成交回报 {trade.vt_symbol} "
            f"{'买入' if trade.direction == Direction.LONG else '卖出'} "
            f"{trade.volume}股 @ {trade.price:.2f} 手续费={trade.commission:.2f}"
        )

    # ---- Tick 模拟器（仅 SIM 网关离线调试） ----

    def _start_sim_ticks(self):
        self._sim_stop.clear()
        self._sim_thread = threading.Thread(
            target=self._sim_tick_loop, daemon=True,
            name=f"sim-tick-{self.strategy_name}",
        )
        self._sim_thread.start()
        self.write_log("SIM Tick 模拟器已启动（仅调试用）")

    def _stop_sim_ticks(self):
        self._sim_stop.set()
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=3)
        self._sim_thread = None

    def _sim_tick_loop(self):
        prices = {}
        for sym in self.vt_symbols:
            p0 = 10.0 + random.random() * 20
            prices[sym] = {"price": p0, "open": p0, "high": p0, "low": p0, "vol_acc": 0}

        while not self._sim_stop.is_set():
            for vt_symbol in self.vt_symbols:
                state = prices[vt_symbol]
                p = state["price"]
                drift = -0.0001 * (p - state["open"]) / state["open"]
                shock = random.gauss(0, 1) * 0.001
                jump = random.gauss(0, 1) * 0.003 if random.random() < 0.02 else 0.0
                p = round(max(0.1, p * (1 + drift + shock + jump)), 2)
                state["price"] = p
                state["high"] = max(state["high"], p)
                state["low"] = min(state["low"], p)
                state["vol_acc"] += random.randint(100, 5000)

                parts = vt_symbol.split(".")
                symbol, exc_str = parts[0], parts[1] if len(parts) > 1 else "SSE"
                try:
                    exc = Exchange(exc_str)
                except ValueError:
                    exc = Exchange.SSE
                spread = max(0.01, p * 0.0005)
                tick = TickData(
                    symbol=symbol, exchange=exc, datetime=datetime.now(),
                    last_price=p, open_price=state["open"],
                    high_price=state["high"], low_price=state["low"],
                    pre_close=state["open"], volume=state["vol_acc"],
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
            self._sim_stop.wait(self.sim_tick_interval)
