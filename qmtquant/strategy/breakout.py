"""唐奇安通道突破（海龟交易法则）。

## 这个策略的每一个价位都是确定的

之前项目里的策略都只有「买入条件/卖出条件」，没有明确的价位。
本策略把每个动作都锚定到具体价格：

======================  ================================================
动作                     价位
======================  ================================================
**入场**                 突破过去 ``entry_window`` 日最高价
**初始止损**             入场价 − ``stop_atr`` × ATR
**加仓**                 每浮盈 0.5×ATR 加一份，最多 ``max_units`` 份
**移动止损**             最高价 × (1 − ``trailing_ratio``)，只上移
**离场**                 跌破过去 ``exit_window`` 日最低价
**仓位**                 风险预算 ÷ 止损距离
======================  ================================================

任何时刻你都能回答「现在该在什么价买、什么价卖、亏多少走人」——
这是它与前面几个策略的根本区别。

## 为什么止损用 ATR 而不是固定百分比

「跌 5% 止损」对不同标的含义完全不同：日常波动 4% 的票，
5% 止损每周被扫两次；日波动 0.8% 的票，5% 止损形同虚设。
ATR 把止损距离统一到「几倍日常波动」的尺度，跨标的可比。

## 仓位由止损距离反推

先定每笔最多亏总资产的 ``risk_per_trade``（默认 1%），
再由入场价与止损价的距离算出该买多少股。
**波动大的票自动买得少**，每笔交易的亏损上限都一样。

散户常见的「每次买 10 万，止损随便设 5%」是反过来的，
结果是每笔交易的实际风险敞口差好几倍。

## 加仓：让对的交易变大

海龟法则的核心不在入场准确率（约 35%），而在
「错了小亏，对了加仓」。每浮盈 0.5×ATR 加一份，
加仓的同时把所有仓位的止损上移到最新一份的止损位。

## ⚠ 已知弱点

- **胜率低**。突破策略天然胜率 30-40%，靠少数大赢利覆盖多数小亏。
  心理上难以坚持，回撤期可能长达一两年。
- **震荡市连续假突破**。价格反复越过通道又跌回，每次都是一次小亏。
  ``exit_window`` 小于 ``entry_window`` 就是为了快速认错。
- **A 股 T+1**：当日买入当日不能卖，止损最快次日执行。
  跳空低开时实际成交价比止损价更差。

⚠ 本策略不构成投资建议。
"""
import logging

from ..core.objects import BarData
from .base import StrategyBase
from .indicators import AverageTrueRange, Donchian
from .trade_manager import (
    EXIT_SIGNAL,
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIME,
    EXIT_TRAILING,
    TradeManager,
)

logger = logging.getLogger(__name__)


class BreakoutStrategy(StrategyBase):
    """通道突破 + ATR 止损 + 金字塔加仓"""

    parameters = [
        "entry_window", "exit_window", "atr_window", "stop_atr",
        "add_atr", "max_units", "risk_per_trade", "trailing_ratio",
        "trailing_start_r", "max_holding_bars", "max_position_ratio",
        "max_positions", "price_buffer", "exit_price_buffer",
    ]
    variables = ["inited", "trading", "pos", "trade_count", "units"]

    #: 入场通道窗口：突破过去 N 日最高价买入
    entry_window: int = 20
    #: 离场通道窗口：跌破过去 N 日最低价卖出。
    #: **必须小于 entry_window** —— 进场要慢（确认趋势），
    #: 出场要快（快速认错）。相等的话进出同一条线，来回打脸
    exit_window: int = 10
    #: ATR 窗口
    atr_window: int = 14
    #: 初始止损 = 入场价 − stop_atr × ATR。海龟原版用 2
    stop_atr: float = 2.0
    #: 每浮盈多少个 ATR 加一份仓。0 表示不加仓
    add_atr: float = 0.5
    #: 最多持有几份（含首份）。海龟原版 4 份
    max_units: int = 4
    #: 单笔最大亏损占总资产比例
    risk_per_trade: float = 0.01
    #: 移动止损比例。0 表示只靠通道离场
    trailing_ratio: float = 0.15
    #: 盈利达到多少 R 才启动移动止损
    trailing_start_r: float = 1.0
    #: 最长持有 Bar 数，0 不限
    max_holding_bars: int = 0
    #: 单标的仓位占总资产上限
    max_position_ratio: float = 0.20
    #: **最多同时持有几只**。
    #:
    #: 这是实测踩出来的：只做单笔风险控制（每笔 1%）是不够的，
    #: 在 949 只候选上跑时同时开出大量仓位，组合层面的风险敞口失控 ——
    #: 年化波动率 32%、最大回撤 -54%，比买入持有还差。
    #:
    #: 「每笔只亏 1%」和「组合只亏 1%」是两回事：
    #: 同时持 50 只，一次系统性下跌会让它们**一起**触发止损。
    #: 单笔风险控制管的是个股意外，组合并发上限管的是系统性风险
    max_positions: int = 10
    #: 买入限价缓冲
    price_buffer: float = 0.03
    #: **卖出限价缓冲，必须不小于买入侧**。
    #: 错过买入只是少赚，错过卖出是实亏 —— 止损单尤其不能挂不上
    exit_price_buffer: float = 0.08

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._coerce_types()
        self._validate()

        self.entry_ch = {s: Donchian(self.entry_window) for s in vt_symbols}
        self.exit_ch = {s: Donchian(self.exit_window) for s in vt_symbols}
        self.atr = {s: AverageTrueRange(self.atr_window) for s in vt_symbols}

        self.manager = TradeManager(
            risk_per_trade=self.risk_per_trade,
            trailing_ratio=self.trailing_ratio,
            trailing_start_r=self.trailing_start_r,
            max_holding_bars=self.max_holding_bars,
            max_position_ratio=self.max_position_ratio)

        self.trade_count: int = 0
        #: 各标的当前持有的份数，加仓上限用
        self.units: dict[str, int] = {}
        #: 下一次加仓的触发价
        self._next_add: dict[str, float] = {}
        self._bar_count: int = 0

    def _coerce_types(self) -> None:
        """参数寻优会从 DataFrame 传 float64，用作 deque(maxlen=) 会 TypeError，
        且该异常常被外层 try/except 吞掉，表现为整个网格静默跑空"""
        for name in ("entry_window", "exit_window", "atr_window",
                     "max_units", "max_holding_bars", "max_positions"):
            setattr(self, name, int(getattr(self, name)))
        for name in ("stop_atr", "add_atr", "risk_per_trade", "trailing_ratio",
                     "trailing_start_r", "max_position_ratio",
                     "price_buffer", "exit_price_buffer"):
            setattr(self, name, float(getattr(self, name)))

    def _validate(self) -> None:
        if self.entry_window < 2:
            raise ValueError(f"entry_window 至少为 2，实际 {self.entry_window}")
        if self.exit_window < 2:
            raise ValueError(f"exit_window 至少为 2，实际 {self.exit_window}")
        if self.exit_window >= self.entry_window:
            raise ValueError(
                f"exit_window 必须小于 entry_window（进场慢、出场快），"
                f"实际 {self.exit_window} >= {self.entry_window}")
        if self.stop_atr <= 0:
            raise ValueError(f"stop_atr 必须为正，实际 {self.stop_atr}")
        if self.max_units < 1:
            raise ValueError(f"max_units 至少为 1，实际 {self.max_units}")
        if self.max_positions < 1:
            raise ValueError(
                f"max_positions 至少为 1，实际 {self.max_positions}")
        if self.add_atr < 0:
            raise ValueError("add_atr 不能为负")
        if self.exit_price_buffer < self.price_buffer:
            raise ValueError(
                f"卖出缓冲必须不小于买入缓冲（错过卖出是实亏），"
                f"实际 {self.exit_price_buffer} < {self.price_buffer}")

    # ------------------------------------------------------------ 生命周期

    def on_init(self) -> None:
        self.write_log(
            f"初始化 入场{self.entry_window}日高 离场{self.exit_window}日低 "
            f"止损{self.stop_atr}×ATR{self.atr_window} "
            f"单笔风险{self.risk_per_trade:.1%} 最多{self.max_units}份 "
            f"并发{self.max_positions}只")
        self.load_bars(max(self.entry_window, self.atr_window) + 20)

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        self.write_log(f"策略停止。累计买卖 {self.trade_count} 次")
        self.write_log(self.manager.summary())

    # ------------------------------------------------------------ 主循环

    def on_bar(self, bar: BarData) -> None:
        if bar.suspended or bar.close_price <= 0:
            return
        symbol = bar.vt_symbol
        entry_ch = self.entry_ch.get(symbol)
        if entry_ch is None:
            return

        self._bar_count += 1
        atr = self.atr[symbol].update(bar.high_price, bar.low_price,
                                      bar.close_price)

        # 关键顺序：**先用旧通道判断，再把当根 Bar 推进去**。
        # 反了的话「突破 N 日新高」会变成恒真 ——
        # 今天的最高价当然是含今天在内的最高价之一
        upper = entry_ch.upper
        lower = self.exit_ch[symbol].lower
        entry_ch.update(bar.high_price, bar.low_price)
        self.exit_ch[symbol].update(bar.high_price, bar.low_price)

        if not self.trading or atr is None or atr <= 0:
            return

        if self.manager.has(symbol):
            self._manage(symbol, bar, lower, atr)
        elif upper is not None and bar.close_price > upper:
            # 并发上限：先到先得。没有这个限制的话，
            # 单笔 1% 风险在组合层面会叠加成失控的总敞口
            if len(self.manager.positions) >= self.max_positions:
                return
            self._enter(symbol, bar, atr)

    # ------------------------------------------------------------ 建仓

    def _enter(self, symbol: str, bar: BarData, atr: float) -> None:
        entry = bar.close_price
        stop = entry - self.stop_atr * atr
        if stop <= 0:
            # ATR 大于价格的一半，说明波动极端，这种标的不碰
            logger.debug("%s ATR %.3f 过大，止损价为负，跳过", symbol, atr)
            return

        total = self._total_value(bar)
        volume = self.manager.position_size(total, entry, stop)
        if volume <= 0:
            return

        price = entry * (1 + self.price_buffer)
        if not self.buy(symbol, price, volume):
            return

        self.manager.open(symbol, entry, volume, stop,
                          bar_index=self._bar_count)
        self.units[symbol] = 1
        self._next_add[symbol] = entry + self.add_atr * atr
        self.trade_count += 1
        self.write_log(
            f"突破买入 {symbol} @{entry:.3f} 止损 {stop:.3f} "
            f"({self.stop_atr}×ATR={atr:.3f}) 数量 {volume:.0f}")

    # ------------------------------------------------------------ 持仓管理

    def _manage(self, symbol: str, bar: BarData, lower: float | None,
                atr: float) -> None:
        reason = self.manager.check(symbol, bar.high_price, bar.low_price,
                                    bar.close_price, self._bar_count)

        # 通道离场：跌破 exit_window 日最低价。
        # 与止损是两套独立机制 —— 止损管「单笔亏多少」，
        # 通道管「趋势是不是结束了」
        if reason is None and lower is not None and bar.close_price < lower:
            reason = EXIT_SIGNAL

        if reason is not None:
            self._exit(symbol, bar, reason)
            return

        if self.add_atr > 0:
            self._maybe_add(symbol, bar, atr)

    def _maybe_add(self, symbol: str, bar: BarData, atr: float) -> None:
        """金字塔加仓：让对的交易变大"""
        if self.units.get(symbol, 0) >= self.max_units:
            return
        trigger = self._next_add.get(symbol)
        if trigger is None or bar.close_price < trigger:
            return

        pos = self.manager.get(symbol)
        if pos is None:
            return

        total = self._total_value(bar)
        stop = bar.close_price - self.stop_atr * atr
        volume = self.manager.position_size(total, bar.close_price, stop)
        if volume <= 0:
            return

        if not self.buy(symbol, bar.close_price * (1 + self.price_buffer),
                        volume):
            return

        pos.volume += volume
        # 加仓后把整体止损上移到最新一份的止损位 ——
        # 否则份数越多，总风险敞口越大，最终一次反向就打掉全部浮盈
        if stop > pos.current_stop:
            pos.current_stop = stop
        self.units[symbol] = self.units.get(symbol, 1) + 1
        self._next_add[symbol] = bar.close_price + self.add_atr * atr
        self.trade_count += 1
        self.write_log(
            f"加仓 {symbol} 第 {self.units[symbol]} 份 @{bar.close_price:.3f} "
            f"止损上移至 {pos.current_stop:.3f}")

    def _exit(self, symbol: str, bar: BarData, reason: str) -> None:
        volume = self.get_pos(symbol)
        if volume <= 0:
            self.manager.close(symbol, reason)
            self.units.pop(symbol, None)
            self._next_add.pop(symbol, None)
            return

        price = bar.close_price * (1 - self.exit_price_buffer)
        if not self.sell(symbol, price, volume):
            return

        pos = self.manager.get(symbol)
        r = pos.r_multiple(bar.close_price) if pos else 0.0
        self.manager.close(symbol, reason)
        self.units.pop(symbol, None)
        self._next_add.pop(symbol, None)
        self.trade_count += 1
        self.write_log(f"{reason} {symbol} @{bar.close_price:.3f} "
                       f"（{r:+.2f}R）")

    # ------------------------------------------------------------ 辅助

    def _total_value(self, bar: BarData) -> float:
        """现金 + 当前标的持仓市值。

        多标的场景下这是近似值：只算了当前 Bar 这只票的市值。
        用总资产而非可用现金的理由是避免仓位随建仓越滚越小。
        """
        value = self.get_cash()
        held = self.get_pos(bar.vt_symbol)
        if held > 0:
            value += held * bar.close_price
        return value

    @property
    def exit_stats(self) -> dict[str, int]:
        """离场原因归因，供报告使用"""
        return self.manager.exit_stats


__all__ = ["BreakoutStrategy", "EXIT_STOP", "EXIT_TARGET", "EXIT_TRAILING",
           "EXIT_TIME", "EXIT_SIGNAL"]
