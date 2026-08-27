"""回测引擎。

事件驱动的日线/分钟级回测，严格遵守 A 股规则：
- T+1：当日买入次日才可卖
- 100 股整数倍买入
- 涨跌停价不成交、停牌跳过
- **信号用 T 日收盘数据，成交发生在 T+1 开盘**，杜绝前视偏差
"""
import logging
from collections import defaultdict
from datetime import datetime

import pandas as pd

from ..config import CostConfig
from ..core.constants import Direction, OrderType, Status, get_price_limit
from ..core.objects import BarData, OrderData, OrderRequest, TradeData
from ..gateway.sim_gateway import calc_cost
from ..risk.drawdown import DrawdownController
from ..strategy.base import StrategyBase
from ..utils.symbol import normalize, split_vt_symbol
from .performance import PerformanceStats, calculate_stats

logger = logging.getLogger(__name__)


class BacktestEngine:
    """历史回测引擎"""

    def __init__(self, initial_capital: float = 1_000_000,
                 cost: CostConfig | None = None,
                 price_limit_ratio: float | None = None,
                 lot_size: int = 100,
                 drawdown: "DrawdownController | None" = None) -> None:
        """
        :param price_limit_ratio: 强制统一的涨跌停幅度。
            默认 None = **按标的代码前缀自动判定**（主板 10%、
            创业板/科创板 20%、北交所 30%）。

            曾经这里硬编码全局 10%，而沪深300 中有 53 只（18%）是 20% 的
            创业板/科创板，它们涨跌超过 10% 的交易日会被误判为涨跌停而拒单，
            回测结果对这些标的系统性失真。仅在需要压制板块差异做对照实验时
            才显式传值。
        :param lot_size: 一手股数，A 股为 100。

            ⚠ 与后复权数据配合时需注意：后复权价高于真实价（茅台约 6 倍），
            一手的**名义成本**因此被放大，会造成本不该有的资金闲置甚至完全买不进。
            实测 100 万本金 / 10 只持仓时，沪深300 中有 36 只取整后为 0 股。
            研究阶段可临时设为 1 以消除该假象，但那样得到的成交量不可实盘复现。
            根本解法是按真实价取整（需复权因子），见 docs/TASKS.md。
        :param drawdown: 回撤控制器。传入后回测与实盘走同一套判定逻辑，
            否则回测会高估策略表现 —— 实盘被回撤控制拦下的仓位，
            回测里却照买不误。
        """
        self.initial_capital = initial_capital
        self.cost = cost or CostConfig()
        self.price_limit_ratio = price_limit_ratio
        self.lot_size = lot_size
        self.drawdown = drawdown

        self.cash: float = initial_capital
        #: vt_symbol -> {"volume": 总量, "available": 可卖, "price": 成本}
        self.positions: dict[str, dict] = {}

        self.strategy: StrategyBase | None = None
        self.history: dict[datetime, dict[str, BarData]] = {}
        self.pending_orders: list[OrderRequest] = []
        #: 标的池。为 None 时策略自行决定交易哪些标的
        self.universe = None

        self.trades: list[TradeData] = []
        self.orders: list[OrderData] = []
        self.equity_curve: dict[datetime, float] = {}

        self._order_count = 0
        self._trade_count = 0
        #: 因取整后不足一手而未能下出的委托，按标的计数。
        #: 后复权价被抬高后这类情况会激增（实测 100 万/10 只时有 36 只完全买不进），
        #: 静默丢弃等于把这些标的悄悄剔出标的池，必须统计出来告警
        self.undersized_orders: dict[str, int] = {}
        #: 被回撤控制拦下的买单数
        self.drawdown_blocked: int = 0
        self._current_bars: dict[str, BarData] = {}
        self._prev_bars: dict[str, BarData] = {}
        self._current_dt: datetime | None = None

    # ------------------------------------------------------------ 数据装载

    def load_data(self, bars: list[BarData]) -> None:
        """装载历史 K 线，按时间戳聚合成截面"""
        grouped: dict[datetime, dict[str, BarData]] = defaultdict(dict)
        for bar in bars:
            grouped[bar.datetime][bar.vt_symbol] = bar
        self.history = dict(sorted(grouped.items()))
        logger.info("已装载 %d 个时间截面，标的数 %d",
                    len(self.history), len({b.vt_symbol for b in bars}))

    def add_strategy(self, strategy_class: type[StrategyBase],
                     vt_symbols: list[str], setting: dict | None = None) -> None:
        # 必须归一化：策略普遍用 vt_symbol 作字典 key，而 BarData.vt_symbol
        # 永远是 `600519.SSE` 格式。若调用方传入 `600519.SH`，key 对不上，
        # 策略会静默地一根 Bar 都处理不到 —— 不报错，只是永远没有信号。
        vt_symbols = [normalize(s) for s in vt_symbols]
        self.strategy = strategy_class(self, strategy_class.__name__, vt_symbols, setting)

    def set_universe(self, provider) -> None:
        """设置标的池。选股型策略通过 self.engine.get_universe(dt) 取当日可选标的。"""
        self.universe = provider
        report = provider.describe_bias()
        logger.info("标的池已设置，%d 个标的", report.size)
        if not report.is_clean:
            logger.warning("标的池存在偏差，回测收益会被系统性高估：\n%s", report.summary())

    def get_universe(self, dt=None) -> list[str]:
        """取指定日期（默认当前回测时点）的可交易标的"""
        if self.universe is None:
            return list(self.strategy.vt_symbols) if self.strategy else []
        return self.universe.get_universe(dt or self._current_dt)

    # ------------------------------------------------------------ 主循环

    def run(self) -> PerformanceStats:
        if not self.strategy:
            raise RuntimeError("未添加策略，请先调用 add_strategy()")
        if not self.history:
            raise RuntimeError("未装载数据，请先调用 load_data()")

        self.strategy.on_init()
        self.strategy.inited = True
        self.strategy.on_start()
        self.strategy.trading = True

        for dt, bars in self.history.items():
            self._current_dt = dt
            self._prev_bars = self._current_bars
            self._current_bars = bars

            # 1) 新交易日开始：释放 T+1 冻结
            self._settle_t1()
            # 2) 用开盘价撮合上一根 Bar 产生的委托，避免前视偏差
            self._match_pending(bars)
            # 3) 推送行情给策略，策略在此产生新信号
            self._push_bars(bars)
            # 4) 按收盘价记录净值，并推给回撤控制器
            equity = self._calc_equity(bars)
            self.equity_curve[dt] = equity
            if self.drawdown is not None:
                self.drawdown.update(equity)

        self.strategy.trading = False
        self.strategy.on_stop()

        equity = pd.Series(self.equity_curve).sort_index()
        stats = calculate_stats(equity, self.trades, self.initial_capital)
        return stats

    def _push_bars(self, bars: dict[str, BarData]) -> None:
        try:
            self.strategy.on_bars(bars)
            for bar in bars.values():
                self.strategy.on_bar(bar)
        except Exception:
            logger.exception("策略处理 Bar 异常 dt=%s", self._current_dt)

    def _settle_t1(self) -> None:
        """新交易日：昨日买入的股票解冻可卖"""
        for pos in self.positions.values():
            pos["available"] = pos["volume"]

    # ------------------------------------------------------------ 撮合

    def _match_pending(self, bars: dict[str, BarData]) -> None:
        """用当根 Bar 的开盘价撮合挂起委托；不可成交的直接作废（不留隔日单）"""
        pending, self.pending_orders = self.pending_orders, []

        # 卖单必须排在买单之前：调仓时先卖出释放资金，买单才有钱成交。
        # 否则同一批调仓指令会因「资金不足」大面积拒单，回测结果严重失真。
        pending.sort(key=lambda r: 0 if r.direction == Direction.SHORT else 1)

        for req in pending:
            bar = bars.get(req.vt_symbol)
            if bar is None or bar.suspended:
                self._reject(req, "标的停牌或无行情")
                continue

            prev = self._prev_bars.get(req.vt_symbol)
            pre_close = prev.close_price if prev else bar.open_price
            ratio = self._limit_ratio(req.vt_symbol)
            limit_up = round(pre_close * (1 + ratio), 2)
            limit_down = round(pre_close * (1 - ratio), 2)

            # 一字涨停买不进，一字跌停卖不出
            if req.direction == Direction.LONG and bar.open_price >= limit_up:
                self._reject(req, "开盘涨停，无法买入")
                continue
            if req.direction == Direction.SHORT and bar.open_price <= limit_down:
                self._reject(req, "开盘跌停，无法卖出")
                continue

            # 限价单需价格可达
            if req.order_type == OrderType.LIMIT:
                if req.direction == Direction.LONG and req.price < bar.open_price:
                    self._reject(req, "限价低于开盘价，未成交")
                    continue
                if req.direction == Direction.SHORT and req.price > bar.open_price:
                    self._reject(req, "限价高于开盘价，未成交")
                    continue

            # 成交价 = 开盘价 + 滑点（买入向上、卖出向下）
            slip = self.cost.slippage_tick * 0.01
            price = bar.open_price + (slip if req.direction == Direction.LONG else -slip)
            price = max(min(price, limit_up), limit_down)
            self._fill(req, price)

    def _limit_ratio(self, vt_symbol: str) -> float:
        """该标的的涨跌停幅度。显式指定时用指定值，否则按板块自动判定。"""
        if self.price_limit_ratio is not None:
            return self.price_limit_ratio
        return get_price_limit(vt_symbol)

    def _fill(self, req: OrderRequest, price: float) -> None:
        volume = req.volume
        fee = calc_cost(price, volume, req.direction, self.cost)

        if req.direction == Direction.LONG:
            need = price * volume + fee
            if need > self.cash:
                self._reject(req, "资金不足")
                return
            self.cash -= need
            pos = self.positions.setdefault(
                req.vt_symbol, {"volume": 0.0, "available": 0.0, "price": 0.0})
            pos["price"] = (pos["price"] * pos["volume"] + price * volume) / (pos["volume"] + volume)
            pos["volume"] += volume
            # T+1：当日买入不可卖，available 不增加
        else:
            pos = self.positions.get(req.vt_symbol)
            if not pos or pos["available"] < volume:
                self._reject(req, "可卖数量不足")
                return
            self.cash += price * volume - fee
            pos["volume"] -= volume
            pos["available"] -= volume
            if pos["volume"] <= 0:
                self.positions.pop(req.vt_symbol, None)

        self._trade_count += 1
        symbol, exchange = split_vt_symbol(req.vt_symbol)
        trade = TradeData(
            symbol=symbol, exchange=exchange,
            orderid=f"BT{self._order_count:08d}", tradeid=f"BT{self._trade_count:08d}",
            direction=req.direction, price=price, volume=volume, commission=fee,
            datetime=self._current_dt, reference=req.reference, gateway_name="BACKTEST",
        )
        self.trades.append(trade)
        try:
            self.strategy.on_trade(trade)
        except Exception:
            logger.exception("策略处理成交回报异常")

    def _reject(self, req: OrderRequest, msg: str) -> None:
        symbol, exchange = split_vt_symbol(req.vt_symbol)
        order = OrderData(
            symbol=symbol, exchange=exchange, orderid=f"BT{self._order_count:08d}",
            direction=req.direction, price=req.price, volume=req.volume,
            status=Status.REJECTED, message=msg, datetime=self._current_dt,
            reference=req.reference, gateway_name="BACKTEST",
        )
        self.orders.append(order)
        logger.debug("回测拒单 %s %s: %s", self._current_dt, req.vt_symbol, msg)

    # ------------------------------------------------------------ 策略调用的接口

    def send_order(self, strategy_name: str, vt_symbol: str, direction: Direction,
                   price: float, volume: float,
                   order_type: OrderType = OrderType.LIMIT) -> str:
        """策略下单：不立即成交，挂到下一根 Bar 开盘撮合"""
        if volume <= 0:
            return ""
        # 回撤控制：达到任一档位即停止开新仓，卖出始终放行
        if (direction == Direction.LONG and self.drawdown is not None
                and not self.drawdown.allow_open()):
            self.drawdown_blocked += 1
            return ""

        # 买入向下取整到一手
        if direction == Direction.LONG:
            volume = int(volume // self.lot_size) * self.lot_size
            if volume <= 0:
                # 不足一手买不了。这在后复权数据上很常见 ——
                # 价格被抬高后一手的成本可达实际的数倍，高价股会被静默排除。
                # 计数以便在报告中告警，而不是无声地少一个候选标的。
                self.undersized_orders[vt_symbol] = (
                    self.undersized_orders.get(vt_symbol, 0) + 1)
                return ""

        self._order_count += 1
        symbol, exchange = split_vt_symbol(vt_symbol)
        req = OrderRequest(symbol=symbol, exchange=exchange, direction=direction,
                           order_type=order_type, price=price, volume=volume,
                           reference=strategy_name)
        self.pending_orders.append(req)
        return f"BACKTEST.BT{self._order_count:08d}"

    def cancel_order(self, vt_orderid: str) -> None:
        """回测中委托只存活一根 Bar，撤单无实际意义"""

    def cancel_all(self, strategy_name: str) -> None:
        self.pending_orders = [
            r for r in self.pending_orders if r.reference != strategy_name
        ]

    def get_cash(self) -> float:
        return self.cash

    def get_pos(self, vt_symbol: str) -> float:
        return self.positions.get(vt_symbol, {}).get("volume", 0)

    def load_bars(self, strategy, days: int, interval: str = "1d") -> None:
        """回测数据已在 load_data 中一次性装载，此处无需额外操作"""

    # ------------------------------------------------------------ 净值

    def _calc_equity(self, bars: dict[str, BarData]) -> float:
        market_value = 0.0
        for vt_symbol, pos in self.positions.items():
            bar = bars.get(vt_symbol)
            price = bar.close_price if bar else pos["price"]
            market_value += pos["volume"] * price
        return self.cash + market_value

    def get_equity_df(self) -> pd.DataFrame:
        s = pd.Series(self.equity_curve).sort_index()
        df = pd.DataFrame({"equity": s})
        df["returns"] = df["equity"].pct_change().fillna(0)
        df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
        return df

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "datetime": t.datetime, "symbol": t.vt_symbol,
            "direction": t.direction.value, "price": t.price,
            "volume": t.volume, "commission": round(t.commission, 2),
            "amount": round(t.price * t.volume, 2), "strategy": t.reference,
        } for t in self.trades])
