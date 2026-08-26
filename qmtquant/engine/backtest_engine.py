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
from ..core.constants import Direction, OrderType, Status
from ..core.objects import BarData, OrderData, OrderRequest, TradeData
from ..gateway.sim_gateway import calc_cost
from ..strategy.base import StrategyBase
from ..utils.symbol import split_vt_symbol
from .performance import PerformanceStats, calculate_stats

logger = logging.getLogger(__name__)


class BacktestEngine:
    """历史回测引擎"""

    def __init__(self, initial_capital: float = 1_000_000,
                 cost: CostConfig | None = None,
                 price_limit_ratio: float = 0.10) -> None:
        """
        :param price_limit_ratio: 涨跌停幅度，主板 10%，创业板/科创板需按标的调整
        """
        self.initial_capital = initial_capital
        self.cost = cost or CostConfig()
        self.price_limit_ratio = price_limit_ratio

        self.cash: float = initial_capital
        #: vt_symbol -> {"volume": 总量, "available": 可卖, "price": 成本}
        self.positions: dict[str, dict] = {}

        self.strategy: StrategyBase | None = None
        self.history: dict[datetime, dict[str, BarData]] = {}
        self.pending_orders: list[OrderRequest] = []

        self.trades: list[TradeData] = []
        self.orders: list[OrderData] = []
        self.equity_curve: dict[datetime, float] = {}

        self._order_count = 0
        self._trade_count = 0
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
        self.strategy = strategy_class(self, strategy_class.__name__, vt_symbols, setting)

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
            # 4) 按收盘价记录净值
            self.equity_curve[dt] = self._calc_equity(bars)

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
        for req in pending:
            bar = bars.get(req.vt_symbol)
            if bar is None or bar.suspended:
                self._reject(req, "标的停牌或无行情")
                continue

            prev = self._prev_bars.get(req.vt_symbol)
            pre_close = prev.close_price if prev else bar.open_price
            limit_up = round(pre_close * (1 + self.price_limit_ratio), 2)
            limit_down = round(pre_close * (1 - self.price_limit_ratio), 2)

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
        # 买入向下取整到 100 股
        if direction == Direction.LONG:
            volume = int(volume // 100) * 100
            if volume <= 0:
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
