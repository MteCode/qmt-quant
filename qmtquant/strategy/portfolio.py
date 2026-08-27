"""选股型（组合）策略基类。

与择时型策略的区别：择时型关心「某只股票现在该不该买」，
选股型关心「今天这一篮子股票该怎么配」。

本基类把「目标持仓 → 委托指令」这段容易写错的逻辑抽出来统一处理，
子类只需实现 `select()` 返回选中的标的即可。
"""
import logging
from datetime import datetime

from ..core.objects import BarData
from .base import StrategyBase

logger = logging.getLogger(__name__)


class PortfolioStrategy(StrategyBase):
    """按周期调仓的等权选股策略基类"""

    parameters = ["rebalance_days", "max_holdings", "cash_buffer", "price_buffer"]

    #: 每隔多少个交易日调仓一次
    rebalance_days: int = 20
    #: 最多同时持有多少只
    max_holdings: int = 10
    #: 预留现金比例，防止因价格波动导致买单资金不足
    cash_buffer: float = 0.05
    #: 限价单相对收盘价的缓冲。信号在 T 日收盘产生、T+1 开盘成交，
    #: 次日跳空超过缓冲时限价单会失效（买单低于开盘价、卖单高于开盘价）。
    #: 实测 2% 在快速上涨行情中会踏空，故取 3%。
    price_buffer: float = 0.03

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._bar_count: int = 0
        self._last_rebalance: datetime | None = None
        #: 最近一次调仓选中的标的，供复盘查看
        self.last_selection: list[str] = []

    # ------------------------------------------------------------ 子类实现

    def select(self, bars: dict[str, BarData], candidates: list[str]) -> list[str]:
        """从 candidates 中选出要持有的标的。

        :param bars: 当前时点的行情截面
        :param candidates: 当日可交易标的（已做上市日等过滤）
        :return: 选中的 vt_symbol 列表，按优先级排序
        """
        raise NotImplementedError

    def update_indicators(self, bars: dict[str, BarData]) -> None:
        """每根 Bar 都会调用，用于维护指标窗口。子类按需覆写。"""

    # ------------------------------------------------------------ 主循环

    def on_bars(self, bars: dict[str, BarData]) -> None:
        self._bar_count += 1
        self.update_indicators(bars)

        if not self.trading:
            return
        if self._bar_count % self.rebalance_days != 0:
            return

        candidates = [s for s in self.engine.get_universe()
                      if s in bars and not bars[s].suspended]
        if not candidates:
            return

        selected = self.select(bars, candidates)[:self.max_holdings]
        self.last_selection = selected
        self.rebalance(selected, bars)

    # ------------------------------------------------------------ 调仓

    def rebalance(self, targets: list[str], bars: dict[str, BarData]) -> None:
        """调仓到目标持仓（等权）。

        先发卖单再发买单 —— 回测引擎会保证卖单优先撮合以释放资金，
        实盘则依赖 T+1 之前已有的可用资金。
        """
        target_set = set(targets)
        held = {s for s, v in self.pos.items() if v > 0}

        # --- 卖出不在目标里的
        for vt_symbol in sorted(held - target_set):
            bar = bars.get(vt_symbol)
            if bar is None or bar.suspended:
                # 停牌卖不掉，下个调仓日再试
                continue
            volume = self.get_pos(vt_symbol)
            if volume > 0:
                self.sell(vt_symbol, bar.close_price * (1 - self.price_buffer), volume)

        # --- 买入新增的
        new_symbols = [s for s in targets if s not in held]
        if not new_symbols:
            return

        # 等权：用总资产而非可用现金计算，避免每次调仓仓位越滚越小。
        # 卖单尚未成交，此处按「目标持仓数」均分总资产。
        total_value = self._estimate_total_value(bars)
        budget_per_name = total_value * (1 - self.cash_buffer) / max(len(targets), 1)

        for vt_symbol in new_symbols:
            bar = bars.get(vt_symbol)
            if bar is None or bar.suspended or bar.close_price <= 0:
                continue
            # 不在此处取整：整手约束由引擎统一处理，
            # 策略自行取整会让「预算不足一手」的情况被静默丢弃，
            # 引擎统计不到，报告里也就看不到标的被排除
            volume = budget_per_name / bar.close_price
            self.buy(vt_symbol, bar.close_price * (1 + self.price_buffer), volume)

    def _estimate_total_value(self, bars: dict[str, BarData]) -> float:
        """现金 + 持仓市值。持仓无当日行情时按成本价估。"""
        value = self.get_cash()
        for vt_symbol, volume in self.pos.items():
            if volume <= 0:
                continue
            bar = bars.get(vt_symbol)
            if bar is not None:
                value += volume * bar.close_price
        return value
