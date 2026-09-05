"""按外部信号打分选股。

## 这个模块的定位

它不产生信号，只**执行**信号。分数从外部传入 —— 可以是 Qlib 的
机器学习预测、自己算的因子、甚至手工名单。

这样分工的理由：Qlib 擅长找信号（Alpha158/360 因子集 + 一整套 ML 模型），
但它的回测对 A 股规则处理得很粗 —— T+1、涨跌停、整手、
停牌不可交易、回撤控制，这些是本项目引擎的强项。

让 Qlib 只输出「每天每只股票的分数」，剩下的交给本引擎，
两边各做各擅长的事。

## 分数面板的口径

``scores`` 是 ``index=日期, columns=vt_symbol`` 的 DataFrame。
取值时用 **asof 语义**：某个调仓日取「不晚于它的最近一期分数」。

⚠ **分数本身的 point-in-time 正确性由调用方保证。**
本模块只保证不会用到 ``scores`` 里晚于当前日期的行 ——
但如果传进来的分数是用未来数据训练出来的（比如模型在测试集上
重新训练过），这里拦不住。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..core.objects import BarData
from .portfolio import PortfolioStrategy

logger = logging.getLogger(__name__)

#: 日成交额下限（元）。选出来买不进去的票没有意义
MIN_TURNOVER = 50_000_000


class SignalRankStrategy(PortfolioStrategy):
    """按外部分数从高到低选股，等权持有"""

    parameters = ["max_holdings", "rebalance_days", "rebalance_phase"]
    variables = ["inited", "trading", "pos", "last_selection",
                 "rebalance_count"]

    #: 持仓只数
    max_holdings: int = 30
    #: 调仓间隔（交易日）。20 ≈ 一个月
    rebalance_days: int = 20

    def __init__(self, engine, strategy_name, vt_symbols, setting=None,
                 scores: pd.DataFrame | None = None,
                 min_turnover: float = MIN_TURNOVER):
        """
        :param scores: index=日期, columns=vt_symbol 的分数面板。
            不传则每次选股返回空 —— **不静默降级到别的规则**，
            那会让回测跑出一条看似正常的曲线而你以为测的是模型信号。
        """
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._coerce_types()
        self._validate()

        self.scores = scores
        self.min_turnover = float(min_turnover)
        self.rebalance_count: int = 0
        #: 最近一次的分数，供复盘查看
        self.last_scores: dict[str, float] = {}
        self._warned = False

        # 预排序 + 预转 numpy，避免每个调仓日都做一次 searchsorted 之外的开销
        if scores is not None and not scores.empty:
            self.scores = scores.sort_index()
            self._score_dates = self.scores.index.to_numpy()
        else:
            self._score_dates = np.array([], dtype="datetime64[ns]")

    def _coerce_types(self) -> None:
        """参数寻优会从 DataFrame 传 float64，用作下标会 TypeError"""
        self.max_holdings = int(self.max_holdings)
        self.rebalance_days = int(self.rebalance_days)
        self.rebalance_phase = int(self.rebalance_phase)

    def _validate(self) -> None:
        if self.max_holdings < 1:
            raise ValueError(f"max_holdings 至少为 1，实际 {self.max_holdings}")
        if self.rebalance_days < 1:
            raise ValueError(
                f"rebalance_days 至少为 1，实际 {self.rebalance_days}")
        if not 0 <= self.rebalance_phase < self.rebalance_days:
            raise ValueError(
                f"rebalance_phase 须在 [0, {self.rebalance_days}) 内，"
                f"实际 {self.rebalance_phase}")

    # ------------------------------------------------------------ 生命周期

    def on_init(self) -> None:
        n = 0 if self.scores is None else len(self.scores)
        self.write_log(f"初始化 分数面板 {n} 期 持仓{self.max_holdings}只 "
                       f"每{self.rebalance_days}日调仓 "
                       f"相位{self.rebalance_phase}")

    def on_stop(self) -> None:
        self.write_log(f"策略停止。调仓 {self.rebalance_count} 次")

    # ------------------------------------------------------------ 选股

    def select(self, bars: dict[str, BarData], candidates: list[str]) -> list[str]:
        self.rebalance_count += 1
        self.last_scores = {}

        row = self._score_asof(self._current_datetime(bars))
        if row is None:
            if not self._warned:
                logger.error("%s 没有可用分数面板 —— 回测将全程空仓",
                             self.strategy_name)
                self._warned = True
            return []

        picked: list[tuple[float, str]] = []
        for vt_symbol in candidates:
            bar = bars.get(vt_symbol)
            if bar is None or bar.turnover < self.min_turnover:
                continue
            value = row.get(vt_symbol)
            if value is None or not np.isfinite(value):
                continue
            self.last_scores[vt_symbol] = float(value)
            picked.append((float(value), vt_symbol))

        # 降序：分数最高的排在最前
        picked.sort(key=lambda kv: kv[0], reverse=True)
        return [s for _, s in picked]

    def _score_asof(self, dt) -> dict[str, float] | None:
        """取「不晚于 dt 的最近一期」分数。

        asof 而非精确匹配：模型可能只在部分日期出分（如月末），
        精确匹配会让绝大多数调仓日拿不到分数而空仓。
        """
        if self.scores is None or dt is None or len(self._score_dates) == 0:
            return None
        pos = np.searchsorted(self._score_dates,
                              np.datetime64(pd.Timestamp(dt)), side="right") - 1
        if pos < 0:
            return None
        row = self.scores.iloc[pos]
        return {k: v for k, v in row.items() if pd.notna(v)}

    @staticmethod
    def _current_datetime(bars: dict[str, BarData]):
        for bar in bars.values():
            return bar.datetime
        return None
