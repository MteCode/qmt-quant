"""ETF 动量轮动 —— 规则版基线。

## 为什么先做规则版而不是直接训模型

没有基线的话，模型跑出个夏普 1.2 你没法判断它是真有用，还是还不如一条
「买过去 20 日涨得最多的那几只」的简单规则。动量轮动是这个问题上公认的
基线，先把它跑出来，模型才有比较的对象。

而且日线有 10 年数据，规则版今天就能验证；分钟线只有 1 年，模型过拟合
风险大得多。

## 为什么 ETF 能做而股票不能

A 股股票 T+1，当日买入当日不可卖 —— 同一套逻辑在股票上跑，止损执行不了、
尾盘平不掉、挂单排在涨停板后面成交不了，这些都实盘验证过。
**ETF 是 T+0**，买卖当日可逆，这是这条线成立的前提。

## 硬性约束

- 资金 20 万，单票 ≤ 20% → **最少持有 5 只**，否则风控按单票占比拒单
- 单笔委托 ≤ 5 万：20 万 / 5 只 = 4 万，刚好在限内
- 总仓位 ≤ 95%

## 绝对动量与防御腿

只做横截面排名（谁最强）会在熊市里买到「跌得最少的」。所以加一道绝对
动量门槛：动量低于 min_momentum 的不持有；全场都不达标时资金转入防御腿
（债券/货币 ETF），而不是硬着头皮满仓。

## 调仓相位

基类文档记录过一个实测：同一份信号、同一组参数，20 日调仓下仅改变相位
（0/4/8/12/16），4 年半累计收益从 -9.62% 到 +73.02%。所以这个策略的回测
必须跑遍相位看分布 —— 只跑一次得到的数字测的不是策略好坏，而是「你碰巧
从哪天开始」。
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

from qmtquant.core.objects import BarData
from qmtquant.strategy.portfolio import PortfolioStrategy

logger = logging.getLogger(__name__)


class EtfRotationStrategy(PortfolioStrategy):
    """按多周期动量排名轮动持有 ETF。"""

    parameters = PortfolioStrategy.parameters + [
        "lookbacks", "weights", "min_momentum", "defensive",
        "vol_adjust", "vol_window",
    ]

    variables = PortfolioStrategy.variables + ["last_selection"]

    #: 动量回看窗口（交易日）。多周期加权比单周期稳 —— 单看 20 日会被
    #: 一次急涨带偏，单看 120 日又太迟钝
    lookbacks: tuple = (20, 60, 120)
    #: 各周期权重，与 lookbacks 等长；长度不匹配时退化为等权
    weights: tuple = (0.5, 0.3, 0.2)
    #: 绝对动量下限。加权动量低于它就不持有 —— 只做横截面排名会在熊市里
    #: 买到「跌得最少的」，那不是收益来源
    min_momentum: float = 0.0
    #: 防御腿：全场都不达标时持有它（债券/货币 ETF）。留空则空仓
    defensive: str = ""
    #: 是否按波动率调整动量（动量/波动，即风险调整后动量）
    vol_adjust: bool = True
    #: 算波动率的窗口
    vol_window: int = 20

    def __init__(self, engine: Any, strategy_name: str,
                 vt_symbols: list[str], setting: dict | None = None):
        # 单票 ≤ 20% 是硬性风控，5 只刚好卡在限上。基类默认 10 只对
        # 20 万本金来说每只才 2 万，手续费占比偏高
        self.max_holdings = 5
        self.rebalance_days = 10

        super().__init__(engine, strategy_name, vt_symbols, setting)

        self._closes: dict[str, deque] = {}
        self._maxlen = max(self.lookbacks) + self.vol_window + 5

    # ------------------------------------------------------------ 指标

    def update_indicators(self, bars: dict[str, BarData]) -> None:
        for vt, bar in bars.items():
            if bar.close_price <= 0:
                continue
            dq = self._closes.get(vt)
            if dq is None:
                dq = self._closes[vt] = deque(maxlen=self._maxlen)
            dq.append(float(bar.close_price))

    def _momentum(self, vt: str) -> float | None:
        """多周期加权动量，可选按波动率调整。

        返回 None 表示历史不足 —— 不足时必须排除而不是当 0 处理，
        否则新上市的 ETF 会以「动量 0」混进排名，在全场普跌时反而排前面。
        """
        dq = self._closes.get(vt)
        if dq is None:
            return None
        n = len(dq)
        need = max(self.lookbacks) + 1
        if n < need:
            return None

        px = list(dq)
        lbs = list(self.lookbacks)
        ws = list(self.weights)
        if len(ws) != len(lbs):
            ws = [1.0 / len(lbs)] * len(lbs)

        score = 0.0
        for lb, w in zip(lbs, ws):
            past = px[-(lb + 1)]
            if past <= 0:
                return None
            score += w * (px[-1] / past - 1.0)

        if not self.vol_adjust:
            return score

        win = px[-(self.vol_window + 1):]
        rets = [win[i] / win[i - 1] - 1.0
                for i in range(1, len(win)) if win[i - 1] > 0]
        if len(rets) < 2:
            return score
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
        vol = var ** 0.5
        # 波动率过小时不放大 —— 除以一个接近 0 的数会把货币 ETF 顶到第一
        return score / vol if vol > 1e-4 else score

    # ------------------------------------------------------------ 选股

    def select(self, bars: dict[str, BarData],
               candidates: list[str]) -> list[str]:
        scored: list[tuple[str, float]] = []
        for vt in candidates:
            if vt == self.defensive:
                continue          # 防御腿不参与横截面排名
            m = self._momentum(vt)
            if m is None or m <= self.min_momentum:
                continue
            scored.append((vt, m))

        if not scored:
            # 全场无正动量：转防御腿而非硬着头皮满仓
            if self.defensive and self.defensive in bars:
                self.write_log("无标的达到绝对动量门槛，转入防御腿 "
                               f"{self.defensive}")
                return [self.defensive]
            self.write_log("无标的达到绝对动量门槛，空仓")
            return []

        scored.sort(key=lambda kv: kv[1], reverse=True)
        picked = [vt for vt, _ in scored[: self.max_holdings]]
        top = ", ".join(f"{vt}:{m:.3f}" for vt, m in scored[:3])
        self.write_log(f"轮动选中 {len(picked)} 只（前三 {top}）")
        return picked

    # ------------------------------------------------------------ 主循环

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """覆写基类：标的池取自 vt_symbols，而非 engine.get_universe()。

        基类实现依赖 `engine.get_universe()`，那是回测引擎独有的方法，
        拿去跑实盘会直接 AttributeError。这个项目里已经因为回测与实盘
        走两套代码吃过大亏 —— 所以这里让同一个类在两个引擎上都能跑，
        而不是再分叉一份实盘专用逻辑。
        """
        self._bar_count += 1
        self.update_indicators(bars)

        if not self.trading:
            return
        if (self._bar_count - self.rebalance_phase) % self.rebalance_days != 0:
            return

        get_uni = getattr(self.engine, "get_universe", None)
        pool = list(get_uni()) if callable(get_uni) else list(self.vt_symbols)
        candidates = [s for s in pool
                      if s in bars and not getattr(bars[s], "suspended", False)]
        if not candidates:
            return

        selected = self.select(bars, candidates)[: self.max_holdings]
        self.last_selection = selected
        self.rebalance(selected, bars)
