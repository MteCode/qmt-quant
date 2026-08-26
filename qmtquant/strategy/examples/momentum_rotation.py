"""动量轮动选股策略示例。

逻辑：每 N 个交易日调仓，选过去 M 日涨幅最高的前 K 只等权持有。

⚠ 这是最经典的教科书策略，用来演示选股回测链路，**不是能赚钱的策略**。
A 股上动量效应远弱于美股，短周期甚至常年反转。真要用请自行做样本外验证。
"""
from collections import defaultdict, deque

from ..portfolio import PortfolioStrategy


class MomentumRotationStrategy(PortfolioStrategy):
    """横截面动量轮动"""

    parameters = PortfolioStrategy.parameters + ["lookback", "skip_recent"]
    variables = ["inited", "trading", "pos", "last_selection"]

    #: 动量计算窗口（交易日）
    lookback: int = 120
    #: 跳过最近 N 日 —— 短期反转效应会污染动量信号，这是学界的标准做法
    skip_recent: int = 20

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        maxlen = self.lookback + self.skip_recent + 5
        self.closes: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def update_indicators(self, bars) -> None:
        for vt_symbol, bar in bars.items():
            # 停牌日价格无意义，跳过以免污染动量窗口
            if not bar.suspended and bar.close_price > 0:
                self.closes[vt_symbol].append(bar.close_price)

    def select(self, bars, candidates: list[str]) -> list[str]:
        need = self.lookback + self.skip_recent
        scored = []

        for vt_symbol in candidates:
            closes = self.closes.get(vt_symbol)
            if closes is None or len(closes) < need:
                continue
            prices = list(closes)
            # 动量 = 从 (lookback+skip) 日前 到 skip 日前 的累计涨幅
            start = prices[-need]
            end = prices[-self.skip_recent - 1]
            if start <= 0:
                continue
            scored.append((end / start - 1, vt_symbol))

        scored.sort(reverse=True)
        return [s for _, s in scored]
