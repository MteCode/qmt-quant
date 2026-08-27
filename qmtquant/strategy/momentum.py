"""横截面动量轮动。

## 这是什么策略

每 N 个交易日调仓，买过去 M 日涨得最好的前 K 只，等权持有。
「强者恒强」—— 学界称横截面动量（cross-sectional momentum），
是 Jegadeesh & Titman (1993) 以来被研究最多的因子。

## ⚠ 在沪深300 上它是失效的，且方向为负

本模块保留并工程化，不代表它可用。实测证据：

| 检验 | 结果 |
|------|------|
| IC（过去60日 vs 未来60日） | **-0.0575**，t = -4.6 |
| 无偏回测 2016-2026 | **-24.56%**，同期指数 +33.12% |
| 最大回撤 | -57.41%，比指数的 -45.60% 还深 |

IC 显著为负意味着**信号方向是反的** —— 涨得最好的那批股票，
接下来跑输。这不是参数没调好，是机制在这个池子上不成立。

为什么：沪深300 是 A 股定价最充分的一段，动量早被套利掉；
而散户主导的市场里短期反转效应远强于动量。

## 那它还留着干什么

1. **对照基准**。新策略至少要打得过一个已知失效的策略。
2. **换池子后需要重测**。中证500/1000 的成分股定价效率低得多，
   动量在小盘股上是否仍为负，是个未验证的问题 ——
   历史成分数据已经拉好，随时可测。
3. `skip_recent` 参数本身就是为规避反转效应设计的，
   它的取值对结果的影响是可研究的对象。

⚠ 本策略不构成投资建议。
"""
import logging
from collections import defaultdict, deque

from ..core.objects import BarData
from .portfolio import PortfolioStrategy

logger = logging.getLogger(__name__)


class MomentumRotationStrategy(PortfolioStrategy):
    """横截面动量轮动"""

    parameters = PortfolioStrategy.parameters + [
        "lookback", "skip_recent", "min_turnover", "reverse",
    ]
    variables = ["inited", "trading", "pos", "last_selection", "rebalance_count"]

    #: 动量计算窗口（交易日）
    lookback: int = 120
    #: 跳过最近 N 日 —— 短期反转效应会污染动量信号，这是学界的标准做法。
    #: 设为 0 则用「截止今日」的涨幅，反转污染最严重
    skip_recent: int = 20
    #: 日成交额下限（元）。流动性差的标的滑点会吃掉动量微薄的预期收益，
    #: 且容易出现「涨停打板打不进去」的虚假回测收益
    min_turnover: float = 50_000_000
    #: **反向模式**：选跌得最惨的而非涨得最好的。
    #: 因为实测 IC 显著为负（-0.0575, t=-4.6），反着做在逻辑上值得一试。
    #: 注意这在经济含义上已经不是动量而是「长周期反转」
    reverse: bool = False

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._coerce_types()
        self._validate()

        maxlen = self.lookback + self.skip_recent + 5
        self.closes: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        #: 调仓次数，换手成本的直接来源
        self.rebalance_count: int = 0
        #: 最近一次打分结果，供复盘查看：vt_symbol -> 动量值
        self.last_scores: dict[str, float] = {}

    def _coerce_types(self) -> None:
        """整数参数强制转型。参数寻优会从 DataFrame 传入 float，
        用作 deque(maxlen=) 或切片下标时会 TypeError ——
        而这个异常常被外层 try/except 吞掉，表现为整个网格静默跑空"""
        for name in ("lookback", "skip_recent", "rebalance_days", "max_holdings"):
            setattr(self, name, int(getattr(self, name)))
        self.min_turnover = float(self.min_turnover)
        self.reverse = bool(self.reverse)

    def _validate(self) -> None:
        if self.lookback < 2:
            raise ValueError(f"lookback 至少为 2，实际 {self.lookback}")
        if self.skip_recent < 0:
            raise ValueError(f"skip_recent 不能为负，实际 {self.skip_recent}")
        if self.rebalance_days < 1:
            raise ValueError(f"rebalance_days 至少为 1，实际 {self.rebalance_days}")
        if self.max_holdings < 1:
            raise ValueError(f"max_holdings 至少为 1，实际 {self.max_holdings}")
        if self.min_turnover < 0:
            raise ValueError("min_turnover 不能为负")

    # ------------------------------------------------------------ 生命周期

    def on_init(self) -> None:
        self.write_log(
            f"初始化 动量窗口={self.lookback} 跳过近{self.skip_recent}日 "
            f"持仓{self.max_holdings}只 每{self.rebalance_days}日调仓"
            + ("（反向模式）" if self.reverse else ""))
        # 预热：不预热的话实盘要等 lookback+skip 根 Bar 才有信号
        self.load_bars(self.lookback + self.skip_recent + 20)

    def on_stop(self) -> None:
        self.write_log(f"策略停止。调仓 {self.rebalance_count} 次，"
                       f"当前持有 {len(self.last_selection)} 只")

    # ------------------------------------------------------------ 指标

    def update_indicators(self, bars: dict[str, BarData]) -> None:
        for vt_symbol, bar in bars.items():
            # 停牌日价格无意义，跳过以免污染动量窗口
            if not bar.suspended and bar.close_price > 0:
                self.closes[vt_symbol].append(bar.close_price)

    def compute_momentum(self, vt_symbol: str) -> float | None:
        """动量 = 从 (lookback+skip) 日前 到 skip 日前 的累计涨幅。

        为什么要跳过最近 skip_recent 日：短期（1 个月内）存在显著的
        反转效应，把它算进动量会让信号自相矛盾 —— 这是 Fama-French
        构造 UMD 因子时的标准处理。
        """
        closes = self.closes.get(vt_symbol)
        need = self.lookback + self.skip_recent
        if closes is None or len(closes) < need:
            return None

        prices = list(closes)
        start = prices[-need]
        # skip_recent=0 时终点就是最新价，负号下标要特殊处理
        end = prices[-1] if self.skip_recent == 0 else prices[-self.skip_recent - 1]
        if start <= 0:
            return None
        return end / start - 1

    # ------------------------------------------------------------ 选股

    def select(self, bars: dict[str, BarData], candidates: list[str]) -> list[str]:
        scored: list[tuple[float, str]] = []
        self.last_scores = {}

        for vt_symbol in candidates:
            bar = bars.get(vt_symbol)
            # 流动性过滤放在打分前，省去无意义的计算
            if bar is None or bar.turnover < self.min_turnover:
                continue
            score = self.compute_momentum(vt_symbol)
            if score is None:
                continue
            self.last_scores[vt_symbol] = score
            scored.append((score, vt_symbol))

        # 反向模式取最低分。用 key 排序而非 reverse=True 取反，
        # 避免同分时因 vt_symbol 参与比较导致的顺序抖动
        scored.sort(key=lambda kv: kv[0], reverse=not self.reverse)
        self.rebalance_count += 1
        return [s for _, s in scored]
