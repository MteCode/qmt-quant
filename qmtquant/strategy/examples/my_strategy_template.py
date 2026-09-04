"""自定义策略模板 —— 复制这个文件开始写自己的策略。

## 三步走

1. 复制本文件到 `qmtquant/strategy/my_xxx.py`，改类名
2. 在 `on_bar` 里写买卖逻辑
3. 用 `scripts/run_backtest.py` 或下面的示例代码回测

## 框架替你做了什么

写策略时**不用管**这些，引擎会处理：

- T+1（当日买入不可卖）
- 涨跌停不成交（按板块判定，主板 10%、创业板/科创板 20%、北交所 30%）
- 整手取整（按真实价，后复权价会被换算回去）
- 停牌不成交（数据里表现为 Bar 缺失）
- 手续费、印花税、过户费、滑点
- 回撤三档控制（6% 停开仓 / 9% 减仓 / 12% 清仓）
- 委托按**次日开盘**成交 —— 这是防前视的关键，你在收盘信号里下单，
  成交发生在下一根 Bar，拿不到当天的收盘价

## 你需要注意的

**别用未来数据。** `on_bar(bar)` 收到的是当日已收盘的 Bar，
用它下单会在次日开盘成交，这是安全的。但如果你自己去读一个
带未来信息的因子面板（比如没做滞后的财报数据），框架拦不住。

**参数别调太多。** 两个整数参数在网格上一扫总能找到「好看」的组合，
那是拟合噪声。写完先跑参数邻域检验：

    python strategies/alstm_ppo_csi1000/sweep_portfolio.py

看是连片区域还是孤立尖刺。
"""
import logging

from ...core.objects import BarData
from ..base import StrategyBase

logger = logging.getLogger(__name__)


class MyStrategy(StrategyBase):
    """示例：20 日新高买入，跌破 10 日均线卖出。

    这是个**动量突破**策略，逻辑简单到足以看清框架怎么用。
    它本身没有经过检验，别直接拿去实盘。
    """

    #: 声明可从配置注入的参数。只有这里列出的才会被 setting 覆盖
    parameters = ["breakout_window", "exit_window", "position_ratio"]

    #: 声明需要持久化的运行时变量（实盘重启后可恢复）
    variables = ["inited", "trading", "pos", "trade_count"]

    # ---------------------------------------------------------- 参数
    #: 突破窗口 —— 收盘价创 N 日新高即买入
    breakout_window: int = 20
    #: 离场窗口 —— 跌破 N 日均线即清仓
    exit_window: int = 10
    #: 单只标的占用可用资金的比例
    position_ratio: float = 0.2

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: 每个标的各自维护价格序列 —— 多标的策略必须按标的分开存，
        #: 混在一起会让 A 股的价格影响 B 股的信号
        self._closes: dict[str, list[float]] = {}
        self.trade_count = 0

    # ---------------------------------------------------------- 生命周期

    def on_init(self) -> None:
        """引擎启动时调用一次。可在这里加载外部数据。"""
        self.write_log(
            f"初始化：突破 {self.breakout_window} 日 / "
            f"离场 {self.exit_window} 日 / 仓位 {self.position_ratio:.0%}")

    def on_start(self) -> None:
        """开始交易前调用。此后 self.trading 为 True。"""
        self.write_log("策略启动")

    def on_stop(self) -> None:
        self.write_log(f"策略停止，共成交 {self.trade_count} 次")

    # ---------------------------------------------------------- 核心逻辑

    def on_bar(self, bar: BarData) -> None:
        """每根 K 线调用一次（每个标的各自触发）。

        注意 bar 是**已收盘**的，在这里下的单会在**次日开盘**成交。
        """
        vt = bar.vt_symbol
        closes = self._closes.setdefault(vt, [])
        closes.append(bar.close_price)

        # 只保留需要的长度，避免长回测里内存无限增长
        keep = max(self.breakout_window, self.exit_window) + 5
        if len(closes) > keep:
            del closes[:-keep]

        # 数据不够算指标时什么都不做 —— 别用不完整的窗口凑合
        if len(closes) < self.breakout_window:
            return

        pos = self.get_pos(vt)

        if pos > 0:
            # --- 持仓中：跌破均线离场
            if len(closes) >= self.exit_window:
                ma = sum(closes[-self.exit_window:]) / self.exit_window
                if bar.close_price < ma:
                    # 卖价挂低一点确保成交（次日开盘撮合）
                    self.sell(vt, bar.close_price * 0.97, pos)
                    self.trade_count += 1
                    self.write_log(
                        f"{vt} 跌破 {self.exit_window} 日均线 "
                        f"{ma:.2f}，清仓 {pos:.0f} 股")
            return

        # --- 空仓：创新高则买入
        # 用 [:-1] 排除当日，否则「今日收盘 >= 含今日的最高价」恒成立
        window_high = max(closes[-self.breakout_window:-1])
        if bar.close_price <= window_high:
            return

        cash = self.get_cash() * self.position_ratio
        if cash <= 0:
            return
        # 买价挂高一点确保成交。整手取整由引擎负责，这里不用管
        price = bar.close_price * 1.03
        volume = cash / price
        if self.buy(vt, price, volume):
            self.trade_count += 1
            self.write_log(
                f"{vt} 突破 {self.breakout_window} 日新高 "
                f"{window_high:.2f}，买入约 {volume:.0f} 股")

    def on_bars(self, bars: dict) -> None:
        """同一时间截面的所有标的一起到达。

        需要**横截面比较**时用这个（比如「今天涨幅前 10 名买入」），
        单标的独立判断用 on_bar 即可。两者会都被调用，别重复下单。
        """


# ==================================================================
# 怎么回测
# ==================================================================
#
#   from pathlib import Path
#   from qmtquant.config import get_config
#   from qmtquant.core.constants import Interval
#   from qmtquant.datafeed.xt_feed import XtDataFeed
#   from qmtquant.engine.backtest_engine import BacktestEngine
#   from qmtquant.strategy.examples.my_strategy_template import MyStrategy
#
#   cfg = get_config()
#   feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
#   symbols = ["600519.SSE", "000001.SZSE"]
#   bars = feed.load_bars(symbols, "2022-01-01", "2026-08-27", Interval.DAILY)
#
#   engine = BacktestEngine(initial_capital=500_000, cost=cfg.cost)
#   engine.load_data(bars)
#   engine.add_strategy(MyStrategy, symbols,
#                       {"breakout_window": 20, "exit_window": 10})
#   stats = engine.run()
#   print(stats.summary())
#
# 写完先做参数邻域检验，别只看最优点：
#
#   python strategies/alstm_ppo_csi1000/sweep_portfolio.py
#
# 判读标准见 USAGE.md 的「判断策略有没有 alpha」一节。
