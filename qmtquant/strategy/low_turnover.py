"""低换手选股。

## 为什么是这个因子

在本项目测过的全部因子里，**换手率是唯一在两段样本上都稳定的**：

=========  ==========  ==========  ==========  ==========
因子        前半段 t     后半段 t     前半多空     后半多空
=========  ==========  ==========  ==========  ==========
换手率      **-2.98**   **-3.18**   -1.48%      -0.76%
低波动      0.84 ✗      3.01 ✓      **-0.92%**  **+1.19%**
BP         0.54 ✗      3.67 ✓      -1.23%      +2.74%
股息率      1.98 ✗      2.98 ✓      +0.56%      +2.02%
=========  ==========  ==========  ==========  ==========

价值因子与低波动都只在 2021-07 之后有效，前半段不显著，
而且多空收益**符号反转** —— 那是 2016-2021 核心资产抱团行情的产物。
只有换手率两段同号同量级。

（区间 2016-01 ~ 2026-08，沪深300 历史成分池，60 日预测周期，
Newey-West 修正后的 t 值。）

## 为什么它不容易被套利掉

低换手股票的共同特征是**关注度低、机构配置少**。
机构不去买它们不是因为不知道，而是：

- 流动性差，大资金进出会显著推动价格
- 缺乏卖方研究覆盖，写不进投资决策流程
- 长期无波动的股票在季度考核里"不出业绩"

这几条对个人投资者都不成立 —— 资金小、不需要向谁解释、
不受季度考核。这是少数「机构做不了而不是不知道」的边际。

## 参数只有 3 个，这是有意的

10 年日线折算下来只有约 43 个独立市场状态，
按每参数 10 个状态算，自由度预算只有 4.3 个参数。
参数超预算时得到的是记忆而非规律，平原检验通过也不能采信。

因此本策略只留三个旋钮，其余全部按原则固定：

- **等权持有** —— 不做权重优化。权重优化需要估计协方差矩阵，
  那是几百个自由度，在这个样本量下必然过拟合
- **单因子** —— 不做多因子合成。合成权重又是新的自由度
- **不设止损** —— 低换手股票本身波动小，个股止损会被正常波动扫掉；
  组合层面的风险由引擎的回撤控制统一管

## ⚠ 已知弱点

- **多空收益薄**。60 日持有的多空价差只有 0.76%~1.48%，
  年化约 3~6% 毛收益，而调仓往返成本约 0.3~0.5%。
  调仓太频会把收益全部吃掉，这是 ``rebalance_days`` 不能设小的原因。
- **流动性差是双刃剑**。选出的票本身换手低，实盘冲击成本会高于回测假设。
  ``min_turnover`` 固定为 5000 万是为了守住可交易性下限。
- **风格依赖**。低换手在成长股行情里会持续跑输，且这种行情可以持续两三年。

⚠ 本策略不构成投资建议。
"""
import logging

from ..core.objects import BarData
from .portfolio import PortfolioStrategy

logger = logging.getLogger(__name__)

#: 日成交额下限（元）。低于此值不参与 —— 实盘冲击成本会远超回测假设。
#: 固定不参与寻优：它是可交易性约束，不是收益参数
MIN_TURNOVER = 50_000_000


class LowTurnoverStrategy(PortfolioStrategy):
    """按换手率从低到高选股，等权持有"""

    # 只暴露三个旋钮，其余继承自基类且不参与寻优
    parameters = ["lookback", "max_holdings", "rebalance_days"]
    variables = ["inited", "trading", "pos", "last_selection",
                 "rebalance_count"]

    #: 换手率的平均窗口（交易日）。
    #: 单日换手率噪声极大 —— 一条利好就能翻几倍，而那不代表长期换手水平。
    #: 取均值是为了测「结构性特征」而非「当天有没有事件」
    lookback: int = 60
    #: 持仓只数。太少则个股风险主导，太多则退化成指数
    max_holdings: int = 20
    #: 调仓间隔（交易日）。**不能设小** ——
    #: 多空价差本身只有 1% 量级，往返成本 0.3~0.5%，
    #: 调仓过频会把因子收益全部吃掉
    rebalance_days: int = 60

    #: 因子列名。固定不可配 —— 换成别的列就是另一个策略了
    FACTOR_COLUMN = "turnover_rate_f"

    def __init__(self, engine, strategy_name, vt_symbols, setting=None,
                 factor_store=None):
        """
        :param factor_store: ``FactorStore`` 实例。不传则**首次选股时**
            按配置里的 store_dir 自建 —— 这样 add_strategy 的签名不用改，
            通用回测入口与参数寻优都能直接用。

            惰性而非在 __init__ 里建，是因为参数网格会构造几十上百个
            策略实例，每个都载入一次因子面板会把内存和时间都吃光。
        """
        super().__init__(engine, strategy_name, vt_symbols, setting)
        self._coerce_types()
        self._validate()

        self.factor_store = factor_store
        self.rebalance_count: int = 0
        #: 最近一次的因子值，供复盘查看
        self.last_scores: dict[str, float] = {}
        self._store_failed = False

    def _coerce_types(self) -> None:
        """参数寻优从 DataFrame 传进来的是 float64，
        用作切片下标会 TypeError，且该异常常被外层吞掉"""
        for name in ("lookback", "max_holdings", "rebalance_days"):
            setattr(self, name, int(getattr(self, name)))

    def _validate(self) -> None:
        if self.lookback < 5:
            raise ValueError(f"lookback 至少为 5，实际 {self.lookback}")
        if self.max_holdings < 1:
            raise ValueError(f"max_holdings 至少为 1，实际 {self.max_holdings}")
        if self.rebalance_days < 1:
            raise ValueError(
                f"rebalance_days 至少为 1，实际 {self.rebalance_days}")
        if self.rebalance_days < 20:
            # 不是硬错误，但值得警告：多空价差只有 1% 量级
            logger.warning(
                "rebalance_days=%d 偏小：换手率因子的多空价差只有 1%% 量级，"
                "而调仓往返成本约 0.3~0.5%%，过频调仓会把收益吃光",
                self.rebalance_days)

    # ------------------------------------------------------------ 生命周期

    def on_init(self) -> None:
        self.write_log(
            f"初始化 换手率{self.lookback}日均值 持仓{self.max_holdings}只 "
            f"每{self.rebalance_days}日调仓")

    def on_stop(self) -> None:
        self.write_log(f"策略停止。调仓 {self.rebalance_count} 次，"
                       f"当前持有 {len(self.last_selection)} 只")

    # ------------------------------------------------------------ 选股

    def select(self, bars: dict[str, BarData], candidates: list[str]) -> list[str]:
        """换手率**从低到高**排序。

        方向来自实测：换手率 IC 为负且两段样本都显著
        （前半 t=-2.98，后半 t=-3.18），即低换手股票后续表现更好。
        """
        self.rebalance_count += 1
        self.last_scores = {}

        store = self._get_store()
        if store is None:
            return []

        scores = store.rolling_mean(
            self._current_datetime(bars), self.FACTOR_COLUMN, self.lookback)
        if not scores:
            return []

        picked: list[tuple[float, str]] = []
        for vt_symbol in candidates:
            bar = bars.get(vt_symbol)
            # 可交易性下限：选出来买不进去的票没有意义
            if bar is None or bar.turnover < MIN_TURNOVER:
                continue
            value = scores.get(vt_symbol)
            if value is None or value <= 0:
                continue
            self.last_scores[vt_symbol] = value
            picked.append((value, vt_symbol))

        # 升序：换手率最低的排在最前
        picked.sort(key=lambda kv: kv[0])
        return [s for _, s in picked]

    def _get_store(self):
        """惰性获取因子面板。

        载入失败只报一次错并返回 None —— 之后每个调仓日都返回空选股。
        **不静默降级到别的规则**：那会让回测跑出一条看似正常的曲线，
        而你以为测的是低换手策略。
        """
        if self.factor_store is not None:
            return self.factor_store
        if self._store_failed:
            return None
        try:
            from ..config import get_config
            from ..datafeed.factor_store import FactorStore
            cfg = get_config()
            self.factor_store = FactorStore(cfg.data.store_dir,
                                            [self.FACTOR_COLUMN])
        except Exception as exc:  # noqa: BLE001 —— 缺数据/缺列都在此兜住
            self._store_failed = True
            logger.error("%s 无法载入因子面板（%s）—— 本次回测将全程空仓。"
                         "请先运行 scripts/download_daily_basic.py",
                         self.strategy_name, exc)
            return None
        return self.factor_store

    @staticmethod
    def _current_datetime(bars: dict[str, BarData]):
        """从行情截面取当前时间。

        不用 engine 的内部状态，保证策略在回测与实盘下行为一致。
        """
        for bar in bars.values():
            return bar.datetime
        return None
