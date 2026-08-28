"""逐仓交易管理：止损、止盈、移动止损、基于风险的仓位。

## 这个模块补的是什么

之前项目里的策略只有「买入条件」和「卖出条件」，没有**每笔交易的
风险边界**。缺的是这几样：

======================  ==================================================
缺什么                   后果
======================  ==================================================
止损价                   一只票能吃掉几十次盈利。均值回归的收益分布是
                        「大量小赚 + 偶尔巨亏」，没有止损时巨亏抹平一切
止盈 / 移动止损           浮盈全部回吐。趋势策略最典型的死法是
                        「赚了 40% 不走，最后平进平出」
基于风险的仓位            每笔满仓 = 每笔赌上全部。波动大的票和波动小的票
                        用同样的钱，风险敞口差好几倍
======================  ==================================================

## 止损距离决定仓位，不是反过来

散户常见做法是「每次买 10 万」，然后止损随便设个 5%。
正确顺序相反：

1. 先定**每笔最多亏多少**（如总资金的 1%）
2. 再由入场价与止损价算出**止损距离**
3. 仓位 = 风险预算 ÷ 止损距离

这样波动大的票自动买得少，波动小的票自动买得多，
**每笔交易的亏损上限都一样**。这是「固定买卖点」真正能落地的前提 ——
止损价不只是一个离场信号，它同时决定了这笔交易该下多大。

## A 股 T+1 的影响

当日买入当日不能卖，所以**止损最快也是次日执行**。
本模块按日线 Bar 判断，触发后由策略在下一根 Bar 发单，
与实盘一致。日内瞬间插针不会触发 —— 这是保守的方向。

⚠ 止损保护的是「单笔亏损上限」，不保护「隔夜跳空」。
跳空低开跌破止损价时，实际成交价会比止损价更差。
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: 离场原因，用于归因统计
EXIT_STOP = "止损"
EXIT_TARGET = "止盈"
EXIT_TRAILING = "移动止损"
EXIT_TIME = "超时"
EXIT_SIGNAL = "信号离场"


@dataclass
class ManagedPosition:
    """一笔在管仓位的完整状态"""

    vt_symbol: str
    entry_price: float
    volume: float
    #: 初始止损价。触及即离场，是这笔交易的亏损上限
    stop_price: float
    #: 止盈价。None 表示不设，靠移动止损或信号离场
    target_price: float | None = None
    #: 建仓时的 Bar 序号，用于超时离场
    entry_bar: int = 0
    #: 建仓以来的最高价，移动止损的锚点
    highest: float = 0.0
    #: 当前生效的止损价（移动止损会抬高它，**永不下调**）
    current_stop: float = 0.0

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError(f"入场价必须为正，实际 {self.entry_price}")
        if self.stop_price >= self.entry_price:
            raise ValueError(
                f"止损价必须低于入场价（只做多），"
                f"实际 止损 {self.stop_price} >= 入场 {self.entry_price}")
        if self.target_price is not None and self.target_price <= self.entry_price:
            raise ValueError(
                f"止盈价必须高于入场价，"
                f"实际 止盈 {self.target_price} <= 入场 {self.entry_price}")
        self.highest = self.entry_price
        self.current_stop = self.stop_price

    @property
    def risk_per_share(self) -> float:
        """**初始**每股风险 = 入场价 − 初始止损价。

        R 倍数必须锚定在初始风险上，不能随移动止损变化 ——
        「赚了 2R」的含义是「赚到当初愿意亏的两倍」，
        锚点一动这个数就失去意义了。
        """
        return self.entry_price - self.stop_price

    @property
    def current_risk_per_share(self) -> float:
        """**当前**每股风险 = 入场价 − 当前止损价。

        移动止损上移后它会变小，甚至为负（止损已在成本之上，
        这笔交易已经锁定盈利）。
        """
        return self.entry_price - self.current_stop

    @property
    def max_loss(self) -> float:
        """这笔仓位当前的最大亏损（不含跳空）。

        用当前止损而非初始止损 —— 止损上移后真实风险已经下降，
        用初始值会系统性高估在管风险。

        ⚠ 加仓后 ``entry_price`` 仍是首份的入场价，
        后续份数的入场价更高，故此值对金字塔仓位是近似。
        """
        return self.current_risk_per_share * self.volume

    def r_multiple(self, price: float) -> float:
        """当前盈亏是初始风险的几倍（R 倍数）。

        交易员用 R 而非百分比衡量盈亏：赚 2R 意味着赚到的钱
        是当初愿意亏的两倍，跨标的、跨仓位大小都可比。
        """
        if self.risk_per_share <= 0:
            return 0.0
        return (price - self.entry_price) / self.risk_per_share


@dataclass
class TradeManager:
    """管理所有在管仓位的离场判断与仓位计算。

    策略只需在建仓时调用 ``open()``、每根 Bar 调用 ``check()``，
    离场逻辑不必在每个策略里重写一遍。
    """

    #: 单笔最大亏损占总资产的比例。1% 是常见起点，
    #: 意味着连亏 10 笔才损失 10%
    risk_per_trade: float = 0.01
    #: 移动止损：价格每创新高，止损上移到「最高价 × (1 − 该比例)」。
    #: 0 表示关闭。**只上移不下调**，否则等于没有止损
    trailing_ratio: float = 0.0
    #: 移动止损的启动门槛（R 倍数）。盈利达到多少 R 才开始移动止损。
    #: 0 表示建仓即启动。设 1.0 是常见做法：先让利润跑出安全垫，
    #: 否则刚建仓就被正常波动扫出去
    trailing_start_r: float = 1.0
    #: 最长持有 Bar 数，0 表示不限。资金被套死在不涨不跌的票上
    #: 是隐性成本 —— 它没亏钱，但占着本可用于其他机会的资金
    max_holding_bars: int = 0
    #: 单个标的最大仓位占总资产比例。防止低波动标的按风险算出天量仓位
    max_position_ratio: float = 0.20

    positions: dict[str, ManagedPosition] = field(default_factory=dict)
    #: 离场原因计数，用于归因
    exit_stats: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade <= 0.5:
            raise ValueError(
                f"risk_per_trade 必须在 (0, 0.5] 之间，实际 {self.risk_per_trade}")
        if not 0 <= self.trailing_ratio < 1:
            raise ValueError("trailing_ratio 必须在 [0, 1) 之间")
        if self.trailing_start_r < 0:
            raise ValueError("trailing_start_r 不能为负")
        if not 0 < self.max_position_ratio <= 1:
            raise ValueError("max_position_ratio 必须在 (0, 1] 之间")

    # ------------------------------------------------------------ 仓位

    def position_size(self, total_value: float, entry_price: float,
                      stop_price: float) -> float:
        """由止损距离反推仓位。

        **止损距离决定仓位，不是反过来。** 每笔交易的亏损上限
        都等于 ``total_value × risk_per_trade``，与标的波动无关。

        :return: 股数（未取整，整手约束由引擎处理）
        """
        if entry_price <= 0 or stop_price >= entry_price:
            return 0.0
        risk_budget = total_value * self.risk_per_trade
        volume = risk_budget / (entry_price - stop_price)

        # 低波动标的的止损距离极小，按风险算出的仓位会大到离谱，
        # 甚至超过总资产。必须有个绝对上限兜底
        cap = total_value * self.max_position_ratio / entry_price
        return min(volume, cap)

    # ------------------------------------------------------------ 建仓

    def open(self, vt_symbol: str, entry_price: float, volume: float,
             stop_price: float, target_price: float | None = None,
             bar_index: int = 0) -> ManagedPosition:
        """登记一笔新仓位。

        :raises ValueError: 止损价不低于入场价等非法组合
        """
        pos = ManagedPosition(
            vt_symbol=vt_symbol, entry_price=entry_price, volume=volume,
            stop_price=stop_price, target_price=target_price,
            entry_bar=bar_index)
        self.positions[vt_symbol] = pos
        logger.debug("%s 建仓 @%.3f 止损 %.3f 止盈 %s 数量 %.0f",
                     vt_symbol, entry_price, stop_price,
                     f"{target_price:.3f}" if target_price else "无", volume)
        return pos

    def close(self, vt_symbol: str, reason: str = EXIT_SIGNAL) -> None:
        """移除仓位并记录离场原因"""
        if self.positions.pop(vt_symbol, None) is not None:
            self.exit_stats[reason] = self.exit_stats.get(reason, 0) + 1

    def has(self, vt_symbol: str) -> bool:
        return vt_symbol in self.positions

    def get(self, vt_symbol: str) -> ManagedPosition | None:
        return self.positions.get(vt_symbol)

    # ------------------------------------------------------------ 每根 Bar

    def check(self, vt_symbol: str, high: float, low: float, close: float,
              bar_index: int = 0) -> str | None:
        """判断该仓位本根 Bar 是否触发离场。

        判断顺序是**先止损后止盈**：同一根 Bar 里最高价触及止盈、
        最低价也触及止损时，无法从日线知道谁先发生。
        取更坏的那个是唯一诚实的做法 —— 假设止盈先到会系统性高估收益。

        :return: 离场原因；不触发则 None
        """
        pos = self.positions.get(vt_symbol)
        if pos is None:
            return None

        if low <= pos.current_stop:
            return EXIT_TRAILING if pos.current_stop > pos.stop_price else EXIT_STOP

        if pos.target_price is not None and high >= pos.target_price:
            return EXIT_TARGET

        if (self.max_holding_bars > 0
                and bar_index - pos.entry_bar >= self.max_holding_bars):
            return EXIT_TIME

        # 未触发离场才更新移动止损。顺序反了会用「本根抬高后的止损」
        # 去判断「本根是否止损」，等于用了未来信息
        self._update_trailing(pos, high)
        return None

    def _update_trailing(self, pos: ManagedPosition, high: float) -> None:
        """价格创新高时上移止损。**只上移不下调**"""
        if high > pos.highest:
            pos.highest = high
        if self.trailing_ratio <= 0:
            return
        # 盈利未达门槛不启动 —— 刚建仓就移动止损会被正常波动扫出去
        if pos.r_multiple(pos.highest) < self.trailing_start_r:
            return
        candidate = pos.highest * (1 - self.trailing_ratio)
        if candidate > pos.current_stop:
            pos.current_stop = candidate

    # ------------------------------------------------------------ 统计

    def summary(self) -> str:
        lines = ["--- 离场原因归因 ---"]
        total = sum(self.exit_stats.values())
        if not total:
            lines.append("  无离场记录")
            return "\n".join(lines)
        for reason, n in sorted(self.exit_stats.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {reason:<10} {n:>5} 次  ({n / total:.1%})")
        lines.append(f"  合计       {total:>5} 次")
        return "\n".join(lines)
