"""回撤控制。

为什么单独做：现有风控只有「当日亏损线」，防不住**连续阴跌**。
每天亏 1%、连亏 20 天，累计亏掉 18%，但一次都不会触及 3% 的日亏线 ——
这正是实盘中最常见的亏损形态，慢刀子割肉。

回撤控制看的是**从峰值算起的累计跌幅**，与单日无关，因此能覆盖这个盲区。

分档设计的理由：一刀切「回撤 10% 就全平」在震荡市会被反复触发，
每次触发都实打实付出交易成本并踏空反弹。分档 + 迟滞恢复是折中：
浅回撤先停止开新仓（不动存量），深回撤才减仓，极深才清仓。

## ⚠ 已知取舍：可能永久停摆

降档要求回撤收复到 `阈值 × recovery_ratio` 以下。若策略长期处于回撤中、
净值再也回不到峰值附近，档位就再也降不回 NORMAL，等于**永久停止开仓**。

实测（沪深300 动量轮动 2021-01~2026-08，默认 10/15/20 参数）：

| 配置 | 收益 | 最大回撤 | 成交笔数 | 买单被拦 |
|------|------|---------|---------|---------|
| 无回撤控制 | 18.14% | -50.83% | 438 | 0 |
| 回撤控制 | 2.05% | -19.97% | **26** | 584 |

回撤确实从 -50.83% 压到 -19.97%，但成交只剩 26 笔 —— 档位只变化过 2 次，
触发后再未恢复。这不是 bug，是参数与策略波动率不匹配：
该策略年化波动 27.72%，10% 的一档线在它身上属于常态波动。

**选参数的经验法则**：一档阈值不应低于策略年化波动率的 0.5 倍，
否则正常波动就会触发。用 `--dd-close-only` 等参数按策略实测调整，
或先关掉回撤控制看清策略本身的回撤形态再定阈值。
"""
import logging
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)

#: 阈值比较容差。回撤 20% 算出来常是 0.19999999999999996，
#: 直接 >= 会漏掉边界。风控在边界上应当保守触发，宁可早一步。
_EPS = 1e-9


class DrawdownLevel(IntEnum):
    """回撤档位。数值递增表示越严重，可直接比较大小。"""
    NORMAL = 0      # 正常交易
    CLOSE_ONLY = 1  # 只平不开：不再开新仓，存量不动
    REDUCE = 2      # 强制减仓：按比例削减持仓
    FLAT = 3        # 全部平仓并停止交易

    @property
    def label(self) -> str:
        return {
            DrawdownLevel.NORMAL: "正常",
            DrawdownLevel.CLOSE_ONLY: "只平不开",
            DrawdownLevel.REDUCE: "强制减仓",
            DrawdownLevel.FLAT: "全平停止",
        }[self]


@dataclass
class DrawdownConfig:
    """回撤控制参数。

    阈值均为**从峰值算起的回撤幅度**（正数，0.10 表示回撤 10%）。
    """
    enabled: bool = True

    #: 一档：停止开新仓
    close_only_threshold: float = 0.10
    #: 二档：强制减仓
    reduce_threshold: float = 0.15
    #: 二档触发时保留的仓位比例（0.5 = 砍掉一半）
    reduce_keep_ratio: float = 0.5
    #: 三档：全部平仓
    flat_threshold: float = 0.20

    #: 恢复迟滞：回撤收复到「触发阈值 × 该系数」以下才降档。
    #: 不设迟滞的话，回撤在阈值附近抖动会导致反复触发与解除，
    #: 每次都付出真金白银的交易成本。
    recovery_ratio: float = 0.7

    #: 峰值最少需要多少个观测点才生效。
    #: 刚启动时峰值就是当前净值，任何下跌都算回撤，会误触发。
    min_observations: int = 20

    def validate(self) -> None:
        if not (0 < self.close_only_threshold < self.reduce_threshold
                < self.flat_threshold < 1):
            raise ValueError(
                "回撤阈值必须满足 0 < 只平不开 < 减仓 < 全平 < 1，"
                f"实际为 {self.close_only_threshold}/{self.reduce_threshold}/"
                f"{self.flat_threshold}")
        if not 0 < self.recovery_ratio <= 1:
            raise ValueError("recovery_ratio 必须在 (0, 1] 之间")
        if not 0 <= self.reduce_keep_ratio < 1:
            raise ValueError("reduce_keep_ratio 必须在 [0, 1) 之间")


@dataclass
class DrawdownState:
    """当前回撤状态，供风控与报告读取"""
    peak: float = 0.0
    current: float = 0.0
    drawdown: float = 0.0
    level: DrawdownLevel = DrawdownLevel.NORMAL
    observations: int = 0
    #: 档位变化历史 [(净值, 回撤, 旧档, 新档)]
    transitions: list = field(default_factory=list)


class DrawdownController:
    """峰值回撤跟踪与分档响应。

    只负责**判断当前该处于哪一档**，不直接下单 ——
    具体动作由 RiskManager / LiveEngine 执行，
    这样回测与实盘能共用同一套判定逻辑。
    """

    def __init__(self, config: DrawdownConfig | None = None) -> None:
        self.config = config or DrawdownConfig()
        self.config.validate()
        self.state = DrawdownState()

    # ------------------------------------------------------------ 核心

    def update(self, equity: float) -> DrawdownLevel:
        """推入最新总资产，返回当前应处的档位。

        :param equity: 账户总资产（现金 + 持仓市值）
        """
        s = self.state
        if equity <= 0:
            return s.level

        s.observations += 1
        s.current = equity
        s.peak = max(s.peak, equity)
        s.drawdown = 1 - equity / s.peak if s.peak > 0 else 0.0

        if not self.config.enabled:
            return DrawdownLevel.NORMAL

        # 观测点不足时峰值不可信：刚启动峰值即当前值，任何下跌都成"回撤"
        if s.observations < self.config.min_observations:
            return s.level

        new_level = self._resolve_level(s.drawdown, s.level)
        if new_level != s.level:
            old = s.level
            s.transitions.append((equity, s.drawdown, old, new_level))
            level_up = new_level > old
            logger.log(
                logging.ERROR if level_up else logging.WARNING,
                "回撤档位 %s -> %s（当前回撤 %.2f%%，峰值 %.2f，现值 %.2f）",
                old.label, new_level.label, s.drawdown * 100, s.peak, equity)
            s.level = new_level
        return s.level

    def _resolve_level(self, dd: float, current: DrawdownLevel) -> DrawdownLevel:
        """由回撤幅度决定档位，降档需满足迟滞条件"""
        c = self.config

        # 升档：立即响应，不打折。阈值取闭区间，边界即触发
        if dd >= c.flat_threshold - _EPS:
            return DrawdownLevel.FLAT
        if dd >= c.reduce_threshold - _EPS:
            target = DrawdownLevel.REDUCE
        elif dd >= c.close_only_threshold - _EPS:
            target = DrawdownLevel.CLOSE_ONLY
        else:
            target = DrawdownLevel.NORMAL

        if target >= current:
            return target

        # 降档：必须回撤收复到 阈值×recovery_ratio 以下，避免阈值附近反复横跳
        thresholds = {
            DrawdownLevel.FLAT: c.flat_threshold,
            DrawdownLevel.REDUCE: c.reduce_threshold,
            DrawdownLevel.CLOSE_ONLY: c.close_only_threshold,
        }
        if dd < thresholds[current] * c.recovery_ratio:
            return target
        return current

    # ------------------------------------------------------------ 查询

    @property
    def level(self) -> DrawdownLevel:
        return self.state.level

    @property
    def drawdown(self) -> float:
        return self.state.drawdown

    def allow_open(self) -> bool:
        """是否允许开新仓"""
        return self.state.level == DrawdownLevel.NORMAL

    def allow_any_trade(self) -> bool:
        """是否允许任何交易。FLAT 档下只允许平仓，由调用方保证方向"""
        return True

    def target_position_ratio(self) -> float:
        """目标仓位比例。REDUCE 档要求削减到该比例，FLAT 档要求清空。"""
        if self.state.level == DrawdownLevel.FLAT:
            return 0.0
        if self.state.level == DrawdownLevel.REDUCE:
            return self.config.reduce_keep_ratio
        return 1.0

    def reset(self) -> None:
        """重置状态。仅用于新一轮回测，实盘不应调用 —— 那等于抹掉回撤记忆。"""
        self.state = DrawdownState()

    def summary(self) -> str:
        s = self.state
        lines = [
            "--- 回撤控制 ---",
            f"峰值净值   : {s.peak:,.2f}",
            f"当前净值   : {s.current:,.2f}",
            f"当前回撤   : {s.drawdown:.2%}",
            f"当前档位   : {s.level.label}",
            f"档位变化   : {len(s.transitions)} 次",
        ]
        for equity, dd, old, new in s.transitions[-5:]:
            lines.append(f"  {old.label} -> {new.label} "
                         f"（回撤 {dd:.2%}，净值 {equity:,.2f}）")
        return "\n".join(lines)
