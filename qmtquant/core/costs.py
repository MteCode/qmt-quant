"""交易成本 —— 全系统唯一事实源。

## 为什么要有这个模块

在此之前，成本常数散在 8 个文件里各写各的：

    scripts/optimize_t0_600711.py      COMMISSION = 0.00025    # 万 2.5
    scripts/backtest_intraday_gbm.py   COMMISSION = 0.00025    # 万 2.5
    strategies/.../run_backtest.py     commission_rate = 0.0003  # 万 3
    qmtquant/config.py                 commission_rate = 0.00025

结果是同一个账户在不同回测里被收不同的佣金，而且**没有任何地方会报错**。
一次审计发现的问题：
  - 佣金率三个版本，全部与券商实际的万 0.854 不符（偏高 2.9~3.5 倍）
  - 最低佣金 5 元：引擎实现了，**所有研究脚本零处实现**
  - 印花税全部写死 0.001，是 2023-08-28 减半**之前**的值
  - 过户费万 0.1 双边：引擎收，研究脚本全部漏掉
  - 模拟撮合网关的滑点从未施加（成交价直接等于市价）

任何一处偏差都会让回测结论失真，而做 T 这类策略对成本极度敏感 ——
毛 edge 中位数约 0.00%，成本差 0.05% 就足以颠倒结论的方向。

## 三条容易搞错的规则

**最低佣金是分段函数，不是费率。** 单笔金额低于
`commission_min / commission_rate`（万0.854 下为 58,548 元）时按 5 元收，
实际费率随金额下降而爆炸：1 万元的单子实际费率是万 5，名义值的 5.9 倍。
做 T 必然把资金拆成小单，几乎全部落在这一段，**用固定费率回测会系统性
低估成本**。

**印花税只在卖出侧收。** 现行 0.05%（2023-08-28 由 0.1% 减半）。
法定，不可协商，任何"降成本"方案都不能假设它降。

**滑点不是费用，是执行假设。** 它不该和费率混在一起谈"往返成本"，
但在回测里必须施加，否则成交价等于市价，回测优于任何真实执行。
A 股最小变动价位固定 0.01 元，所以滑点的相对大小依赖股价：
10 元的票，一个 tick 就是 0.1%。固定比例滑点对低价股严重乐观。
"""
from __future__ import annotations

from dataclasses import dataclass

from qmtquant.config import CostConfig

__all__ = [
    "CostModel",
    "DEFAULT_COST",
    "fee",
    "round_trip_rate",
    "min_commission_threshold",
    "effective_commission_rate",
]


@dataclass(frozen=True)
class CostModel:
    """一笔交易的完整成本，全部以成交金额为基数。

    与 CostConfig 的区别：CostConfig 是**配置**（可由 yaml 覆盖），
    CostModel 是**计算**。分开是为了让研究脚本能拿到计算能力而不必
    依赖整个配置加载链路。
    """

    commission_rate: float = 0.0000854   # 佣金万 0.854
    commission_min: float = 5.0          # 单笔最低佣金（元）
    stamp_tax_rate: float = 0.0005       # 印花税万 5，仅卖出，2023-08-28 起
    transfer_fee_rate: float = 0.00001   # 过户费万 0.1，双边
    slippage_rate: float = 0.0005        # 滑点万 5，单边（执行假设，非费用）

    @classmethod
    def from_config(cls, cfg: CostConfig, slippage_rate: float = 0.0005
                    ) -> "CostModel":
        """由 CostConfig 构造。

        slippage_tick 是「几个最小变动价位」，换算成比例需要股价，
        这里不做换算 —— 调用方若要按股价换算，用 slippage_from_tick()。
        """
        return cls(
            commission_rate=cfg.commission_rate,
            commission_min=cfg.commission_min,
            stamp_tax_rate=cfg.stamp_tax_rate,
            transfer_fee_rate=cfg.transfer_fee_rate,
            slippage_rate=slippage_rate,
        )

    # ------------------------------------------------------------ 费用

    def commission(self, amount: float) -> float:
        """佣金，含最低 5 元下限。"""
        return max(amount * self.commission_rate, self.commission_min)

    def fee(self, amount: float, is_sell: bool) -> float:
        """一笔成交的法定与券商费用合计，**不含滑点**。

        滑点单列是因为它性质不同：费用是确定支出，滑点是执行质量假设。
        把两者混在一个数里，会让「谈佣金能省多少」这种问题算不清楚。
        """
        if amount <= 0:
            return 0.0
        c = self.commission(amount)
        t = amount * self.stamp_tax_rate if is_sell else 0.0
        f = amount * self.transfer_fee_rate
        return c + t + f

    def cost(self, amount: float, is_sell: bool) -> float:
        """费用 + 滑点，回测里买卖两侧的实际总成本。"""
        return self.fee(amount, is_sell) + amount * self.slippage_rate

    # ------------------------------------------------------------ 派生量

    def round_trip(self, amount: float) -> float:
        """一次做 T 往返（买入 amount + 卖出 amount）的总成本，绝对金额。"""
        return self.cost(amount, False) + self.cost(amount, True)

    def round_trip_rate(self, amount: float) -> float:
        """往返成本率。随金额变化 —— 最低佣金让小单的费率显著更高。"""
        if amount <= 0:
            return 0.0
        return self.round_trip(amount) / amount

    def round_trip_rate_nominal(self) -> float:
        """不考虑最低佣金的名义往返费率，即金额足够大时的极限值。"""
        return (2 * self.commission_rate + 2 * self.transfer_fee_rate
                + self.stamp_tax_rate + 2 * self.slippage_rate)

    def min_commission_threshold(self) -> float:
        """最低佣金的临界成交金额。低于它，实际费率高于名义费率。"""
        if self.commission_rate <= 0:
            return float("inf")
        return self.commission_min / self.commission_rate

    def effective_commission_rate(self, amount: float) -> float:
        """给定单笔金额下的实际佣金率。"""
        if amount <= 0:
            return 0.0
        return self.commission(amount) / amount

    def slippage_from_tick(self, price: float, ticks: float = 1.0) -> float:
        """按最小变动价位换算滑点比例。

        A 股最小变动价位固定 0.01 元，所以同样一个 tick，
        10 元的票是 0.1%，50 元的票只有 0.02%。
        固定比例滑点对低价股乐观，对高价股保守。
        """
        if price <= 0:
            return self.slippage_rate
        return ticks * 0.01 / price

    def describe(self) -> dict:
        """写进 summary.json 的成本模型说明。

        只记名义费率会掩盖最低佣金的分段效应，让别人（和以后的自己）
        以为回测用的是那个数。所以连同分档费率一起记。
        """
        thr = self.min_commission_threshold()
        return {
            "commission_rate": self.commission_rate,
            "commission_min": self.commission_min,
            "stamp_tax_rate": self.stamp_tax_rate,
            "transfer_fee_rate": self.transfer_fee_rate,
            "slippage_rate": self.slippage_rate,
            "round_trip_nominal": self.round_trip_rate_nominal(),
            "min_commission_threshold": round(thr, 2),
            "round_trip_by_amount": {
                str(a): round(self.round_trip_rate(a), 6)
                for a in (5000, 10000, 25000, 50000, 100000, 200000)
            },
        }

    def banner(self) -> str:
        """回测脚本启动时打印，让报告自己说明用的是什么成本。"""
        thr = self.min_commission_threshold()
        by = "  ".join(f"{a // 1000}k={self.round_trip_rate(a):.4%}"
                       for a in (10000, 25000, 50000))
        return (
            f"成本：佣金 {self.commission_rate:.6%}"
            f"（最低 {self.commission_min:.0f} 元/笔）"
            f" + 印花税 {self.stamp_tax_rate:.2%}（仅卖出）"
            f" + 过户费 {self.transfer_fee_rate:.3%}（双边）"
            f" + 滑点 {self.slippage_rate:.2%}（单边）\n"
            f"名义往返 {self.round_trip_rate_nominal():.4%}；"
            f"单笔 {thr:,.0f} 元以下最低佣金生效：{by}"
        )


#: 全系统默认成本模型。研究脚本直接 import 这个，不要各自定义常数。
DEFAULT_COST = CostModel()


# --------------------------------------------------------- 模块级便捷函数

def fee(amount: float, is_sell: bool, model: CostModel | None = None) -> float:
    return (model or DEFAULT_COST).fee(amount, is_sell)


def round_trip_rate(amount: float, model: CostModel | None = None) -> float:
    return (model or DEFAULT_COST).round_trip_rate(amount)


def min_commission_threshold(model: CostModel | None = None) -> float:
    return (model or DEFAULT_COST).min_commission_threshold()


def effective_commission_rate(amount: float,
                              model: CostModel | None = None) -> float:
    return (model or DEFAULT_COST).effective_commission_rate(amount)
