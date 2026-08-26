"""标的池（Universe）抽象。

为什么需要这一层：选股回测的结论对「每个调仓日到底能选哪些股票」极度敏感。
把它抽象出来，才能让偏差**显式可见**，而不是藏在一句
`vt_symbols = get_sector_stocks("沪深300")` 里。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class BiasReport:
    """标的池的偏差说明。

    回测报告必须带上它 —— 一个有幸存者偏差的漂亮曲线，
    比没有回测更危险，因为它会让人误以为验证过了。
    """
    #: 幸存者偏差：已退市/已被剔除的标的是否缺失
    survivorship: bool = True
    #: 成分股前视：是否用了「未来才知道」的指数成分名单
    membership_lookahead: bool = True
    #: 是否按上市日过滤（防止在标的上市前就交易它）
    listing_filtered: bool = False
    #: 标的池规模
    size: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.survivorship and not self.membership_lookahead

    def summary(self) -> str:
        lines = ["--- 标的池偏差说明 ---", f"标的数量        : {self.size}"]
        lines.append(f"幸存者偏差      : {'存在 ⚠' if self.survivorship else '已消除'}")
        lines.append(f"成分股前视偏差  : {'存在 ⚠' if self.membership_lookahead else '已消除'}")
        lines.append(f"上市日过滤      : {'已启用' if self.listing_filtered else '未启用 ⚠'}")
        for n in self.notes:
            lines.append(f"  · {n}")
        if not self.is_clean:
            lines.append("  ⚠ 回测收益被系统性高估，不可直接外推到实盘")
        return "\n".join(lines)


def _to_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    raise TypeError(f"无法识别的日期类型: {type(d)}")


class UniverseProvider(ABC):
    """按日期返回可交易标的列表"""

    @abstractmethod
    def get_universe(self, dt) -> list[str]:
        """返回该日可交易的 vt_symbol 列表"""

    @abstractmethod
    def describe_bias(self) -> BiasReport:
        """说明这个标的池带有哪些偏差"""

    def all_symbols(self) -> list[str]:
        """全区间可能出现过的所有标的，用于一次性装载数据"""
        raise NotImplementedError
