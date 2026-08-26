"""标的池的几种实现。

按「偏差从大到小」排列：
1. StaticUniverse        —— 当前指数成分快照，偏差最大，只适合快速试验
2. PointInTimeUniverse   —— 叠加上市日过滤，消除「交易尚未上市的股票」
3. HistoricalUniverse    —— 读历史成分股 CSV，可完全消除偏差（需外部数据）
"""
import logging
from pathlib import Path

import pandas as pd

from ..utils.symbol import normalize
from .base import BiasReport, UniverseProvider, _to_date

logger = logging.getLogger(__name__)


class StaticUniverse(UniverseProvider):
    """固定标的池：整个回测区间用同一份名单。

    ⚠ 若名单来自「当前」的指数成分，会同时引入两种偏差：
    - 幸存者偏差：这些年被调出指数、退市的股票不在名单里，而它们通常表现更差
    - 成分股前视：2020 年就用上了 2026 年才知道的成分名单
    """

    def __init__(self, symbols: list[str], source: str = "静态名单") -> None:
        self.symbols = [normalize(s) for s in symbols]
        self.source = source

    def get_universe(self, dt) -> list[str]:
        return list(self.symbols)

    def all_symbols(self) -> list[str]:
        return list(self.symbols)

    def describe_bias(self) -> BiasReport:
        return BiasReport(
            survivorship=True,
            membership_lookahead=True,
            listing_filtered=False,
            size=len(self.symbols),
            notes=[
                f"名单来源：{self.source}（当前快照，非历史成分）",
                "已退市/已调出指数的标的完全缺失",
                "回测早期就使用了当期才确定的成分名单",
            ],
        )


class PointInTimeUniverse(UniverseProvider):
    """叠加上市日/退市日过滤的标的池。

    修掉的是**可修的那部分**：不在标的上市之前就交易它。
    修不掉的是幸存者偏差 —— 那需要一份包含已退市标的的历史名单，
    QMT 数据源提供不了（退市股在其中完全不存在）。
    """

    def __init__(self, base: UniverseProvider, listing_dates: dict[str, pd.Timestamp],
                 delist_dates: dict[str, pd.Timestamp] | None = None,
                 min_days_since_ipo: int = 60) -> None:
        """
        :param listing_dates: vt_symbol -> 上市日
        :param delist_dates:  vt_symbol -> 退市日（可选）
        :param min_days_since_ipo: 上市后多少个自然日内不纳入。
            次新股波动极端、无历史数据可用于计算指标，纳入会污染回测。
        """
        self.base = base
        self.listing_dates = {normalize(k): pd.Timestamp(v)
                              for k, v in listing_dates.items()}
        self.delist_dates = {normalize(k): pd.Timestamp(v)
                             for k, v in (delist_dates or {}).items()}
        self.min_days_since_ipo = min_days_since_ipo

    def get_universe(self, dt) -> list[str]:
        d = pd.Timestamp(_to_date(dt))
        result = []
        for symbol in self.base.get_universe(dt):
            listed = self.listing_dates.get(symbol)
            # 拿不到上市日就保守剔除，宁可少选也不要引入未来数据
            if listed is None:
                continue
            if (d - listed).days < self.min_days_since_ipo:
                continue
            delisted = self.delist_dates.get(symbol)
            if delisted is not None and d >= delisted:
                continue
            result.append(symbol)
        return result

    def all_symbols(self) -> list[str]:
        return self.base.all_symbols()

    def describe_bias(self) -> BiasReport:
        report = self.base.describe_bias()
        report.listing_filtered = True
        report.notes.append(
            f"已按上市日过滤，上市不足 {self.min_days_since_ipo} 个自然日的标的不纳入"
        )
        if self.delist_dates:
            report.notes.append(f"已知退市日的标的 {len(self.delist_dates)} 只，到期后剔除")
        else:
            report.notes.append("无退市日数据，幸存者偏差未消除")
        return report


class HistoricalUniverse(UniverseProvider):
    """历史成分股标的池：从 CSV 读取每个日期的真实成分名单。

    CSV 格式（两列，可含表头）::

        date,symbol
        2020-01-02,000001.SZ
        2020-01-02,600000.SH
        ...

    数据来源建议（QMT 提供不了）：
    - akshare: `index_stock_cons_csindex` / `stock_zh_a_st_em`
    - Tushare Pro: `index_weight` 接口（需积分）
    - 中证指数官网历史成分文件

    只要 CSV 里包含了当时在册、后来退市的标的，幸存者偏差即可完全消除。
    """

    def __init__(self, csv_path: str | Path, source: str = "历史成分 CSV") -> None:
        self.csv_path = Path(csv_path)
        self.source = source

        df = pd.read_csv(self.csv_path)
        cols = {c.lower(): c for c in df.columns}
        if "date" not in cols or "symbol" not in cols:
            raise ValueError(f"CSV 必须包含 date 与 symbol 两列，实际: {list(df.columns)}")

        df["_date"] = pd.to_datetime(df[cols["date"]])
        df["_symbol"] = df[cols["symbol"]].astype(str).map(normalize)

        # 按日期分组，get_universe 用 asof 语义取「不晚于该日的最近一期名单」
        self._by_date: dict[pd.Timestamp, list[str]] = {
            d: sorted(set(g["_symbol"])) for d, g in df.groupby("_date")
        }
        self._dates = sorted(self._by_date)
        self._all = sorted(set(df["_symbol"]))
        logger.info("历史成分加载完成：%d 期名单，%d 个标的，%s ~ %s",
                    len(self._dates), len(self._all),
                    self._dates[0].date(), self._dates[-1].date())

    def get_universe(self, dt) -> list[str]:
        d = pd.Timestamp(_to_date(dt))
        # 取不晚于 d 的最近一期名单，避免用到未来的调整结果
        idx = None
        for i, cur in enumerate(self._dates):
            if cur > d:
                break
            idx = i
        return list(self._by_date[self._dates[idx]]) if idx is not None else []

    def all_symbols(self) -> list[str]:
        return list(self._all)

    def describe_bias(self) -> BiasReport:
        return BiasReport(
            survivorship=False,
            membership_lookahead=False,
            listing_filtered=True,
            size=len(self._all),
            notes=[
                f"名单来源：{self.source}",
                f"共 {len(self._dates)} 期历史名单，按调仓日 asof 取用",
                "前提：CSV 必须包含当时在册、后来退市的标的，否则幸存者偏差仍在",
            ],
        )
