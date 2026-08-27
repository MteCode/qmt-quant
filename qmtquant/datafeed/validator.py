"""数据校验与清洗。

存在的理由：脏数据**不会报错**，只会让回测结论悄悄失效。
本项目已经踩过两次同类的坑（时区偏移一天、成交量单位差 100 倍），
两次都是靠人工抽查发现的 —— 这些规则就是把抽查沉淀成可重复执行的检查。

设计原则：
1. **只报告，不静默修改**。自动"修复"数据比脏数据更危险，
   因为你不知道它改了什么。修改必须显式调用 `clean()` 并留下记录。
2. 每条规则给出严重级别，让人能区分"必须处理"与"知道就行"。
3. 规则可单独测试，不依赖真实数据源。
"""
import logging
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from ..core.constants import get_price_limit

logger = logging.getLogger(__name__)


class Severity(Enum):
    """问题严重级别"""
    #: 数据错误，用它回测的结论一定不可信
    ERROR = "错误"
    #: 可疑，需人工判断（可能是真实极端行情，也可能是脏数据）
    WARNING = "可疑"
    #: 信息，通常无需处理
    INFO = "提示"


@dataclass
class Issue:
    """一条数据问题"""
    rule: str
    severity: Severity
    vt_symbol: str
    count: int = 0
    detail: str = ""
    #: 出问题的日期样例，最多保留几个用于排查
    samples: list = field(default_factory=list)

    def __str__(self) -> str:
        s = f"[{self.severity.value}] {self.vt_symbol} {self.rule}: {self.detail}"
        if self.samples:
            preview = ", ".join(str(x) for x in self.samples[:3])
            s += f" (样例: {preview})"
        return s


class BarValidator:
    """K 线数据校验器"""

    def __init__(self, price_limit_tolerance: float = 1.5,
                 max_samples: int = 5) -> None:
        """
        :param price_limit_tolerance: 涨跌幅容忍倍数。
            超过 `涨跌停幅度 × 该倍数` 才判定为数据错误。
            留余量是因为：除权日、连续停牌复牌后的首日、
            新股上市首日都可能出现远超涨跌停的真实涨跌幅。
        :param max_samples: 每条问题保留几个日期样例
        """
        self.tolerance = price_limit_tolerance
        self.max_samples = max_samples

    def validate(self, df: pd.DataFrame, vt_symbol: str,
                 trading_dates: list | None = None) -> list[Issue]:
        """校验单个标的的 K 线。

        :param df: DatetimeIndex + open/high/low/close/volume 列
        :param trading_dates: 交易日历，提供时才做缺失日检测
        """
        issues: list[Issue] = []
        if df.empty:
            issues.append(Issue("空数据", Severity.ERROR, vt_symbol,
                                detail="没有任何 K 线"))
            return issues

        issues += self._check_ohlc(df, vt_symbol)
        issues += self._check_non_positive(df, vt_symbol)
        issues += self._check_price_jump(df, vt_symbol)
        issues += self._check_volume_price_conflict(df, vt_symbol)
        issues += self._check_duplicates(df, vt_symbol)
        issues += self._check_monotonic(df, vt_symbol)
        if trading_dates is not None:
            issues += self._check_missing_dates(df, vt_symbol, trading_dates)
        return issues

    # ------------------------------------------------------------ 规则

    def _samples(self, index) -> list:
        return [str(x)[:19] for x in list(index)[:self.max_samples]]

    def _check_ohlc(self, df: pd.DataFrame, vt_symbol: str) -> list[Issue]:
        """OHLC 之间的逻辑关系必须成立"""
        issues = []

        bad = df[df["high"] < df["low"]]
        if len(bad):
            issues.append(Issue("最高价低于最低价", Severity.ERROR, vt_symbol,
                                len(bad), f"{len(bad)} 根 K 线 high < low",
                                self._samples(bad.index)))

        for col in ("open", "close"):
            out = df[(df[col] > df["high"]) | (df[col] < df["low"])]
            if len(out):
                issues.append(Issue(f"{col} 超出高低价区间", Severity.ERROR,
                                    vt_symbol, len(out),
                                    f"{len(out)} 根 K 线 {col} 不在 [low, high] 内",
                                    self._samples(out.index)))
        return issues

    def _check_non_positive(self, df: pd.DataFrame, vt_symbol: str) -> list[Issue]:
        """价格必须为正。停牌日价格为 0 是常见情况，单独归类为可疑而非错误。"""
        issues = []
        price_cols = ["open", "high", "low", "close"]

        zero = df[(df[price_cols] == 0).all(axis=1)]
        if len(zero):
            issues.append(Issue("价格全为零", Severity.WARNING, vt_symbol,
                                len(zero),
                                f"{len(zero)} 根 K 线价格全为 0（通常是停牌）",
                                self._samples(zero.index)))

        neg = df[(df[price_cols] < 0).any(axis=1)]
        if len(neg):
            issues.append(Issue("价格为负", Severity.ERROR, vt_symbol, len(neg),
                                f"{len(neg)} 根 K 线出现负价格",
                                self._samples(neg.index)))

        nan = df[df[price_cols].isna().any(axis=1)]
        if len(nan):
            issues.append(Issue("价格缺失", Severity.ERROR, vt_symbol, len(nan),
                                f"{len(nan)} 根 K 线价格为空",
                                self._samples(nan.index)))
        return issues

    def _check_price_jump(self, df: pd.DataFrame, vt_symbol: str) -> list[Issue]:
        """涨跌幅超过理论涨跌停一定倍数 → 数据错误或复权跳变"""
        symbol = vt_symbol.split(".")[0]
        limit = get_price_limit(symbol) * self.tolerance

        valid = df[df["close"] > 0]
        if len(valid) < 2:
            return []

        pct = valid["close"].pct_change()
        jump = valid[pct.abs() > limit]
        if not len(jump):
            return []

        worst = pct.abs().max()
        return [Issue(
            "涨跌幅异常", Severity.WARNING, vt_symbol, len(jump),
            f"{len(jump)} 根 K 线涨跌幅超过 {limit:.0%}（最大 {worst:.1%}）"
            f"，可能是复权跳变、停牌复牌或数据错误",
            self._samples(jump.index),
        )]

    def _check_volume_price_conflict(self, df: pd.DataFrame,
                                     vt_symbol: str) -> list[Issue]:
        """成交量为 0 但价格有波动 —— 自相矛盾"""
        if "volume" not in df.columns:
            return []

        zero_vol = df[df["volume"] == 0]
        if not len(zero_vol):
            return []

        moved = zero_vol[zero_vol["high"] != zero_vol["low"]]
        if not len(moved):
            return []

        return [Issue(
            "零成交量但价格波动", Severity.ERROR, vt_symbol, len(moved),
            f"{len(moved)} 根 K 线成交量为 0 却有价格波动",
            self._samples(moved.index),
        )]

    def _check_duplicates(self, df: pd.DataFrame, vt_symbol: str) -> list[Issue]:
        dup = df.index[df.index.duplicated()]
        if not len(dup):
            return []
        return [Issue("重复时间戳", Severity.ERROR, vt_symbol, len(dup),
                      f"{len(dup)} 个时间戳重复", self._samples(dup))]

    def _check_monotonic(self, df: pd.DataFrame, vt_symbol: str) -> list[Issue]:
        if df.index.is_monotonic_increasing:
            return []
        return [Issue("时间未升序", Severity.ERROR, vt_symbol, 1,
                      "K 线未按时间升序排列，回测会读到乱序数据")]

    def _check_missing_dates(self, df: pd.DataFrame, vt_symbol: str,
                             trading_dates: list) -> list[Issue]:
        """对照交易日历找缺口。

        注意：长期停牌会造成合法缺失，因此归为可疑而非错误。
        """
        have = set(pd.DatetimeIndex(df.index).normalize())
        expect = set(pd.DatetimeIndex(trading_dates).normalize())

        # 只检查数据覆盖区间内的缺失，区间外的属于「还没下载」
        if not have:
            return []
        lo, hi = min(have), max(have)
        expect = {d for d in expect if lo <= d <= hi}

        missing = sorted(expect - have)
        if not missing:
            return []

        return [Issue(
            "交易日缺失", Severity.WARNING, vt_symbol, len(missing),
            f"区间内缺少 {len(missing)} 个交易日（可能长期停牌）",
            [str(d.date()) for d in missing[:self.max_samples]],
        )]


#: 「公告日必须晚于报告期」只对定期报告成立。
#: 实测 000001：这四张表 0 例违反；而 Capital 表 24/186 违反。
#: 原因是 Capital 的「报告期」实为**股本变更生效日**，变更通常先公告后生效，
#: 公告早于生效日是正常的，且对 point-in-time 无害（确实更早就知道了）。
PERIODIC_REPORT_TABLES = {"Balance", "Income", "CashFlow", "PershareIndex"}


class FinancialValidator:
    """财务数据校验器"""

    def __init__(self, max_samples: int = 5) -> None:
        self.max_samples = max_samples

    def validate(self, df: pd.DataFrame, vt_symbol: str, table: str) -> list[Issue]:
        issues: list[Issue] = []
        if df.empty:
            return [Issue("空数据", Severity.WARNING, vt_symbol,
                          detail=f"{table} 无数据")]

        issues += self._check_announce_after_report(df, vt_symbol, table)
        issues += self._check_duplicate_reports(df, vt_symbol, table)
        issues += self._check_future_announce(df, vt_symbol, table)
        return issues

    def _check_announce_after_report(self, df: pd.DataFrame, vt_symbol: str,
                                     table: str) -> list[Issue]:
        """定期报告的公告日不可能早于报告期。

        若出现，说明 point-in-time 过滤会放行尚未披露的数据 —— 直接导致前视偏差。
        非定期报告（如股本变更）不适用此规则。
        """
        if table not in PERIODIC_REPORT_TABLES:
            return []
        if "announce_date" not in df.columns or "report_date" not in df.columns:
            return []

        valid = df.dropna(subset=["announce_date", "report_date"])
        bad = valid[valid["announce_date"] < valid["report_date"]]
        if not len(bad):
            return []

        return [Issue(
            "公告日早于报告期", Severity.ERROR, vt_symbol, len(bad),
            f"{table} 有 {len(bad)} 期公告日早于报告期，会造成前视偏差",
            [str(d)[:10] for d in bad["report_date"].head(self.max_samples)],
        )]

    def _check_duplicate_reports(self, df: pd.DataFrame, vt_symbol: str,
                                 table: str) -> list[Issue]:
        """同一报告期出现多条记录 = **财报追溯重述**。

        这不是脏数据，反而是 point-in-time 的必要条件：
        2007 年报在 2008-03 首次披露，又在 2009-03 被重述，
        两条都保留才能还原「当时到底知道什么」。

        但它对取数逻辑提出要求：必须按 (报告期, 公告日) 排序，
        只按公告日排序会取到旧报告期的重述版。此处仅作提示。
        """
        if table not in PERIODIC_REPORT_TABLES:
            return []
        if "report_date" not in df.columns:
            return []

        n = int(df["report_date"].dropna().duplicated().sum())
        if not n:
            return []
        return [Issue("报告期存在重述", Severity.INFO, vt_symbol, n,
                      f"{table} 有 {n} 期被追溯重述（正常现象，"
                      f"get_asof 已按 (报告期, 公告日) 排序处理）")]

    def _check_future_announce(self, df: pd.DataFrame, vt_symbol: str,
                               table: str) -> list[Issue]:
        """公告日晚于今天 —— 数据源出错，会让该记录永远取不到"""
        if "announce_date" not in df.columns:
            return []
        future = df[df["announce_date"] > pd.Timestamp.today()]
        if not len(future):
            return []
        return [Issue(
            "公告日在未来", Severity.WARNING, vt_symbol, len(future),
            f"{table} 有 {len(future)} 期公告日晚于今天，这些记录永远不会被取到",
            [str(d)[:10] for d in future["announce_date"].head(self.max_samples)],
        )]


def summarize(issues: list[Issue]) -> pd.DataFrame:
    """把问题清单汇总成表：按规则统计标的数与问题数"""
    if not issues:
        return pd.DataFrame(columns=["规则", "级别", "标的数", "问题总数"])

    rows = {}
    for issue in issues:
        key = (issue.rule, issue.severity.value)
        r = rows.setdefault(key, {"规则": issue.rule, "级别": issue.severity.value,
                                  "标的数": set(), "问题总数": 0})
        r["标的数"].add(issue.vt_symbol)
        r["问题总数"] += max(issue.count, 1)

    df = pd.DataFrame([
        {**r, "标的数": len(r["标的数"])} for r in rows.values()
    ])
    order = {Severity.ERROR.value: 0, Severity.WARNING.value: 1,
             Severity.INFO.value: 2}
    return df.sort_values(["级别", "问题总数"],
                          key=lambda s: s.map(order) if s.name == "级别" else -s,
                          ).reset_index(drop=True)
