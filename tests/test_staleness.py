"""数据新鲜度检查。

## 为什么原来没有

BarValidator 检查 OHLC 逻辑、跳空、重复、单调性、缺失交易日 ——
唯独不检查「这份数据是不是旧的」。于是「下载脚本挂了三天」这件事
没有任何东西会发现：回测照跑，只是用的是三天前的世界。

## 为什么必须是跨标的判断

拿今天的日期比会得到三种假警报：

- 退市股停在 2017 年是**正确**的
- 今天可能是周末或节假日，本来就没有新数据
- 所有标的都停在同一天时，那天就是最后一个交易日

所以基准取全体标的里最新的那个日期。只要还有标的在更新，
那个日期就是市场的当前状态。
"""
from __future__ import annotations

import pandas as pd
import pytest

from qmtquant.datafeed.validator import Severity, check_staleness


def _cohort(n: int, date: str = "20260904") -> dict:
    return {f"{600000 + i}.SSE": date for i in range(n)}


class TestNoFalseAlarms:
    def test_all_same_date_is_clean(self):
        """全市场停在同一天 = 那天就是最后一个交易日，不是停更。"""
        assert check_staleness(_cohort(100)) == []

    def test_weekend_gap_tolerated(self):
        """周五收盘后到周一，落后 3 天是正常的。"""
        c = _cohort(100, "20260904")
        c["600999.SSE"] = "20260901"          # 落后 3 天
        assert check_staleness(c, max_lag_days=3) == []

    def test_small_cohort_skipped(self):
        """样本太小时「最新日期」本身就不可信，不做判断。"""
        c = {"600000.SSE": "20260904", "600001.SSE": "20200101"}
        assert check_staleness(c, min_cohort=20) == []

    def test_known_stopped_excluded(self):
        c = _cohort(100)
        c["600005.SSE"] = "20170123"
        assert check_staleness(c, known_stopped={"600005.SSE"}) == []


class TestDetectsRealBreakage:
    def test_recent_break_is_error(self):
        """近期停更是真问题 —— 下载可能挂了。"""
        c = _cohort(100)
        c["600999.SSE"] = "20260820"          # 落后 15 天
        issues = check_staleness(c)
        errs = [i for i in issues if i.severity is Severity.ERROR]
        assert errs, "近期停更应当报 ERROR"
        assert "600999.SSE" in errs[0].detail

    def test_long_gone_is_info_not_error(self):
        """退市股不该和「昨天下载挂了」混在一起报 ——
        混在一起会让人一眼看不出哪个要紧，很快就学会忽略整个检查。"""
        c = _cohort(100)
        c["600005.SSE"] = "20170123"          # 九年前
        issues = check_staleness(c)
        assert not [i for i in issues if i.severity is Severity.ERROR]
        infos = [i for i in issues if i.severity is Severity.INFO]
        assert infos and "600005.SSE" in infos[0].detail

    def test_both_kinds_reported_separately(self):
        c = _cohort(100)
        c["600999.SSE"] = "20260820"
        c["600005.SSE"] = "20170123"
        issues = check_staleness(c)
        sev = {i.severity for i in issues}
        assert sev == {Severity.ERROR, Severity.INFO}

    def test_whole_market_frozen_is_caught(self):
        """下载全挂：只有少数标的更新到今天，大部分停在几天前。

        这是最危险的情形 —— 数据看着"有"，但整体落后。
        """
        c = _cohort(100, "20260820")
        c["600000.SSE"] = "20260904"          # 只有一只是新的
        issues = check_staleness(c)
        errs = [i for i in issues if i.severity is Severity.ERROR]
        assert errs and errs[0].count == 99


class TestInputRobustness:
    @pytest.mark.parametrize("val", [
        "20260904", 20260904, "2026-09-04", pd.Timestamp("2026-09-04"),
    ])
    def test_accepts_common_date_forms(self, val):
        c = _cohort(50)
        c["600999.SSE"] = val
        assert check_staleness(c) == []

    def test_unparsable_dates_ignored_not_crash(self):
        c = _cohort(100)
        c["600999.SSE"] = "不是日期"
        c["600998.SSE"] = None
        issues = check_staleness(c)
        assert isinstance(issues, list)

    def test_empty_input(self):
        assert check_staleness({}) == []


class TestOnRealData:
    """跑真实数据，确认它在这个仓库里给出的是有意义的结果，
    而不是恒空或全量误报。"""

    def test_real_universe(self):
        import glob

        latest = {}
        for f in glob.glob("data/1d/*/*.parquet")[:800]:
            try:
                df = pd.read_parquet(f, columns=["close"])
            except Exception:                    # noqa: BLE001
                continue
            if len(df):
                parts = f.replace("\\", "/").split("/")
                latest[f"{parts[-1][:-8]}.{parts[-2]}"] = df.index[-1]
        if len(latest) < 100:
            pytest.skip("本地日线数据不足")

        issues = check_staleness(latest)
        # 不断言具体数量（会随数据更新而变），只断言它没在两个极端上
        assert len(issues) <= 2
        for i in issues:
            assert i.count < len(latest) * 0.5, \
                "报出超过一半标的停更，说明基准取错了"
