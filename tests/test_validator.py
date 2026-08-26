"""数据校验与 point-in-time 取数测试。"""
import pandas as pd
import pytest

from qmtquant.datafeed.financial import FinancialStore
from qmtquant.datafeed.validator import (
    BarValidator,
    FinancialValidator,
    Severity,
    get_price_limit,
    summarize,
)


def bars(dates, o, h, low, c, v=None):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({"open": o, "high": h, "low": low, "close": c,
                         "volume": v if v is not None else [1000] * len(dates)},
                        index=idx)


class TestPriceLimit:
    @pytest.mark.parametrize("symbol,expected", [
        ("600519", 0.10),   # 主板
        ("000001", 0.10),
        ("300750", 0.20),   # 创业板
        ("301999", 0.20),
        ("688981", 0.20),   # 科创板
        ("830799", 0.30),   # 北交所
    ])
    def test_by_prefix(self, symbol, expected):
        assert get_price_limit(symbol) == expected


class TestBarValidator:
    @pytest.fixture
    def v(self):
        return BarValidator()

    def test_clean_data_no_issues(self, v):
        df = bars(["2024-01-02", "2024-01-03"],
                  [10, 10.2], [10.5, 10.6], [9.8, 10.0], [10.2, 10.4])
        assert v.validate(df, "000001.SZSE") == []

    def test_high_below_low(self, v):
        df = bars(["2024-01-02"], [10], [9.0], [10.5], [10])
        issues = v.validate(df, "000001.SZSE")
        assert any(i.rule == "最高价低于最低价" and i.severity is Severity.ERROR
                   for i in issues)

    def test_close_outside_range(self, v):
        df = bars(["2024-01-02"], [10], [10.5], [9.8], [12.0])
        issues = v.validate(df, "000001.SZSE")
        assert any("close 超出高低价区间" in i.rule for i in issues)

    def test_negative_price_is_error(self, v):
        """前复权对高分红股会产生负价格 —— 实测 601919 最低 -5.14"""
        df = bars(["2024-01-02"], [-3.4], [-3.3], [-3.5], [-3.44])
        issues = v.validate(df, "601919.SSE")
        assert any(i.rule == "价格为负" and i.severity is Severity.ERROR
                   for i in issues)

    def test_zero_price_is_warning_not_error(self, v):
        """停牌日价格为 0 很常见，不该报错"""
        df = bars(["2024-01-02"], [0], [0], [0], [0], v=[0])
        issues = v.validate(df, "000001.SZSE")
        zero = [i for i in issues if i.rule == "价格全为零"]
        assert zero and zero[0].severity is Severity.WARNING

    def test_price_jump_detected(self, v):
        """主板单日涨 30% 不可能，必是数据错误或复权跳变"""
        df = bars(["2024-01-02", "2024-01-03"],
                  [10, 13], [10.1, 13.1], [9.9, 12.9], [10, 13])
        issues = v.validate(df, "600000.SSE")
        assert any(i.rule == "涨跌幅异常" for i in issues)

    def test_normal_limit_up_not_flagged(self, v):
        """正常涨停不该报警"""
        df = bars(["2024-01-02", "2024-01-03"],
                  [10, 11], [10, 11], [10, 11], [10, 11])
        issues = v.validate(df, "600000.SSE")
        assert not any(i.rule == "涨跌幅异常" for i in issues)

    def test_chinext_allows_larger_move(self, v):
        """创业板 20% 涨跌停，15% 涨幅属正常"""
        df = bars(["2024-01-02", "2024-01-03"],
                  [10, 11.5], [10, 11.5], [10, 11.5], [10, 11.5])
        assert not any(i.rule == "涨跌幅异常"
                       for i in v.validate(df, "300750.SZSE"))

    def test_zero_volume_with_price_move(self, v):
        df = bars(["2024-01-02"], [10], [10.5], [9.5], [10], v=[0])
        issues = v.validate(df, "000001.SZSE")
        assert any(i.rule == "零成交量但价格波动" and i.severity is Severity.ERROR
                   for i in issues)

    def test_unsorted_index(self, v):
        df = bars(["2024-01-03", "2024-01-02"],
                  [10, 10], [10, 10], [10, 10], [10, 10])
        assert any(i.rule == "时间未升序" for i in v.validate(df, "000001.SZSE"))

    def test_duplicate_timestamps(self, v):
        df = bars(["2024-01-02", "2024-01-02"],
                  [10, 10], [10, 10], [10, 10], [10, 10])
        assert any(i.rule == "重复时间戳" for i in v.validate(df, "000001.SZSE"))

    def test_missing_trading_dates(self, v):
        df = bars(["2024-01-02", "2024-01-04"],
                  [10, 10], [10, 10], [10, 10], [10, 10])
        cal = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        issues = v.validate(df, "000001.SZSE", trading_dates=list(cal))
        assert any(i.rule == "交易日缺失" for i in issues)

    def test_empty_dataframe(self, v):
        issues = v.validate(pd.DataFrame(), "000001.SZSE")
        assert issues and issues[0].severity is Severity.ERROR


class TestFinancialValidator:
    @pytest.fixture
    def v(self):
        return FinancialValidator()

    def test_announce_before_report_is_error(self, v):
        df = pd.DataFrame({
            "report_date": [pd.Timestamp("2024-06-30")],
            "announce_date": [pd.Timestamp("2024-05-01")],
        })
        issues = v.validate(df, "000001.SZSE", "Income")
        assert any(i.rule == "公告日早于报告期" and i.severity is Severity.ERROR
                   for i in issues)

    def test_capital_table_exempt(self, v):
        """Capital 的「报告期」是股本变更生效日，先公告后生效属正常"""
        df = pd.DataFrame({
            "report_date": [pd.Timestamp("2024-06-30")],
            "announce_date": [pd.Timestamp("2024-05-01")],
        })
        issues = v.validate(df, "000001.SZSE", "Capital")
        assert not any(i.rule == "公告日早于报告期" for i in issues)

    def test_restatement_is_info_not_error(self, v):
        """同一报告期多条 = 追溯重述，是 point-in-time 的必要条件"""
        df = pd.DataFrame({
            "report_date": [pd.Timestamp("2007-12-31")] * 2,
            "announce_date": [pd.Timestamp("2008-03-20"),
                              pd.Timestamp("2009-03-20")],
        })
        issues = v.validate(df, "000001.SZSE", "Income")
        dup = [i for i in issues if i.rule == "报告期存在重述"]
        assert dup and dup[0].severity is Severity.INFO

    def test_future_announce_flagged(self, v):
        df = pd.DataFrame({
            "report_date": [pd.Timestamp("2024-06-30")],
            "announce_date": [pd.Timestamp.today() + pd.Timedelta(days=30)],
        })
        assert any(i.rule == "公告日在未来"
                   for i in v.validate(df, "000001.SZSE", "Income"))


class TestGetAsofRestatement:
    """财报重述场景下的 point-in-time 取数。

    实测过的真实 bug：000001 在 2009-03-20 同一天公告了 2008 年报
    与 2007 年报重述版；只按公告日排序会取到 2007 年的重述数据
    （EPS 0.97）而非 2008 年报（EPS 0.20），因子值完全错误且不报错。
    """

    @pytest.fixture
    def store(self, tmp_path):
        s = FinancialStore(str(tmp_path))
        df = pd.DataFrame({
            "report_date": pd.to_datetime(
                ["2007-12-31", "2008-12-31", "2007-12-31", "2009-03-31"]),
            "announce_date": pd.to_datetime(
                ["2008-03-20", "2009-03-20", "2009-03-20", "2009-04-24"]),
            "eps": [1.27, 0.20, 0.97, 0.36],
        })
        df.to_parquet(s._path("000001.SZSE", "Income"))
        return s

    def test_returns_latest_report_period(self, store):
        """同日公告新年报与旧年报重述，必须取新年报"""
        r = store.get_asof("000001.SZSE", "Income", "2009-04-01")
        assert r["report_date"] == pd.Timestamp("2008-12-31")
        assert r["eps"] == 0.20

    def test_before_restatement_sees_original(self, store):
        """重述之前只能看到最初披露的版本"""
        r = store.get_asof("000001.SZSE", "Income", "2008-06-01")
        assert r["report_date"] == pd.Timestamp("2007-12-31")
        assert r["eps"] == 1.27

    def test_picks_latest_revision_of_same_period(self, store):
        """同一报告期有多版时，取当时已知的最新修订"""
        r = store.get_asof("000001.SZSE", "Income", "2009-03-25",
                           fields=["report_date", "eps"])
        # 2008 年报报告期更新，应优先于 2007 的重述
        assert r["report_date"] == pd.Timestamp("2008-12-31")

    def test_no_data_before_first_announcement(self, store):
        assert store.get_asof("000001.SZSE", "Income", "2008-01-01") is None

    def test_latest_query_gets_newest_period(self, store):
        r = store.get_asof("000001.SZSE", "Income", "2024-01-01")
        assert r["report_date"] == pd.Timestamp("2009-03-31")


class TestSummarize:
    def test_empty(self):
        assert summarize([]).empty

    def test_groups_by_rule(self):
        v = BarValidator()
        df = bars(["2024-01-02"], [10], [9.0], [10.5], [10])
        issues = v.validate(df, "000001.SZSE") + v.validate(df, "000002.SZSE")
        out = summarize(issues)
        row = out[out["规则"] == "最高价低于最低价"].iloc[0]
        assert row["标的数"] == 2
