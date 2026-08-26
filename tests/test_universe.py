"""标的池测试：重点是偏差过滤的正确性。"""
import pandas as pd
import pytest

from qmtquant.universe.providers import (
    HistoricalUniverse,
    PointInTimeUniverse,
    StaticUniverse,
)


class TestStaticUniverse:
    def test_returns_same_list_any_date(self):
        u = StaticUniverse(["000001.SZ", "600519.SH"])
        assert u.get_universe(pd.Timestamp("2020-01-02")) == ["000001.SZSE", "600519.SSE"]
        assert u.get_universe(pd.Timestamp("2026-01-02")) == ["000001.SZSE", "600519.SSE"]

    def test_declares_its_biases(self):
        """静态池必须如实声明两种偏差，不能假装干净"""
        r = StaticUniverse(["000001.SZ"]).describe_bias()
        assert r.survivorship is True
        assert r.membership_lookahead is True
        assert r.is_clean is False
        assert "⚠" in r.summary()


class TestPointInTimeUniverse:
    @pytest.fixture
    def base(self):
        return StaticUniverse(["000001.SZ", "600519.SH", "301999.SZ"])

    def test_excludes_not_yet_listed(self, base):
        """核心：不能在标的上市之前就把它选进来"""
        u = PointInTimeUniverse(base, listing_dates={
            "000001.SZ": "1991-04-03",
            "600519.SH": "2001-08-27",
            "301999.SZ": "2025-12-03",     # 2025 年才上市
        }, min_days_since_ipo=0)

        early = u.get_universe(pd.Timestamp("2021-01-04"))
        assert "301999.SZSE" not in early
        assert "600519.SSE" in early

        late = u.get_universe(pd.Timestamp("2026-06-01"))
        assert "301999.SZSE" in late

    def test_min_days_since_ipo(self, base):
        """次新股无历史数据可算指标，上市初期应排除"""
        u = PointInTimeUniverse(base, listing_dates={
            "000001.SZ": "2024-01-01",
            "600519.SH": "2001-08-27",
            "301999.SZ": "2001-08-27",
        }, min_days_since_ipo=60)

        # 上市第 30 天，不足 60 天
        assert "000001.SZSE" not in u.get_universe(pd.Timestamp("2024-01-31"))
        # 上市第 90 天
        assert "000001.SZSE" in u.get_universe(pd.Timestamp("2024-04-01"))

    def test_excludes_after_delisting(self, base):
        u = PointInTimeUniverse(base, listing_dates={
            "000001.SZ": "1991-04-03", "600519.SH": "2001-08-27",
            "301999.SZ": "2001-08-27",
        }, delist_dates={"000001.SZ": "2023-06-01"}, min_days_since_ipo=0)

        assert "000001.SZSE" in u.get_universe(pd.Timestamp("2023-05-31"))
        assert "000001.SZSE" not in u.get_universe(pd.Timestamp("2023-06-01"))
        assert "000001.SZSE" not in u.get_universe(pd.Timestamp("2024-01-01"))

    def test_unknown_listing_date_excluded(self, base):
        """拿不到上市日就保守剔除，宁可少选也不能引入未来数据"""
        u = PointInTimeUniverse(base, listing_dates={"600519.SH": "2001-08-27"},
                                min_days_since_ipo=0)
        result = u.get_universe(pd.Timestamp("2024-01-01"))
        assert result == ["600519.SSE"]

    def test_bias_report_marks_listing_filtered(self, base):
        u = PointInTimeUniverse(base, listing_dates={"600519.SH": "2001-08-27"})
        r = u.describe_bias()
        assert r.listing_filtered is True
        # 上市日过滤修不掉幸存者偏差，不能谎报
        assert r.survivorship is True


class TestHistoricalUniverse:
    @pytest.fixture
    def csv(self, tmp_path):
        p = tmp_path / "hist.csv"
        pd.DataFrame({
            "date": ["2020-01-02", "2020-01-02", "2022-06-01", "2022-06-01"],
            "symbol": ["000001.SZ", "600519.SH", "600519.SH", "000002.SZ"],
        }).to_csv(p, index=False)
        return p

    def test_asof_semantics(self, csv):
        """取不晚于该日的最近一期名单，不能用未来的调整结果"""
        u = HistoricalUniverse(csv)

        # 第一期之前 -> 空
        assert u.get_universe(pd.Timestamp("2019-12-31")) == []
        # 第一期与第二期之间 -> 用第一期
        assert u.get_universe(pd.Timestamp("2021-05-01")) == ["000001.SZSE", "600519.SSE"]
        # 第二期之后 -> 用第二期
        assert u.get_universe(pd.Timestamp("2023-01-01")) == ["000002.SZSE", "600519.SSE"]

    def test_all_symbols_covers_union(self, csv):
        u = HistoricalUniverse(csv)
        assert set(u.all_symbols()) == {"000001.SZSE", "000002.SZSE", "600519.SSE"}

    def test_bias_report_clean(self, csv):
        r = HistoricalUniverse(csv).describe_bias()
        assert r.survivorship is False
        assert r.membership_lookahead is False
        assert r.is_clean is True

    def test_missing_columns_rejected(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"d": ["2020-01-02"], "s": ["000001.SZ"]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="date"):
            HistoricalUniverse(p)
