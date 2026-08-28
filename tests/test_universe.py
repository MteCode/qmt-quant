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


class TestFullMarketUniverse:
    """全市场 PIT 标的池：唯一同时消除幸存者偏差与成分股前视的实现"""

    @pytest.fixture
    def meta(self):
        return pd.DataFrame({
            "vt_symbol": ["000001.SZ", "600519.SH", "000004.SZ", "301999.SZ"],
            "name": ["平安银行", "贵州茅台", "国华退", "次新股"],
            "listing_date": [pd.Timestamp("1991-04-03"), pd.Timestamp("2001-08-27"),
                             pd.Timestamp("1990-12-01"), pd.Timestamp("2025-12-03")],
            "delist_date": [pd.NaT, pd.NaT, pd.Timestamp("2026-07-14"), pd.NaT],
            "status": ["listed", "listed", "delisted", "listed"],
        })

    def test_delisted_included_before_delisting(self, meta):
        """核心：退市股在退市**之前**必须留在标的池里。

        它缺席正是幸存者偏差的来源 —— 当年能买到，回测就必须能选到。
        """
        from qmtquant.universe.providers import FullMarketUniverse
        u = FullMarketUniverse(meta, min_days_since_ipo=0)
        assert "000004.SZSE" in u.get_universe(pd.Timestamp("2024-01-02"))

    def test_delisted_excluded_after_delisting(self, meta):
        from qmtquant.universe.providers import FullMarketUniverse
        u = FullMarketUniverse(meta, min_days_since_ipo=0)
        assert "000004.SZSE" not in u.get_universe(pd.Timestamp("2026-08-01"))

    def test_not_yet_listed_excluded(self, meta):
        from qmtquant.universe.providers import FullMarketUniverse
        u = FullMarketUniverse(meta, min_days_since_ipo=0)
        assert "301999.SZSE" not in u.get_universe(pd.Timestamp("2021-01-04"))

    def test_min_days_since_ipo(self, meta):
        from qmtquant.universe.providers import FullMarketUniverse
        u = FullMarketUniverse(meta, min_days_since_ipo=60)
        assert "301999.SZSE" not in u.get_universe(pd.Timestamp("2025-12-20"))
        assert "301999.SZSE" in u.get_universe(pd.Timestamp("2026-06-01"))

    def test_bias_report_is_clean_with_delisted(self, meta):
        from qmtquant.universe.providers import FullMarketUniverse
        r = FullMarketUniverse(meta).describe_bias()
        assert r.survivorship is False
        assert r.membership_lookahead is False
        assert r.is_clean is True

    def test_bias_report_flags_missing_delisted(self, meta):
        """名单里没有退市股时，必须如实报告幸存者偏差仍在"""
        from qmtquant.universe.providers import FullMarketUniverse
        live_only = meta[meta["status"] == "listed"]
        r = FullMarketUniverse(live_only).describe_bias()
        assert r.survivorship is True
        assert r.is_clean is False

    def test_missing_columns_rejected(self):
        from qmtquant.universe.providers import FullMarketUniverse
        with pytest.raises(ValueError, match="缺少列"):
            FullMarketUniverse(pd.DataFrame({"vt_symbol": ["000001.SZ"]}))


class TestNamedSymbolsBias:
    """使用者直接点名的标的不该被报成幸存者偏差。

    误报会让真正的偏差警告失去分量 —— 每次回测都亮红灯，
    用户就不看了，而选股回测里那个警告是致命的。
    """

    def test_named_symbols_no_survivorship(self):
        from qmtquant.universe.providers import StaticUniverse
        r = StaticUniverse(["510300.SH"], source="命令行指定",
                           from_index_snapshot=False).describe_bias()
        assert not r.survivorship
        assert not r.membership_lookahead
        assert r.is_clean

    def test_still_warns_about_selection_bias(self):
        """数据里没有偏差，不代表「为什么选这几只」没有偏差"""
        from qmtquant.universe.providers import StaticUniverse
        r = StaticUniverse(["600519.SH"], from_index_snapshot=False
                           ).describe_bias()
        assert any("选择动作" in n for n in r.notes)

    def test_index_snapshot_still_flagged(self):
        from qmtquant.universe.providers import StaticUniverse
        r = StaticUniverse(["600519.SH"], source="沪深300 当前成分快照"
                           ).describe_bias()
        assert r.survivorship and r.membership_lookahead

    def test_default_assumes_snapshot(self):
        """默认必须是「有偏差」—— 安全的方向是宁可误报也不漏报"""
        from qmtquant.universe.providers import StaticUniverse
        assert StaticUniverse(["600519.SH"]).describe_bias().survivorship
