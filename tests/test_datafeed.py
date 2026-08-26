"""数据源测试：重点验证周线/月线合成的正确性。

这部分不依赖 xtquant，可在任意环境运行。
"""
import pandas as pd
import pytest

from qmtquant.core.constants import Interval
from qmtquant.datafeed.xt_feed import XtDataFeed


@pytest.fixture
def feed(tmp_path):
    return XtDataFeed(str(tmp_path), dividend_type="front")


def write_daily(feed: XtDataFeed, vt_symbol: str, df: pd.DataFrame) -> None:
    df.to_parquet(feed._path(vt_symbol, Interval.DAILY))


def make_daily(dates: list[str], opens, highs, lows, closes,
               volumes=None) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    n = len(dates)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes if volumes is not None else [1000] * n,
        "amount": [10000] * n,
    }, index=idx)


class TestResampleWeekly:
    def test_weekly_ohlc_correct(self, feed):
        """周线 = 首日开盘、末日收盘、期间最高最低、成交量求和"""
        # 2024-01-01(一) 是元旦休市，用 01-02 到 01-05 构成完整一周
        df = make_daily(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            opens=[10.0, 11.0, 12.0, 11.5],
            highs=[11.0, 13.0, 12.5, 12.0],
            lows=[9.5, 10.8, 11.0, 11.0],
            closes=[10.8, 12.0, 11.6, 11.9],
            volumes=[100, 200, 300, 400],
        )
        write_daily(feed, "000001.SZSE", df)

        result = feed.resample_from_daily(["000001.SZSE"], Interval.WEEKLY)
        assert result["ok"] == ["000001.SZSE"]

        out = pd.read_parquet(feed._path("000001.SZSE", Interval.WEEKLY))
        assert len(out) == 1
        row = out.iloc[0]
        assert row["open"] == 10.0      # 周一开盘
        assert row["close"] == 11.9     # 周五收盘
        assert row["high"] == 13.0      # 周内最高
        assert row["low"] == 9.5        # 周内最低
        assert row["volume"] == 1000    # 求和

    def test_weekly_splits_across_weeks(self, feed):
        """跨周数据应切成多根周线"""
        df = make_daily(
            ["2024-01-02", "2024-01-03",           # 第 1 周
             "2024-01-08", "2024-01-09", "2024-01-10"],  # 第 2 周
            opens=[10, 11, 20, 21, 22],
            highs=[12, 12, 23, 23, 24],
            lows=[9, 10, 19, 20, 21],
            closes=[11, 11.5, 21, 22, 23],
        )
        write_daily(feed, "000001.SZSE", df)
        feed.resample_from_daily(["000001.SZSE"], Interval.WEEKLY)

        out = pd.read_parquet(feed._path("000001.SZSE", Interval.WEEKLY))
        assert len(out) == 2
        assert out.iloc[0]["open"] == 10
        assert out.iloc[0]["close"] == 11.5
        assert out.iloc[1]["open"] == 20
        assert out.iloc[1]["close"] == 23

    def test_suspended_days_excluded(self, feed):
        """停牌日（volume=0）不能污染周线的开收盘价"""
        df = make_daily(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            opens=[0.0, 11.0, 12.0, 11.5],     # 周一停牌，价格为 0
            highs=[0.0, 13.0, 12.5, 12.0],
            lows=[0.0, 10.8, 11.0, 11.0],
            closes=[0.0, 12.0, 11.6, 11.9],
            volumes=[0, 200, 300, 400],        # 周一 volume=0
        )
        write_daily(feed, "000001.SZSE", df)
        feed.resample_from_daily(["000001.SZSE"], Interval.WEEKLY)

        out = pd.read_parquet(feed._path("000001.SZSE", Interval.WEEKLY))
        # 开盘价应取周二的 11.0，而不是停牌日的 0
        assert out.iloc[0]["open"] == 11.0
        assert out.iloc[0]["low"] == 10.8
        assert out.iloc[0]["volume"] == 900

    def test_monthly_resample(self, feed):
        df = make_daily(
            ["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-29"],
            opens=[10, 15, 20, 25],
            highs=[16, 16, 26, 26],
            lows=[9, 14, 19, 24],
            closes=[15, 15.5, 25, 25.5],
        )
        write_daily(feed, "000001.SZSE", df)
        feed.resample_from_daily(["000001.SZSE"], Interval.MONTHLY)

        out = pd.read_parquet(feed._path("000001.SZSE", Interval.MONTHLY))
        assert len(out) == 2
        assert out.iloc[0]["open"] == 10
        assert out.iloc[0]["close"] == 15.5
        assert out.iloc[1]["open"] == 20
        assert out.iloc[1]["close"] == 25.5

    def test_missing_daily_reports_failure(self, feed):
        """没有日线时应报失败，而不是静默产出空文件"""
        result = feed.resample_from_daily(["999999.SZSE"], Interval.WEEKLY)
        assert result["failed"] == ["999999.SZSE"]
        assert not feed._path("999999.SZSE", Interval.WEEKLY).exists()


class TestLoadBars:
    def test_load_and_filter_by_date(self, feed):
        df = make_daily(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            opens=[10, 11, 12], highs=[11, 12, 13],
            lows=[9, 10, 11], closes=[10.5, 11.5, 12.5],
        )
        write_daily(feed, "000001.SZSE", df)

        bars = feed.load_bars(["000001.SZSE"], "2024-01-03", "2024-01-04")
        assert len(bars) == 2
        assert bars[0].open_price == 11
        assert bars[0].vt_symbol == "000001.SZSE"
        assert bars[0].interval is Interval.DAILY

    def test_suspended_flag(self, feed):
        df = make_daily(["2024-01-02"], opens=[10], highs=[10],
                        lows=[10], closes=[10], volumes=[0])
        write_daily(feed, "000001.SZSE", df)

        bars = feed.load_bars(["000001.SZSE"], "2024-01-01", "2024-12-31")
        assert bars[0].suspended is True

    def test_missing_file_returns_empty(self, feed):
        assert feed.load_bars(["999999.SZSE"], "2024-01-01", "2024-12-31") == []

    def test_has_data(self, feed):
        assert not feed.has_data("000001.SZSE", Interval.DAILY)
        write_daily(feed, "000001.SZSE",
                    make_daily(["2024-01-02"], [10], [10], [10], [10]))
        assert feed.has_data("000001.SZSE", Interval.DAILY)


class TestSummary:
    def test_summary_counts_files(self, feed):
        for code in ["000001.SZSE", "600519.SSE"]:
            write_daily(feed, code,
                        make_daily(["2024-01-02"], [10], [10], [10], [10]))
        df = feed.summary()
        assert not df.empty
        row = df[df["周期"] == "1d"].iloc[0]
        assert row["标的数"] == 2

    def test_empty_store(self, feed):
        assert feed.summary().empty
