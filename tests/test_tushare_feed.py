"""Tushare 数据源测试。

不联网：通过注入假的 pro 对象来验证限流、重试、月度切分与代码转换。
真实连通性由 scripts/check_tushare.py 负责。
"""
import os
from unittest.mock import patch

import pandas as pd
import pytest

from qmtquant.config import TushareConfig
from qmtquant.datafeed.tushare_feed import (
    TushareClient,
    TushareError,
    TushareFeed,
    resolve_token,
)

CFG = TushareConfig(token="t" * 32, calls_per_minute=0, max_retry=3,
                    retry_base_delay=0.0)


class FakePro:
    """记录调用参数的假接口"""

    def __init__(self, responses=None, fail_times=0, exc=None):
        self.responses = responses or {}
        self.calls: list[tuple[str, dict]] = []
        self.fail_times = fail_times
        self.exc = exc or RuntimeError("网络抖动")

    def __getattr__(self, name):
        def call(**params):
            self.calls.append((name, params))
            if self.fail_times > 0:
                self.fail_times -= 1
                raise self.exc
            r = self.responses.get(name, pd.DataFrame())
            return r(**params) if callable(r) else r
        return call


def make_client(pro, cfg=CFG) -> TushareClient:
    c = TushareClient(cfg)
    c._pro = pro
    return c


def weight_rows(codes, trade_date):
    return pd.DataFrame({
        "index_code": ["000300.SH"] * len(codes),
        "con_code": codes,
        "trade_date": [trade_date] * len(codes),
        "weight": [1.0] * len(codes),
    })


class TestResolveToken:
    def test_env_wins_over_config(self):
        with patch.dict(os.environ, {"TUSHARE_TOKEN": "from_env"}):
            assert resolve_token(TushareConfig(token="from_cfg")) == "from_env"

    def test_falls_back_to_config(self):
        with patch.dict(os.environ, {"TUSHARE_TOKEN": ""}):
            assert resolve_token(TushareConfig(token="from_cfg")) == "from_cfg"

    def test_strips_whitespace(self):
        with patch.dict(os.environ, {"TUSHARE_TOKEN": "  abc  "}):
            assert resolve_token(None) == "abc"

    def test_raises_with_actionable_message(self):
        with patch.dict(os.environ, {"TUSHARE_TOKEN": ""}):
            with pytest.raises(TushareError, match="TUSHARE_TOKEN"):
                resolve_token(TushareConfig())


class TestRetry:
    def test_recovers_after_transient_failure(self):
        pro = FakePro({"stock_basic": pd.DataFrame({"ts_code": ["000001.SZ"]})},
                      fail_times=2)
        df = make_client(pro).query("stock_basic")
        assert len(df) == 1
        assert len(pro.calls) == 3

    def test_gives_up_after_max_retry(self):
        pro = FakePro(fail_times=99)
        with pytest.raises(TushareError, match="重试 3 次"):
            make_client(pro).query("stock_basic")
        assert len(pro.calls) == 3

    def test_points_error_fails_fast(self):
        """积分不足是配置问题，重试没有意义，且会拖慢排查"""
        pro = FakePro(fail_times=99, exc=RuntimeError("抱歉，您每分钟最多访问该接口，积分不足"))
        with pytest.raises(TushareError, match="积分不足"):
            make_client(pro).query("index_weight")
        assert len(pro.calls) == 1, "积分错误不应重试"

    def test_none_response_becomes_empty_df(self):
        pro = FakePro({"stock_basic": None})
        assert make_client(pro).query("stock_basic").empty


class TestThrottle:
    def test_sleeps_when_over_limit(self):
        cfg = TushareConfig(token="x" * 32, calls_per_minute=2,
                            retry_base_delay=0.0)
        client = make_client(FakePro(), cfg)
        with patch("qmtquant.datafeed.tushare_feed.time.sleep") as slept:
            for _ in range(3):
                client.query("stock_basic")
            assert slept.called, "超过每分钟上限应等待"

    def test_no_sleep_under_limit(self):
        cfg = TushareConfig(token="x" * 32, calls_per_minute=100,
                            retry_base_delay=0.0)
        client = make_client(FakePro(), cfg)
        with patch("qmtquant.datafeed.tushare_feed.time.sleep") as slept:
            for _ in range(5):
                client.query("stock_basic")
            assert not slept.called

    def test_zero_disables_throttle(self):
        client = make_client(FakePro())
        with patch("qmtquant.datafeed.tushare_feed.time.sleep") as slept:
            for _ in range(50):
                client.query("stock_basic")
            assert not slept.called


class TestIndexWeight:
    def test_requests_month_by_month(self):
        """单次最多 4000 行，沪深300 一次请求最多覆盖 13 个月，
        所以必须按月切分，否则静默截断"""
        pro = FakePro({"index_weight":
                       lambda **p: weight_rows(["000001.SZ"], p["start_date"])})
        TushareFeed(client=make_client(pro)).index_weight(
            "000300.SH", "2024-01-01", "2024-06-30")
        assert len(pro.calls) == 6
        starts = [c[1]["start_date"] for c in pro.calls]
        assert starts == ["20240101", "20240201", "20240301",
                          "20240401", "20240501", "20240601"]

    def test_month_end_is_last_day(self):
        pro = FakePro({"index_weight": lambda **p: pd.DataFrame()})
        TushareFeed(client=make_client(pro)).index_weight(
            "000300.SH", "2024-02-01", "2024-02-28")
        assert pro.calls[0][1]["end_date"] == "20240229", "闰年 2 月是 29 日"

    def test_converts_to_vt_symbol(self):
        pro = FakePro({"index_weight":
                       lambda **p: weight_rows(["000001.SZ", "600519.SH"],
                                               "20240131")})
        df = TushareFeed(client=make_client(pro)).index_weight(
            "000300.SH", "2024-01-01", "2024-01-31")
        assert set(df["symbol"]) == {"000001.SZSE", "600519.SSE"}

    def test_output_columns(self):
        pro = FakePro({"index_weight":
                       lambda **p: weight_rows(["000001.SZ"], "20240131")})
        df = TushareFeed(client=make_client(pro)).index_weight(
            "000300.SH", "2024-01-01", "2024-01-31")
        assert list(df.columns) == ["date", "symbol", "weight"]
        assert df["date"].iloc[0] == pd.Timestamp("2024-01-31")

    def test_deduplicates(self):
        pro = FakePro({"index_weight":
                       lambda **p: weight_rows(["000001.SZ"], "20240131")})
        df = TushareFeed(client=make_client(pro)).index_weight(
            "000300.SH", "2024-01-01", "2024-03-31")
        assert len(df) == 1, "三个月返回同一行，应去重"

    def test_empty_range(self):
        pro = FakePro()
        df = TushareFeed(client=make_client(pro)).index_weight(
            "000300.SH", "2024-03-05", "2024-03-01")
        assert df.empty
        assert list(df.columns) == ["date", "symbol", "weight"]
        assert not pro.calls

    def test_skips_empty_months(self):
        def resp(**p):
            return (weight_rows(["000001.SZ"], "20240131")
                    if p["start_date"] == "20240101" else pd.DataFrame())
        df = TushareFeed(client=make_client(FakePro({"index_weight": resp}))
                         ).index_weight("000300.SH", "2024-01-01", "2024-03-31")
        assert len(df) == 1

    def test_warns_on_row_cap(self, caplog):
        codes = [f"{i:06d}.SZ" for i in range(4000)]
        pro = FakePro({"index_weight": lambda **p: weight_rows(codes, "20240131")})
        TushareFeed(client=make_client(pro)).index_weight(
            "000300.SH", "2024-01-01", "2024-01-31")
        assert "截断" in caplog.text

    def test_sorted_by_date(self):
        def resp(**p):
            return weight_rows(["000001.SZ"], p["start_date"])
        df = TushareFeed(client=make_client(FakePro({"index_weight": resp}))
                         ).index_weight("000300.SH", "2024-01-01", "2024-03-31")
        assert df["date"].is_monotonic_increasing


class TestDailyBasic:
    def test_converts_symbol_and_date(self):
        pro = FakePro({"daily_basic": pd.DataFrame({
            "ts_code": ["600519.SH"], "trade_date": ["20240115"],
            "pe_ttm": [30.0], "pb": [8.0]})})
        df = TushareFeed(client=make_client(pro)).daily_basic("2024-01-15")
        assert df["symbol"].iloc[0] == "600519.SSE"
        assert df["trade_date"].iloc[0] == pd.Timestamp("2024-01-15")

    def test_accepts_both_date_formats(self):
        pro = FakePro({"daily_basic": pd.DataFrame()})
        feed = TushareFeed(client=make_client(pro))
        feed.daily_basic("2024-01-15")
        feed.daily_basic("20240115")
        assert [c[1]["trade_date"] for c in pro.calls] == ["20240115"] * 2

    def test_empty_passthrough(self):
        pro = FakePro({"daily_basic": pd.DataFrame()})
        assert TushareFeed(client=make_client(pro)).daily_basic("2024-01-15").empty


class TestFinaIndicator:
    def test_parses_announce_date(self):
        """公告日必须可用 —— 按报告期取数是前视偏差"""
        pro = FakePro({"fina_indicator": pd.DataFrame({
            "ts_code": ["600519.SH"], "end_date": ["20231231"],
            "ann_date": ["20240403"], "roe": [30.0]})})
        df = TushareFeed(client=make_client(pro)).fina_indicator("600519.SSE")
        assert df["ann_date"].iloc[0] == pd.Timestamp("2024-04-03")
        assert df["end_date"].iloc[0] == pd.Timestamp("2023-12-31")
        assert df["symbol"].iloc[0] == "600519.SSE"

    def test_sends_tushare_format_code(self):
        pro = FakePro({"fina_indicator": pd.DataFrame()})
        TushareFeed(client=make_client(pro)).fina_indicator("600519.SSE")
        assert pro.calls[0][1]["ts_code"] == "600519.SH"


class TestTradeDates:
    def test_returns_sorted_timestamps(self):
        pro = FakePro({"trade_cal": pd.DataFrame({
            "cal_date": ["20240104", "20240102", "20240103"]})})
        dates = TushareFeed(client=make_client(pro)).trade_dates(
            "2024-01-01", "2024-01-05")
        assert dates == [pd.Timestamp(f"2024-01-0{d}") for d in (2, 3, 4)]

    def test_only_open_days(self):
        pro = FakePro({"trade_cal": pd.DataFrame()})
        TushareFeed(client=make_client(pro)).trade_dates("2024-01-01", "2024-01-05")
        assert pro.calls[0][1]["is_open"] == "1"


class TestCheck:
    def test_reports_both_levels(self):
        pro = FakePro({
            "stock_basic": pd.DataFrame({"ts_code": ["000001.SZ"] * 5000}),
            "index_weight": weight_rows(["000001.SZ"], "20240131"),
        })
        info = make_client(pro).check()
        assert info["token_ok"] and info["points_2000_ok"]
        assert info["stock_count"] == 5000

    def test_empty_index_weight_means_no_points(self):
        """积分未到账时接口静默返回空，必须识别出来"""
        pro = FakePro({"stock_basic": pd.DataFrame({"ts_code": ["000001.SZ"]}),
                       "index_weight": pd.DataFrame()})
        info = make_client(pro).check()
        assert info["token_ok"]
        assert not info["points_2000_ok"]
        assert any("积分" in n for n in info["notes"])
