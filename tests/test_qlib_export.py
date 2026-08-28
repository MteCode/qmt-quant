"""Qlib 数据导出测试。

两处错了不会报错、只会静默出错的地方：

1. **.bin 的偏移量** —— 写错会让整条序列在日历上错位。
   Qlib 照样读得出来，只是每个值都对应到了错误的日期。
2. **成分区间的边界** —— 调出日算不算在册。算了的话调样日会同时
   出现新旧两批，实测 2017-06-30 查出 330 只而非 300 只。
"""
import numpy as np
import pandas as pd
import pytest

from qmtquant.datafeed.qlib_export import (
    QlibExporter,
    from_qlib_code,
    read_bin,
    to_qlib_code,
    write_bin,
)


class TestCodeConversion:
    def test_sse(self):
        assert to_qlib_code("600000.SSE") == "sh600000"

    def test_szse(self):
        assert to_qlib_code("000001.SZSE") == "sz000001"

    def test_bse(self):
        assert to_qlib_code("430047.BSE") == "bj430047"

    def test_roundtrip(self):
        for vt in ("600000.SSE", "000001.SZSE", "430047.BSE"):
            assert from_qlib_code(to_qlib_code(vt)) == vt

    def test_unknown_exchange_rejected(self):
        with pytest.raises(ValueError, match="未知交易所"):
            to_qlib_code("600000.NYSE")

    def test_unknown_prefix_rejected(self):
        with pytest.raises(ValueError, match="未知 Qlib 代码前缀"):
            from_qlib_code("us600000")


class TestBinFormat:
    @staticmethod
    def _cal(n=10):
        return pd.bdate_range("2024-01-02", periods=n)

    def test_roundtrip_from_start(self, tmp_path):
        cal = self._cal()
        vals = np.arange(10, dtype=float)
        write_bin(tmp_path / "close.day.bin", vals, 0)
        got = read_bin(tmp_path / "close.day.bin", cal)
        assert list(got.index) == list(cal)
        assert got.to_numpy() == pytest.approx(vals)

    def test_offset_aligns_dates(self, tmp_path):
        """偏移量写错会让整条序列错位 —— Qlib 照样能读，只是日期全错"""
        cal = self._cal()
        write_bin(tmp_path / "close.day.bin",
                  np.array([100.0, 101.0, 102.0]), 5)
        got = read_bin(tmp_path / "close.day.bin", cal)
        assert list(got.index) == list(cal[5:8])
        assert got.iloc[0] == pytest.approx(100.0)

    def test_float32_precision(self, tmp_path):
        """Qlib 的格式就是 float32，高价股会有可见误差"""
        write_bin(tmp_path / "c.day.bin", np.array([8137.234567]), 0)
        got = read_bin(tmp_path / "c.day.bin", self._cal(1))
        assert got.iloc[0] == pytest.approx(8137.234567, rel=1e-6)

    def test_nan_preserved(self, tmp_path):
        """停牌日必须写 NaN 占位，不能跳过 —— 跳过会让后续全部错位"""
        write_bin(tmp_path / "c.day.bin", np.array([1.0, np.nan, 3.0]), 0)
        got = read_bin(tmp_path / "c.day.bin", self._cal())
        assert np.isnan(got.iloc[1])
        assert got.iloc[2] == pytest.approx(3.0)

    def test_empty_file(self, tmp_path):
        (tmp_path / "e.day.bin").write_bytes(b"")
        assert read_bin(tmp_path / "e.day.bin", self._cal()).empty


class TestIndexMembership:
    """成分区间的边界。

    真 bug：调出时把区间末端设成「它消失的那一期」，导致调样日
    新旧两批同时在册 —— 实测 2017-06-30 查出 330 只而非 300 只。
    那等于让策略在个股被剔除当天仍持有它。
    """

    @staticmethod
    def _csv(tmp_path, rows):
        path = tmp_path / "w.csv"
        pd.DataFrame(rows, columns=["date", "symbol", "weight"]).to_csv(
            path, index=False)
        return str(path)

    @staticmethod
    def _spans(codes):
        t = pd.Timestamp("2020-01-01")
        return {c: (t, t) for c in codes}

    def _lines(self, path):
        return [ln.split("\t")
                for ln in path.read_text(encoding="utf-8").strip().splitlines()]

    def test_exit_boundary_is_previous_period(self, tmp_path):
        """A 在第 3 期被 B 顶替：A 的区间应止于第 2 期"""
        csv = self._csv(tmp_path, [
            ("2020-01-31", "600000.SSE", 1.0),
            ("2020-02-29", "600000.SSE", 1.0),
            ("2020-03-31", "000001.SZSE", 1.0),
        ])
        exporter = QlibExporter("x", tmp_path / "out")
        path = exporter.write_index_members(
            csv, "test", self._spans(["600000.SSE", "000001.SZSE"]))
        rows = {r[0]: r[1:] for r in self._lines(path)}
        assert rows["sh600000"] == ["2020-01-31", "2020-02-29"], (
            "调出区间应止于上一期，不是消失当期")
        assert rows["sz000001"][0] == "2020-03-31"

    def test_rejoin_creates_two_spans(self, tmp_path):
        """调出后又调入应写成两段区间，而非一段跨越空窗期"""
        csv = self._csv(tmp_path, [
            ("2020-01-31", "600000.SSE", 1.0),
            ("2020-02-29", "000001.SZSE", 1.0),
            ("2020-03-31", "600000.SSE", 1.0),
        ])
        exporter = QlibExporter("x", tmp_path / "out")
        path = exporter.write_index_members(
            csv, "test", self._spans(["600000.SSE", "000001.SZSE"]))
        rows = [r for r in self._lines(path) if r[0] == "sh600000"]
        assert len(rows) == 2, f"应有两段在册区间，实际 {rows}"

    def test_still_member_uses_last_period(self, tmp_path):
        """从未调出的股票，区间末端用最后一期"""
        csv = self._csv(tmp_path, [
            ("2020-01-31", "600000.SSE", 1.0),
            ("2020-02-29", "600000.SSE", 1.0),
        ])
        exporter = QlibExporter("x", tmp_path / "out")
        path = exporter.write_index_members(csv, "test",
                                            self._spans(["600000.SSE"]))
        assert self._lines(path)[0][2] == "2020-02-29"

    def test_skips_symbols_without_price_data(self, tmp_path):
        """本地没行情的标的不能写进 instruments —— Qlib 会找不到 .bin 而报错"""
        csv = self._csv(tmp_path, [
            ("2020-01-31", "600000.SSE", 1.0),
            ("2020-01-31", "999999.SSE", 1.0),
        ])
        exporter = QlibExporter("x", tmp_path / "out")
        path = exporter.write_index_members(csv, "test",
                                            self._spans(["600000.SSE"]))
        text = path.read_text(encoding="utf-8")
        assert "sh600000" in text and "sh999999" not in text


class TestCalendarAndFeatures:
    """日历必须是所有标的的并集，且每只股票在日历上连续。"""

    @staticmethod
    def _make_store(tmp_path, data: dict):
        """data = {vt_symbol: {日期字符串: 收盘价}}"""
        for vt, series in data.items():
            code, _, ex = vt.rpartition(".")
            d = tmp_path / "1d" / ex
            d.mkdir(parents=True, exist_ok=True)
            idx = [int(k.replace("-", "")) for k in series]
            df = pd.DataFrame(
                {"open": list(series.values()), "high": list(series.values()),
                 "low": list(series.values()), "close": list(series.values()),
                 "volume": [1000.0] * len(series),
                 "amount": [1e6] * len(series)},
                index=idx)
            df.to_parquet(d / f"{code}.parquet")
        return str(tmp_path)

    def test_calendar_is_union(self, tmp_path):
        """A 停牌那天 B 有交易 —— 日历必须包含它，否则 B 的数据会错位"""
        store = self._make_store(tmp_path, {
            "600000.SSE": {"2024-01-02": 10.0, "2024-01-04": 12.0},
            "000001.SZSE": {"2024-01-02": 20.0, "2024-01-03": 21.0,
                            "2024-01-04": 22.0},
        })
        exporter = QlibExporter(store, tmp_path / "out")
        cal = exporter.build_calendar(
            ["600000.SSE", "000001.SZSE"], "2024-01-01", "2024-01-31")
        assert len(cal) == 3, f"应为三个交易日的并集，实际 {list(cal)}"

    def test_gap_filled_with_nan(self, tmp_path):
        """停牌日在区间内要填 NaN，保持与日历连续对齐"""
        store = self._make_store(tmp_path, {
            "600000.SSE": {"2024-01-02": 10.0, "2024-01-04": 12.0},
            "000001.SZSE": {"2024-01-02": 20.0, "2024-01-03": 21.0,
                            "2024-01-04": 22.0},
        })
        exporter = QlibExporter(store, tmp_path / "out")
        cal = exporter.build_calendar(
            ["600000.SSE", "000001.SZSE"], "2024-01-01", "2024-01-31")
        exporter.export_features(["600000.SSE"], fields=("close",))

        got = read_bin(
            tmp_path / "out" / "features" / "sh600000" / "close.day.bin", cal)
        assert len(got) == 3
        assert np.isnan(got.iloc[1]), "停牌日应为 NaN 占位"
        assert got.iloc[2] == pytest.approx(12.0), "占位后末值仍须对齐"

    def test_offset_when_listed_late(self, tmp_path):
        """晚上市的股票要写正确的起始偏移，不能从 0 开始"""
        store = self._make_store(tmp_path, {
            "600000.SSE": {"2024-01-02": 10.0, "2024-01-03": 11.0,
                           "2024-01-04": 12.0},
            "000001.SZSE": {"2024-01-04": 22.0},
        })
        exporter = QlibExporter(store, tmp_path / "out")
        cal = exporter.build_calendar(
            ["600000.SSE", "000001.SZSE"], "2024-01-01", "2024-01-31")
        exporter.export_features(["000001.SZSE"], fields=("close",))

        got = read_bin(
            tmp_path / "out" / "features" / "sz000001" / "close.day.bin", cal)
        assert list(got.index) == [cal[2]], "应只覆盖上市后的那一天"
        assert got.iloc[0] == pytest.approx(22.0)

    def test_factor_is_one(self, tmp_path):
        """本地存的已是后复权价，factor 恒为 1 —— 与 Qlib 官方数据约定不同"""
        store = self._make_store(
            tmp_path, {"600000.SSE": {"2024-01-02": 10.0}})
        exporter = QlibExporter(store, tmp_path / "out")
        cal = exporter.build_calendar(["600000.SSE"], "2024-01-01", "2024-01-31")
        exporter.export_features(["600000.SSE"], fields=("close", "factor"))
        got = read_bin(
            tmp_path / "out" / "features" / "sh600000" / "factor.day.bin", cal)
        assert got.iloc[0] == pytest.approx(1.0)

    def test_export_requires_calendar(self, tmp_path):
        exporter = QlibExporter(str(tmp_path), tmp_path / "out")
        with pytest.raises(RuntimeError, match="build_calendar"):
            exporter.export_features(["600000.SSE"])

    def test_calendar_written_as_text(self, tmp_path):
        store = self._make_store(tmp_path, {
            "600000.SSE": {"2024-01-02": 10.0, "2024-01-03": 11.0}})
        exporter = QlibExporter(store, tmp_path / "out")
        exporter.build_calendar(["600000.SSE"], "2024-01-01", "2024-01-31")
        path = exporter.write_calendar()
        assert path.read_text(encoding="utf-8").split() == [
            "2024-01-02", "2024-01-03"]
