"""回测报告与图表测试。

图表本身不校验像素，只校验**数据变换是否正确** ——
画错的图比没有图更危险，因为它看起来很可信。
"""
import pandas as pd
import pytest

from qmtquant.engine.performance import PerformanceStats
from qmtquant.report import charts
from qmtquant.report.html_report import BacktestReport, bars_to_frame
from qmtquant.universe.providers import StaticUniverse


@pytest.fixture
def equity():
    idx = pd.bdate_range("2023-01-02", periods=60)
    # 先涨后跌再回升，便于验证回撤区间
    values = ([100 + i for i in range(20)]
              + [120 - i * 2 for i in range(20)]
              + [80 + i for i in range(20)])
    return pd.Series(values, index=idx, dtype=float)


@pytest.fixture
def trades():
    return pd.DataFrame({
        "datetime": pd.to_datetime(["2023-01-05", "2023-02-01",
                                    "2023-01-06", "2023-02-02"]),
        "symbol": ["000001.SZSE", "000001.SZSE", "600519.SSE", "600519.SSE"],
        "direction": ["买入", "卖出", "买入", "卖出"],
        "price": [10.0, 12.0, 100.0, 90.0],
        "volume": [1000.0, 1000.0, 100.0, 100.0],
        "commission": [5.0, 17.0, 5.0, 14.0],
        "amount": [10000.0, 12000.0, 10000.0, 9000.0],
        "strategy": ["S"] * 4,
    })


@pytest.fixture
def stats():
    return PerformanceStats(
        start_date="2023-01-02", end_date="2023-03-24", trading_days=60,
        initial_capital=100.0, final_capital=99.0, total_return=-0.01,
        annual_return=-0.04, max_drawdown=-0.35, max_drawdown_duration=40,
        sharpe_ratio=-0.5, calmar_ratio=-0.11, volatility=0.3,
        total_trades=4, win_rate=0.5, profit_factor=1.2,
        total_commission=41.0, turnover_rate=0.41,
    )


class TestCharts:
    def test_equity_normalized_to_one(self, equity):
        """策略与基准都要归一化，否则量级差异让图没法看"""
        fig = charts.equity_curve(equity)
        y = fig.data[0].y
        assert y[0] == pytest.approx(1.0)
        assert y[-1] == pytest.approx(equity.iloc[-1] / equity.iloc[0])

    def test_benchmark_aligned_and_normalized(self, equity):
        bench = pd.Series(range(2000, 2000 + len(equity)),
                          index=equity.index, dtype=float)
        fig = charts.equity_curve(equity, bench)
        assert len(fig.data) == 2
        assert fig.data[1].y[0] == pytest.approx(1.0)

    def test_benchmark_reindexed_to_strategy_dates(self, equity):
        """基准多出来的日期不能让曲线错位"""
        extra = pd.bdate_range("2022-01-03", periods=300)
        bench = pd.Series(range(len(extra)), index=extra, dtype=float)
        fig = charts.equity_curve(equity, bench)
        assert len(fig.data[1].x) == len(equity)

    def test_drawdown_is_non_positive(self, equity):
        fig = charts.drawdown_curve(equity)
        assert max(fig.data[0].y) <= 0.001

    def test_drawdown_matches_manual_calc(self, equity):
        fig = charts.drawdown_curve(equity)
        expected = (equity / equity.cummax() - 1).min() * 100
        assert min(fig.data[0].y) == pytest.approx(expected)

    def test_monthly_heatmap_shape(self, equity):
        fig = charts.monthly_heatmap(equity)
        assert fig.data
        # 12 个月一行
        assert len(fig.data[0].x) == 12

    def test_price_chart_marks_both_directions(self, trades):
        bars = pd.DataFrame(
            {"close": [10.0, 11.0, 12.0]},
            index=pd.to_datetime(["2023-01-05", "2023-01-20", "2023-02-01"]))
        sub = trades[trades["symbol"] == "000001.SZSE"]
        fig = charts.price_with_trades(bars, sub, "000001.SZSE")
        names = {t.name for t in fig.data}
        assert {"收盘价", "买入", "卖出"} <= names

    def test_symbol_pnl_signs(self, trades):
        """000001 赚钱、600519 亏钱，方向不能反"""
        pnl = charts._realized_pnl_by_symbol(trades)
        assert pnl["000001.SZSE"] > 0
        assert pnl["600519.SSE"] < 0

    def test_symbol_pnl_includes_commission(self, trades):
        """盈亏必须扣掉双边手续费"""
        pnl = charts._realized_pnl_by_symbol(trades)
        gross = (12.0 - 10.0) * 1000
        assert pnl["000001.SZSE"] < gross

    def test_position_count_never_negative(self, equity, trades):
        fig = charts.position_count(equity.index, trades)
        assert min(fig.data[0].y) >= 0

    def test_empty_inputs_do_not_crash(self):
        empty = pd.Series(dtype=float)
        for fig in (charts.equity_curve(empty), charts.drawdown_curve(empty),
                    charts.monthly_heatmap(empty)):
            assert fig is not None
        assert charts.symbol_pnl(pd.DataFrame()) is not None


class TestBarsToFrame:
    def test_returns_none_for_multi_symbol(self):
        assert bars_to_frame([object()], ["a.SZSE", "b.SZSE"]) is None

    def test_returns_none_for_empty(self):
        assert bars_to_frame([], ["a.SZSE"]) is None


class TestReport:
    def _html(self, **kw):
        return BacktestReport(**kw).to_html()

    def test_contains_core_sections(self, stats, equity, trades):
        h = self._html(stats=stats,
                       equity_df=pd.DataFrame({"equity": equity}),
                       trades_df=trades)
        for kw in ["总收益率", "最大回撤", "绩效明细", "成交明细"]:
            assert kw in h

    def test_bias_warning_rendered(self, stats, equity):
        bias = StaticUniverse(["000001.SZ"]).describe_bias()
        h = self._html(stats=stats, equity_df=pd.DataFrame({"equity": equity}),
                       bias_report=bias)
        assert "标的池存在偏差" in h
        assert "幸存者偏差" in h

    def test_undersized_orders_warning(self, stats, equity):
        """不足一手的标的被静默排除，报告必须点出来"""
        h = self._html(stats=stats, equity_df=pd.DataFrame({"equity": equity}),
                       undersized_orders={"600519.SSE": 12, "000651.SZSE": 8})
        assert "因不足一手而完全买不进" in h
        assert "600519.SSE" in h

    def test_no_undersized_section_when_clean(self, stats, equity):
        h = self._html(stats=stats, equity_df=pd.DataFrame({"equity": equity}))
        assert "因不足一手而完全买不进" not in h

    def test_rejected_orders_grouped(self, stats, equity):
        h = self._html(stats=stats, equity_df=pd.DataFrame({"equity": equity}),
                       rejected_orders=["开盘涨停，无法买入"] * 3 + ["资金不足"])
        assert "委托拒单统计" in h
        assert "开盘涨停，无法买入" in h

    def test_empty_trades_explains_why(self, stats, equity):
        h = self._html(stats=stats, equity_df=pd.DataFrame({"equity": equity}))
        assert "没有产生任何成交" in h

    def test_self_contained(self, stats, equity):
        """必须内联 plotly.js，否则离线打开是空白页"""
        h = self._html(stats=stats, equity_df=pd.DataFrame({"equity": equity}))
        assert "<script>" in h
        assert len(h) > 1_000_000

    def test_escapes_title(self, stats, equity):
        h = self._html(stats=stats, equity_df=pd.DataFrame({"equity": equity}),
                       title="<script>alert(1)</script>")
        assert "<script>alert(1)</script>" not in h
        assert "&lt;script&gt;" in h

    def test_save_writes_file(self, stats, equity, tmp_path):
        p = BacktestReport(stats=stats,
                           equity_df=pd.DataFrame({"equity": equity})
                           ).save(tmp_path / "sub" / "r.html")
        assert p.exists()
        assert p.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
