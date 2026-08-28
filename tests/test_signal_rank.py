"""外部信号选股策略测试。

关键不变量：
- **asof 取分**：调仓日取「不晚于它的最近一期」，绝不用未来的分数
- **没分数就空仓**，不静默降级到别的规则
"""
import numpy as np
import pandas as pd
import pytest

from qmtquant.core.constants import Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.backtest_engine import BacktestEngine
from qmtquant.strategy.signal_rank import SignalRankStrategy

A, B, C = "600000.SSE", "600001.SSE", "600002.SSE"


def make(setting=None, scores=None, min_turnover=0.0) -> SignalRankStrategy:
    return SignalRankStrategy(
        BacktestEngine(), "T", [A, B, C], setting or {},
        scores=scores, min_turnover=min_turnover)


def bar(symbol="600000", dt="2024-03-15", turnover=1e9) -> BarData:
    return BarData(symbol=symbol, exchange=Exchange.SSE,
                   datetime=pd.Timestamp(dt).to_pydatetime(),
                   interval=Interval.DAILY, open_price=10.0, high_price=10.0,
                   low_price=10.0, close_price=10.0, volume=1_000_000,
                   turnover=turnover)


def bars_at(dt, turnovers=None) -> dict:
    turnovers = turnovers or {}
    return {s: bar(s.split(".")[0], dt, turnovers.get(s, 1e9))
            for s in (A, B, C)}


def panel(rows: dict) -> pd.DataFrame:
    """rows = {日期: {vt_symbol: 分数}}"""
    return pd.DataFrame(rows).T.rename_axis("datetime").pipe(
        lambda d: d.set_axis(pd.DatetimeIndex(d.index)))


class TestValidation:
    def test_max_holdings_minimum(self):
        with pytest.raises(ValueError, match="max_holdings"):
            make({"max_holdings": 0})

    def test_rebalance_days_minimum(self):
        with pytest.raises(ValueError, match="rebalance_days"):
            make({"rebalance_days": 0})

    def test_float_params_coerced(self):
        s = make({"max_holdings": 30.0, "rebalance_days": 20.0})
        assert isinstance(s.max_holdings, int)
        assert isinstance(s.rebalance_days, int)


class TestRanking:
    def test_sorts_by_score_desc(self):
        s = make(scores=panel({"2024-01-31": {A: 0.1, B: 0.9, C: 0.5}}))
        assert s.select(bars_at("2024-03-15"), [A, B, C]) == [B, C, A]

    def test_excludes_nan_scores(self):
        s = make(scores=panel({"2024-01-31": {A: 0.1, B: np.nan, C: 0.5}}))
        assert s.select(bars_at("2024-03-15"), [A, B, C]) == [C, A]

    def test_excludes_missing_bar(self):
        s = make(scores=panel({"2024-01-31": {A: 0.1, B: 0.9, C: 0.5}}))
        got = s.select({A: bar("600000")}, [A, B, C])
        assert got == [A]

    def test_turnover_floor(self):
        """选出来买不进去的票没有意义"""
        s = make(scores=panel({"2024-01-31": {A: 0.9, B: 0.5, C: 0.1}}),
                 min_turnover=1e9)
        got = s.select(bars_at("2024-03-15", {A: 1e6}), [A, B, C])
        assert A not in got and got == [B, C]

    def test_records_scores_for_review(self):
        s = make(scores=panel({"2024-01-31": {A: 0.1, B: 0.9, C: 0.5}}))
        s.select(bars_at("2024-03-15"), [A, B, C])
        assert s.last_scores[B] == pytest.approx(0.9)

    def test_scores_reset_each_call(self):
        """上一期的分数残留会让复盘看到不存在的候选"""
        s = make(scores=panel({"2024-01-31": {A: 0.1, B: 0.9, C: 0.5}}))
        s.select(bars_at("2024-03-15"), [A, B, C])
        s.select({A: bar("600000")}, [A])
        assert B not in s.last_scores

    def test_rebalance_counted(self):
        s = make(scores=panel({"2024-01-31": {A: 0.1}}))
        for _ in range(3):
            s.select(bars_at("2024-03-15"), [A])
        assert s.rebalance_count == 3


class TestAsofSemantics:
    """分数面板可能只在部分日期出分（如月末），必须 asof 取用。
    精确匹配会让绝大多数调仓日拿不到分数而空仓。"""

    def _panel(self):
        return panel({
            "2024-01-31": {A: 0.9, B: 0.1},
            "2024-02-29": {A: 0.1, B: 0.9},
        })

    def test_uses_latest_available(self):
        s = make(scores=self._panel())
        assert s.select(bars_at("2024-02-15"), [A, B])[0] == A, "应用 1 月底那期"

    def test_switches_after_new_period(self):
        s = make(scores=self._panel())
        assert s.select(bars_at("2024-03-15"), [A, B])[0] == B, "应用 2 月底那期"

    def test_exact_date_uses_that_period(self):
        s = make(scores=self._panel())
        assert s.select(bars_at("2024-02-29"), [A, B])[0] == B

    def test_never_uses_future_scores(self):
        """核心：调仓日早于任何一期分数时必须空仓，不能用未来的"""
        s = make(scores=self._panel())
        assert s.select(bars_at("2024-01-02"), [A, B]) == []


class TestNoSilentFallback:
    def test_empty_without_scores(self):
        """没分数就空仓 —— 静默降级会让回测跑出一条看似正常的曲线，
        而你以为测的是模型信号"""
        assert make().select(bars_at("2024-03-15"), [A, B, C]) == []

    def test_empty_panel(self):
        s = make(scores=pd.DataFrame())
        assert s.select(bars_at("2024-03-15"), [A, B, C]) == []

    def test_warns_only_once(self, caplog):
        s = make()
        for _ in range(5):
            s.select(bars_at("2024-03-15"), [A])
        assert caplog.text.count("没有可用分数面板") <= 1


class TestStatePersistence:
    def test_counters_in_variables(self):
        s = make()
        assert "rebalance_count" in s.variables
        assert "last_selection" in s.variables

    def test_restore(self):
        s = make()
        s.restore_variables({"rebalance_count": 7, "last_selection": [A],
                             "pos": {A: 200}})
        assert s.rebalance_count == 7
        assert s.get_pos(A) == 200
