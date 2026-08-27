"""横截面动量轮动测试。

重点验证 skip_recent 的语义 —— 它是动量因子里最容易写错的一环：
跳过最近 N 日是为了规避短期反转效应，边界off-by-one 会让信号被反转污染。
"""
import pandas as pd
import pytest

from qmtquant.core.constants import Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.backtest_engine import BacktestEngine
from qmtquant.strategy.momentum import MomentumRotationStrategy

BASE = {"lookback": 5, "skip_recent": 0, "rebalance_days": 1,
        "max_holdings": 2, "min_turnover": 0}


def make(setting=None, symbols=None) -> MomentumRotationStrategy:
    symbols = symbols or ["600519.SSE", "000001.SZSE"]
    return MomentumRotationStrategy(
        BacktestEngine(), "T", symbols, dict(BASE, **(setting or {})))


def bar(symbol="600519", exchange=Exchange.SSE, close=10.0, turnover=1e9,
        dt="2024-01-02") -> BarData:
    return BarData(symbol=symbol, exchange=exchange,
                   datetime=pd.Timestamp(dt).to_pydatetime(),
                   interval=Interval.DAILY, open_price=close, high_price=close,
                   low_price=close, close_price=close, volume=1_000_000,
                   turnover=turnover)


def feed(s: MomentumRotationStrategy, vt_symbol: str, prices: list[float]):
    for p in prices:
        s.closes[vt_symbol].append(p)


class TestValidation:
    def test_lookback_minimum(self):
        with pytest.raises(ValueError, match="lookback"):
            make({"lookback": 1})

    def test_skip_recent_non_negative(self):
        with pytest.raises(ValueError, match="skip_recent"):
            make({"skip_recent": -1})

    def test_rebalance_days_minimum(self):
        with pytest.raises(ValueError, match="rebalance_days"):
            make({"rebalance_days": 0})

    def test_max_holdings_minimum(self):
        with pytest.raises(ValueError, match="max_holdings"):
            make({"max_holdings": 0})

    def test_min_turnover_non_negative(self):
        with pytest.raises(ValueError, match="min_turnover"):
            make({"min_turnover": -1})

    def test_float_params_coerced(self):
        """参数寻优从 DataFrame 传入的是 float64，
        用作 deque(maxlen=) 会 TypeError，而该异常常被外层吞掉"""
        s = make({"lookback": 20.0, "skip_recent": 5.0,
                  "rebalance_days": 10.0, "max_holdings": 3.0})
        for name in ("lookback", "skip_recent", "rebalance_days", "max_holdings"):
            assert isinstance(getattr(s, name), int), name


class TestMomentumComputation:
    def test_none_until_enough_bars(self):
        s = make({"lookback": 5, "skip_recent": 2})
        feed(s, "600519.SSE", [10.0] * 6)
        assert s.compute_momentum("600519.SSE") is None
        s.closes["600519.SSE"].append(10.0)
        assert s.compute_momentum("600519.SSE") is not None

    def test_unknown_symbol(self):
        assert make().compute_momentum("999999.SSE") is None

    def test_simple_return_without_skip(self):
        s = make({"lookback": 5, "skip_recent": 0})
        feed(s, "600519.SSE", [10, 11, 12, 13, 20])
        # 起点 prices[-5]=10，终点为最新价 20
        assert s.compute_momentum("600519.SSE") == pytest.approx(1.0)

    def test_skip_excludes_recent_bars(self):
        """跳过的那段涨跌不应计入动量 —— 这正是 skip_recent 的目的"""
        s = make({"lookback": 4, "skip_recent": 2})
        # 窗口 6 根：起点 prices[-6]=10，终点 prices[-3]=20，
        # 最后两根暴跌 100→1 必须被排除
        feed(s, "600519.SSE", [10, 12, 15, 20, 100, 1])
        assert s.compute_momentum("600519.SSE") == pytest.approx(1.0)

    def test_skip_boundary_off_by_one(self):
        """skip_recent=1 应恰好排除最后一根"""
        s = make({"lookback": 3, "skip_recent": 1})
        feed(s, "600519.SSE", [10, 15, 20, 999])
        assert s.compute_momentum("600519.SSE") == pytest.approx(1.0)

    def test_zero_start_price_rejected(self):
        s = make({"lookback": 3, "skip_recent": 0})
        feed(s, "600519.SSE", [0.0, 5.0, 10.0])
        assert s.compute_momentum("600519.SSE") is None

    def test_negative_momentum(self):
        s = make({"lookback": 3, "skip_recent": 0})
        feed(s, "600519.SSE", [20, 15, 10])
        assert s.compute_momentum("600519.SSE") == pytest.approx(-0.5)


class TestIndicatorUpdate:
    def test_suspended_bar_skipped(self):
        """停牌日价格无意义，计入会污染动量窗口"""
        s = make()
        b = bar()
        b.suspended = True
        s.update_indicators({"600519.SSE": b})
        assert len(s.closes["600519.SSE"]) == 0

    def test_zero_price_skipped(self):
        s = make()
        s.update_indicators({"600519.SSE": bar(close=0.0)})
        assert len(s.closes["600519.SSE"]) == 0

    def test_normal_bar_recorded(self):
        s = make()
        s.update_indicators({"600519.SSE": bar(close=42.0)})
        assert list(s.closes["600519.SSE"]) == [42.0]


class TestSelection:
    def _two_stocks(self, setting=None):
        s = make(setting)
        feed(s, "600519.SSE", [10, 11, 12, 13, 20])   # +100%
        feed(s, "000001.SZSE", [10, 10, 10, 10, 11])  # +10%
        return s

    def test_ranks_by_momentum_desc(self):
        s = self._two_stocks()
        assert s.select({"600519.SSE": bar(), "000001.SZSE": bar()},
                        ["000001.SZSE", "600519.SSE"])[0] == "600519.SSE"

    def test_reverse_mode_picks_losers(self):
        """IC 显著为负（-0.0575, t=-4.6），反向模式在逻辑上值得一试"""
        s = self._two_stocks({"reverse": True})
        assert s.select({"600519.SSE": bar(), "000001.SZSE": bar()},
                        ["000001.SZSE", "600519.SSE"])[0] == "000001.SZSE"

    def test_turnover_filter_excludes(self):
        s = self._two_stocks({"min_turnover": 1e10})
        assert s.select({"600519.SSE": bar(turnover=1e9),
                         "000001.SZSE": bar(turnover=1e9)},
                        ["600519.SSE", "000001.SZSE"]) == []

    def test_missing_bar_excluded(self):
        s = self._two_stocks()
        assert s.select({"600519.SSE": bar()},
                        ["600519.SSE", "000001.SZSE"]) == ["600519.SSE"]

    def test_insufficient_history_excluded(self):
        s = make()
        feed(s, "600519.SSE", [10, 11])   # 不足 lookback
        assert s.select({"600519.SSE": bar()}, ["600519.SSE"]) == []

    def test_scores_recorded_for_review(self):
        s = self._two_stocks()
        s.select({"600519.SSE": bar(), "000001.SZSE": bar()},
                 ["600519.SSE", "000001.SZSE"])
        assert s.last_scores["600519.SSE"] == pytest.approx(1.0)

    def test_scores_reset_each_call(self):
        """上一期的分数残留会让复盘看到不存在的持仓候选"""
        s = self._two_stocks()
        s.select({"600519.SSE": bar(), "000001.SZSE": bar()},
                 ["600519.SSE", "000001.SZSE"])
        s.select({"600519.SSE": bar()}, ["600519.SSE"])
        assert "000001.SZSE" not in s.last_scores

    def test_rebalance_counted(self):
        s = self._two_stocks()
        for _ in range(3):
            s.select({"600519.SSE": bar()}, ["600519.SSE"])
        assert s.rebalance_count == 3


class TestStatePersistence:
    def test_counters_in_variables(self):
        s = make()
        assert "rebalance_count" in s.variables
        assert "last_selection" in s.variables

    def test_restore(self):
        s = make()
        s.restore_variables({"rebalance_count": 9,
                             "last_selection": ["600519.SSE"],
                             "pos": {"600519.SSE": 100}})
        assert s.rebalance_count == 9
        assert s.last_selection == ["600519.SSE"]
        assert s.get_pos("600519.SSE") == 100
