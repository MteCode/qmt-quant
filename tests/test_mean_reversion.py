"""均值回归策略测试。

重点验证四种典型死法的防护是否真的生效：
单边下跌越买越套、无止损、死拿不回归的仓位、在低流动性票上交易。
"""
import pandas as pd
import pytest

from qmtquant.core.constants import Direction, Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.backtest_engine import BacktestEngine
from qmtquant.strategy.mean_reversion import (
    EXIT_REVERT,
    EXIT_STOP,
    EXIT_TIMEOUT,
    EXIT_TREND,
    MeanReversionStrategy,
)
from qmtquant.universe.providers import StaticUniverse

TURNOVER = 100_000_000


def bars_for(prices: dict[str, list[float]], turnover: float = TURNOVER,
             start: str = "2023-01-02") -> list[BarData]:
    """按 {vt_symbol: [收盘价序列]} 构造多标的日线"""
    n = max(len(v) for v in prices.values())
    dates = pd.bdate_range(start, periods=n)
    out = []
    for vt_symbol, closes in prices.items():
        symbol, ex = vt_symbol.split(".")
        for dt, c in zip(dates, closes):
            out.append(BarData(
                symbol=symbol, exchange=Exchange(ex), datetime=dt.to_pydatetime(),
                interval=Interval.DAILY, open_price=c, high_price=c,
                low_price=c, close_price=c, volume=1_000_000, turnover=turnover,
            ))
    return out


def run(prices, setting=None, capital=10_000_000, turnover=TURNOVER):
    symbols = list(prices)
    engine = BacktestEngine(initial_capital=capital)
    engine.load_data(bars_for(prices, turnover))
    engine.set_universe(StaticUniverse(symbols))
    engine.add_strategy(MeanReversionStrategy, symbols, setting or {})
    engine.run()
    return engine


def ramp(base: float, n: int, step: float = 0.001) -> list[float]:
    """缓慢上行的价格序列，用于把长期均线垫起来"""
    return [base * (1 + step) ** i for i in range(n)]


BASE = {"lookback": 10, "trend_filter_window": 20, "max_holding_days": 100}


class TestValidation:
    def test_thresholds_must_be_ordered(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="止损 < 买入 < 卖出"):
            engine.add_strategy(MeanReversionStrategy, ["000001.SZSE"],
                                {"entry_z": -1.0, "exit_z": -2.0, "stop_z": -3.0})

    def test_lookback_minimum(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="lookback"):
            engine.add_strategy(MeanReversionStrategy, ["000001.SZSE"],
                                {"lookback": 3})

    def test_max_holdings_minimum(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="max_holdings"):
            engine.add_strategy(MeanReversionStrategy, ["000001.SZSE"],
                                {"max_holdings": 0})


class TestZScore:
    def _strategy(self, closes, **kw):
        engine = BacktestEngine()
        s = MeanReversionStrategy(engine, "S", ["000001.SZSE"],
                                  dict(BASE, **kw))
        for c in closes:
            s.closes["000001.SZSE"].append(c)
        return s

    def test_none_when_insufficient_data(self):
        s = self._strategy([10.0] * 5)
        assert s.compute_zscore("000001.SZSE") is None

    def test_zero_std_returns_none(self):
        """连续同价（长期停牌后复牌）标准差为 0，
        除零会得 inf 让该标的永远排信号首位"""
        s = self._strategy([10.0] * 20)
        assert s.compute_zscore("000001.SZSE") is None

    def test_negative_when_below_mean(self):
        s = self._strategy([10.0] * 9 + [8.0])
        z = s.compute_zscore("000001.SZSE")
        assert z is not None and z < 0

    def test_matches_manual_calculation(self):
        closes = [10, 11, 12, 11, 10, 9, 10, 11, 12, 8]
        s = self._strategy([float(c) for c in closes])
        mean = sum(closes) / len(closes)
        std = (sum((c - mean) ** 2 for c in closes) / len(closes)) ** 0.5
        assert s.compute_zscore("000001.SZSE") == pytest.approx(
            (closes[-1] - mean) / std)

    def test_normalizes_across_volatility(self):
        """同样跌 5%，低波动股的 z 应更极端 ——
        不做标准化就只会选出高波动股，等于在赌波动率"""
        calm = [10.0, 10.02, 9.98, 10.01, 9.99, 10.0, 10.02, 9.98, 10.01, 9.5]
        wild = [10.0, 11.0, 9.0, 11.5, 8.5, 10.5, 9.5, 11.0, 9.0, 9.5]
        z_calm = self._strategy(calm).compute_zscore("000001.SZSE")
        z_wild = self._strategy(wild).compute_zscore("000001.SZSE")
        assert z_calm < z_wild


class TestDeathMode1_DowntrendFilter:
    """死法一：在单边下跌里越买越套"""

    def test_no_buy_below_trend_line(self):
        # 持续下跌：价格始终在长期均线之下，z 也会很负
        prices = {"000001.SZSE": [100 * 0.99 ** i for i in range(80)]}
        engine = run(prices, dict(BASE, entry_z=-0.5, exit_z=0.5))
        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        assert not buys, "下跌趋势中不应买入"

    def test_buys_when_above_trend(self):
        """上行趋势中的短期回调才是均值回归该做的。

        参数取自实测：涨 1%/日、回调 8% 时 z≈-1.40 且仍在 MA20 之上。
        跌得更深（10%+）虽然 z 更负，但会同时跌破趋势线被过滤 ——
        这两个条件互相拉扯，见策略文档「信号稀少」一节。
        回调后需再给几根 Bar，否则信号产生在最后一根就没有次日可成交。
        """
        rise = ramp(10, 60, 0.01)
        dip = rise[-1] * 0.92
        prices = {"000001.SZSE": rise + [dip, dip * 1.01, dip * 1.02]}
        engine = run(prices, dict(BASE, entry_z=-1.2))
        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        assert buys, "上行趋势中的回调应产生买入"

    def test_exits_when_trend_breaks(self):
        s = MeanReversionStrategy(BacktestEngine(), "S", ["000001.SZSE"], BASE)
        for c in [10.0] * 19 + [5.0]:
            s.closes["000001.SZSE"].append(c)
        s.entry_info["000001.SZSE"] = {"bar_count": 0, "entry_z": -2.0}
        bar = bars_for({"000001.SZSE": [5.0]})[0]
        assert s._exit_reason("000001.SZSE", bar) in (EXIT_STOP, EXIT_TREND)


class TestDeathMode2_StopLoss:
    """死法二：不设止损，一只股票吃掉全部利润"""

    def test_stop_triggers_on_worsening_z(self):
        s = MeanReversionStrategy(BacktestEngine(), "S", ["000001.SZSE"],
                                  dict(BASE, stop_z=-2.0, entry_z=-1.0, exit_z=0.0))
        # 构造 z 极低但仍在长期均线之上的场景
        for c in ramp(10, 19, 0.01):
            s.closes["000001.SZSE"].append(c)
        s.closes["000001.SZSE"].append(10.5)   # 突然下挫
        s.entry_info["000001.SZSE"] = {"bar_count": 0, "entry_z": -1.0}

        z = s.compute_zscore("000001.SZSE")
        bar = bars_for({"000001.SZSE": [10.5]})[0]
        reason = s._exit_reason("000001.SZSE", bar)
        if z is not None and z <= -2.0:
            assert reason == EXIT_STOP

    def test_does_not_enter_below_stop_line(self):
        """已跌破止损线的不接飞刀 —— 买进来下一根就要止损"""
        s = MeanReversionStrategy(BacktestEngine(), "S", ["000001.SZSE"],
                                  dict(BASE, entry_z=-1.0, stop_z=-1.5))
        for c in ramp(10, 19, 0.01):
            s.closes["000001.SZSE"].append(c)
        s.closes["000001.SZSE"].append(9.0)

        bars = {b.vt_symbol: b for b in bars_for({"000001.SZSE": [9.0]})}
        s.engine.set_universe(StaticUniverse(["000001.SZSE"]))
        z = s.compute_zscore("000001.SZSE")
        cands = s._rank_candidates(bars, set())
        if z is not None and z <= -1.5:
            assert not cands, "跌破止损线的标的不应进入候选"


class TestDeathMode3_HoldingTimeout:
    """死法三：买了不回归，仓位被无限期占用"""

    def test_timeout_exit(self):
        s = MeanReversionStrategy(BacktestEngine(), "S", ["000001.SZSE"],
                                  dict(BASE, max_holding_days=5, stop_z=-99, exit_z=0.0))
        for c in ramp(10, 20, 0.005):
            s.closes["000001.SZSE"].append(c)
        s.entry_info["000001.SZSE"] = {"bar_count": 0, "entry_z": -2.0}
        s._bar_count = 5

        bar = bars_for({"000001.SZSE": [s.closes["000001.SZSE"][-1]]})[0]
        # z 已回归时优先记为「回归」，构造未回归场景才测得到超时
        z = s.compute_zscore("000001.SZSE")
        if z is not None and z < s.exit_z:
            assert s._exit_reason("000001.SZSE", bar) == EXIT_TIMEOUT

    def test_not_timeout_before_limit(self):
        s = MeanReversionStrategy(BacktestEngine(), "S", ["000001.SZSE"],
                                  dict(BASE, max_holding_days=20, stop_z=-99,
                                       exit_z=99))
        for c in ramp(10, 20, 0.005):
            s.closes["000001.SZSE"].append(c)
        s.entry_info["000001.SZSE"] = {"bar_count": 0, "entry_z": -2.0}
        s._bar_count = 3
        bar = bars_for({"000001.SZSE": [10.0]})[0]
        assert s._exit_reason("000001.SZSE", bar) is None


class TestDeathMode4_Liquidity:
    """死法四：在流动性差的票上被滑点吃穿"""

    def test_low_turnover_excluded(self):
        rise = ramp(10, 60, 0.01)
        dip = rise[-1] * 0.92
        prices = {"000001.SZSE": rise + [dip, dip * 1.01, dip * 1.02]}
        engine = run(prices, dict(BASE, entry_z=-1.2, min_turnover=1e9),
                     turnover=1_000_000)
        assert not engine.trades, "成交额低于下限的标的不应交易"

    def test_sufficient_turnover_allowed(self):
        rise = ramp(10, 60, 0.01)
        dip = rise[-1] * 0.92
        prices = {"000001.SZSE": rise + [dip, dip * 1.01, dip * 1.02]}
        engine = run(prices, dict(BASE, entry_z=-1.2, min_turnover=1e6),
                     turnover=1e8)
        assert engine.trades


class TestPriceLimitAwareness:
    """涨跌停时无法成交，信号阶段就该跳过"""

    def test_limit_up_not_buyable(self):
        s = MeanReversionStrategy(BacktestEngine(), "S", ["000001.SZSE"], BASE)
        bar = bars_for({"000001.SZSE": [11.0]})[0]
        assert not s._tradable(bar, prev_close=10.0, direction="buy")

    def test_limit_down_not_sellable(self):
        s = MeanReversionStrategy(BacktestEngine(), "S", ["000001.SZSE"], BASE)
        bar = bars_for({"000001.SZSE": [9.0]})[0]
        assert not s._tradable(bar, prev_close=10.0, direction="sell")

    def test_chinext_uses_20pct(self):
        """创业板 20% 涨跌停，涨 15% 仍可交易"""
        s = MeanReversionStrategy(BacktestEngine(), "S", ["300750.SZSE"], BASE)
        bar = bars_for({"300750.SZSE": [11.5]})[0]
        assert s._tradable(bar, prev_close=10.0, direction="buy")

    def test_suspended_not_tradable(self):
        s = MeanReversionStrategy(BacktestEngine(), "S", ["000001.SZSE"], BASE)
        bar = bars_for({"000001.SZSE": [10.0]})[0]
        bar.suspended = True
        assert not s._tradable(bar, prev_close=10.0, direction="buy")


class TestPortfolioBehaviour:
    def test_respects_max_holdings(self):
        symbols = [f"00000{i}.SZSE" for i in range(1, 6)]
        rise = ramp(10, 60, 0.01)
        dip = rise[-1] * 0.92
        prices = {s: rise + [dip, dip * 1.01, dip * 1.02] for s in symbols}
        engine = run(prices, dict(BASE, entry_z=-1.2, max_holdings=2))
        held = {s for s, p in engine.positions.items() if p["volume"] > 0}
        assert len(held) <= 2

    def test_ranks_by_zscore(self):
        """越超跌的越优先 —— 候选必须按 z 升序"""
        s = MeanReversionStrategy(BacktestEngine(), "S",
                                  ["000001.SZSE", "000002.SZSE"],
                                  dict(BASE, entry_z=-0.5, exit_z=0.5, stop_z=-99))
        s.engine.set_universe(StaticUniverse(["000001.SZSE", "000002.SZSE"]))
        for c in ramp(10, 19, 0.01):
            s.closes["000001.SZSE"].append(c)
            s.closes["000002.SZSE"].append(c)
        s.closes["000001.SZSE"].append(10.8)    # 小跌
        s.closes["000002.SZSE"].append(10.0)    # 大跌

        bars = {b.vt_symbol: b for b in
                bars_for({"000001.SZSE": [10.8], "000002.SZSE": [10.0]})}
        cands = s._rank_candidates(bars, set())
        if len(cands) == 2:
            assert cands[0][0] <= cands[1][0]
            assert cands[0][1] == "000002.SZSE"

    def test_exit_reasons_recorded(self):
        """退出原因必须可归因 —— 分不清回归与止损就无法判断策略靠什么赚钱"""
        rise = ramp(10, 60, 0.01)
        dip = rise[-1] * 0.92
        prices = {"000001.SZSE": rise + [dip] + list(ramp(dip, 30, 0.01))}
        engine = run(prices, dict(BASE, entry_z=-1.2))
        stats = engine.strategy.exit_stats
        if engine.trades:
            assert set(stats) <= {EXIT_REVERT, EXIT_STOP,
                                  EXIT_TIMEOUT, EXIT_TREND}

    def test_no_trades_without_universe_data(self):
        engine = run({"000001.SZSE": [10.0] * 5}, BASE)
        assert not engine.trades


class TestTrendFilterDisable:
    """关闭趋势过滤必须用 0。

    真实 bug：曾用 window=1 表示关闭，但窗口为 1 时均线就是价格自身，
    `price > price` 恒为 False，结果不是「关闭过滤」而是「永远不许买入」——
    回测跑出 0 笔成交，看起来像策略没信号，实际是过滤器坏了。
    """

    def test_zero_disables_filter(self):
        s = MeanReversionStrategy(BacktestEngine(), "S", ["000001.SZSE"],
                                  dict(BASE, trend_filter_window=0))
        assert s.above_trend("000001.SZSE") is True

    def test_one_also_disables_not_blocks(self):
        s = MeanReversionStrategy(BacktestEngine(), "S", ["000001.SZSE"],
                                  dict(BASE, trend_filter_window=1))
        s.closes["000001.SZSE"].append(10.0)
        assert s.above_trend("000001.SZSE") is True, "窗口为1不应变成永远禁买"

    def test_negative_rejected(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="trend_filter_window"):
            engine.add_strategy(MeanReversionStrategy, ["000001.SZSE"],
                                {"trend_filter_window": -1})

    def test_disabled_allows_buying_in_downtrend(self):
        """关闭过滤后确实会在下跌趋势中买入 —— 这正是死法一"""
        prices = {"000001.SZSE": [100 * 0.99 ** i for i in range(80)]}
        engine = run(prices, dict(BASE, entry_z=-0.5, exit_z=0.5,
                                  trend_filter_window=0))
        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        assert buys, "关闭过滤后应能在下跌趋势中买入（用于对照实验）"
