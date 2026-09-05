"""回测引擎测试：重点验证 A 股规则与无前视偏差。"""
from pathlib import Path
import pandas as pd
import pytest

from qmtquant.config import CostConfig
from qmtquant.core.constants import Direction, Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.backtest_engine import BacktestEngine
from qmtquant.engine.performance import calculate_stats
from qmtquant.strategy.base import StrategyBase


def make_bars(prices: list[float], symbol="000001", exchange=Exchange.SZSE,
              opens: list[float] | None = None) -> list[BarData]:
    """构造日线，open 默认等于当日 close 便于断言"""
    dates = pd.bdate_range("2023-01-02", periods=len(prices))
    bars = []
    for i, (dt, close) in enumerate(zip(dates, prices)):
        o = opens[i] if opens else close
        bars.append(BarData(
            symbol=symbol, exchange=exchange, datetime=dt.to_pydatetime(),
            interval=Interval.DAILY, open_price=o, high_price=max(o, close),
            low_price=min(o, close), close_price=close, volume=1_000_000,
        ))
    return bars


class BuyOnceStrategy(StrategyBase):
    """第一根 Bar 买入 100 股，之后每根都尝试卖出，用于测 T+1"""
    parameters = []

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bar_count = 0
        self.sell_attempts = 0

    def on_bar(self, bar):
        self.bar_count += 1
        if self.bar_count == 1:
            self.buy(bar.vt_symbol, bar.close_price * 1.1, 100)
        elif self.get_pos(bar.vt_symbol) > 0:
            self.sell_attempts += 1
            self.sell(bar.vt_symbol, bar.close_price * 0.9,
                      self.get_pos(bar.vt_symbol))


class BuyHighLimitStrategy(StrategyBase):
    """只在第一根 Bar 买入，限价给足 25% 缓冲。

    用于测涨跌停逻辑：限价必须高于次日开盘价，否则会先被
    「限价低于开盘价」拒掉，测不到涨跌停判断。
    """
    parameters = []

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bar_count = 0

    def on_bar(self, bar):
        self.bar_count += 1
        if self.bar_count == 1:
            self.buy(bar.vt_symbol, bar.close_price * 1.25, 100)


class TestBacktestRules:
    def test_no_lookahead_fill_on_next_open(self):
        """T 日下的单必须在 T+1 开盘成交，不能在 T 日成交"""
        # 第 2 根 open=10.5，与第 1 根 close=10 明显不同，可区分成交发生在哪根
        bars = make_bars([10, 11, 12], opens=[10, 10.5, 12])
        engine = BacktestEngine(initial_capital=100_000)
        engine.load_data(bars)
        engine.add_strategy(BuyOnceStrategy, ["000001.SZSE"])
        engine.run()

        assert len(engine.trades) >= 1
        buy = engine.trades[0]
        # 第一根 bar 下单，成交日期应是第二根 bar 的日期，成交价基于其开盘价 10.5
        assert buy.datetime == bars[1].datetime
        assert buy.price == pytest.approx(10.5, abs=0.05)

    def test_t1_cannot_sell_same_day(self):
        """当日买入当日不可卖"""
        bars = make_bars([10, 10, 10, 10])
        engine = BacktestEngine(initial_capital=100_000)
        engine.load_data(bars)
        engine.add_strategy(BuyOnceStrategy, ["000001.SZSE"])
        engine.run()

        buys = [t for t in engine.trades if t.direction == Direction.LONG]
        sells = [t for t in engine.trades if t.direction == Direction.SHORT]
        assert len(buys) == 1
        # 卖出成交日必须晚于买入成交日
        for s in sells:
            assert s.datetime > buys[0].datetime

    def test_limit_up_blocks_buy(self):
        """开盘一字涨停无法买入"""
        # 第 1 根 close=10，第 2 根 open=11.5（超过 +10%）
        bars = make_bars([10, 11.5], opens=[10, 11.5])
        engine = BacktestEngine(initial_capital=100_000)
        engine.load_data(bars)
        engine.add_strategy(BuyOnceStrategy, ["000001.SZSE"])
        engine.run()

        assert len(engine.trades) == 0
        assert any("涨停" in o.message for o in engine.orders)

    def test_round_lot_enforced(self):
        """买入数量向下取整到 100 股"""
        engine = BacktestEngine()
        vt_orderid = engine.send_order("s", "000001.SZSE", Direction.LONG, 10, 150)
        assert vt_orderid
        assert engine.pending_orders[0].volume == 100

        # 不足 100 股直接不下单
        assert engine.send_order("s", "000001.SZSE", Direction.LONG, 10, 50) == ""

    def test_suspended_bar_not_filled(self):
        bars = make_bars([10, 10])
        bars[1].suspended = True
        engine = BacktestEngine(initial_capital=100_000)
        engine.load_data(bars)
        engine.add_strategy(BuyOnceStrategy, ["000001.SZSE"])
        engine.run()
        assert len(engine.trades) == 0

    def test_st_checker_handles_multiple_spans(self):
        """同一标的可能有多段 ST 区间，判定必须逐段检查。

        实测 300472.SZSE 有两段连续区间（2025-05-06~2026-09-02、
        2026-09-03 至今）；620 只标的最后一段已结束但更早还有区间。
        只看最后一段或只看首段都会错。
        """
        import pandas as pd
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from download_st_history import build_checker

        df = pd.DataFrame([
            {"vt_symbol": "000001.SZSE", "name": "ST甲",
             "start_date": pd.Timestamp("2020-01-01"),
             "end_date": pd.Timestamp("2020-12-31"), "reason": ""},
            {"vt_symbol": "000001.SZSE", "name": "*ST甲",
             "start_date": pd.Timestamp("2023-01-01"),
             "end_date": pd.NaT, "reason": ""},
        ])
        is_st = build_checker(df)

        assert not is_st("000001.SZSE", "2019-06-01")   # 首段之前
        assert is_st("000001.SZSE", "2020-06-01")       # 首段之内
        assert not is_st("000001.SZSE", "2021-06-01")   # 两段之间已摘帽
        assert is_st("000001.SZSE", "2024-06-01")       # 次段之内
        assert is_st("000001.SZSE", "2030-01-01")       # end_date 为空=至今
        assert not is_st("600000.SSE", "2020-06-01")    # 从未 ST

    def test_st_uses_5pct_limit(self, monkeypatch):
        """ST 标的涨跌停 5%，且必须按**当日**状态判定。

        某只股票 2023 年被 ST、2025 年摘帽，按当前状态回测 2023 年
        会用 10% 撮合本该 5% 的标的，让本该拒单的委托成交，
        系统性高估可成交性。
        """
        import pandas as pd

        engine = BacktestEngine()
        # 只在 2023 年是 ST
        monkeypatch.setattr(
            engine, "_is_st",
            lambda vt, dt: pd.Timestamp(dt).year == 2023)

        d2023 = pd.Timestamp("2023-06-01")
        d2025 = pd.Timestamp("2025-06-01")
        assert engine._limit_ratio("600000.SSE", d2023) == 0.05
        assert engine._limit_ratio("600000.SSE", d2025) == 0.10
        # 不传日期时退回按板块判定，不应误用 5%
        assert engine._limit_ratio("600000.SSE") == 0.10
        # 创业板非 ST 日仍是 20%
        assert engine._limit_ratio("300750.SZSE", d2025) == 0.20

    def test_lot_rounding_uses_real_price(self, monkeypatch):
        """整手取整必须按真实股数，而非被抬高的后复权价。

        后复权价比真实价高（茅台约 5.4 倍），一手的名义成本同比例放大。
        直接按后复权价取整会把高价股静默剔出标的池 ——
        实测 100 万/10 只时沪深300 有 36 只取整后为 0 股，
        而它们往往正是大盘蓝筹。
        """
        engine = BacktestEngine(initial_capital=100_000)
        # 后复权价 540、真实价 100 -> 因子 5.4
        monkeypatch.setattr(engine, "_adj_factor", lambda vt: 5.4)

        # 预算够买 50 个后复权单位；真实股数 = 50 x 5.4 = 270 -> 取整 200
        vt_orderid = engine.send_order(
            "s", "600519.SSE", Direction.LONG, 540, 50)
        assert vt_orderid, "按真实价可买 200 股，不应被判为不足一手"
        # 记录的是后复权口径的数量：200 / 5.4
        assert engine.pending_orders[0].volume == pytest.approx(200 / 5.4)

        # 无复权因子时退回旧行为：50 直接取整到 0，下不出去
        engine2 = BacktestEngine(initial_capital=100_000)
        monkeypatch.setattr(engine2, "_adj_factor", lambda vt: None)
        assert engine2.send_order(
            "s", "600519.SSE", Direction.LONG, 540, 50) == ""
        assert engine2.undersized_orders.get("600519.SSE") == 1

    def test_missing_bar_not_filled(self):
        """停牌在真实数据里表现为 **Bar 缺失**，而非 suspended 标志。

        实测 QMT 的 suspendFlag 在 760 万行日线里恒为 0，
        停牌日直接不返回 Bar —— 200 只在市股票中 178 只有日期缺口，
        合计 19634 个停牌日。因此这条路径才是实际生效的那条，
        必须单独固化，不能只测 suspended 标志。
        """
        # 只有一根 Bar：策略在这根上下买单，但没有下一根可供撮合。
        # 委托按次日开盘成交，缺失下一根 Bar 即等同停牌
        bars = make_bars([10])

        engine = BacktestEngine(initial_capital=100_000)
        engine.load_data(bars)
        engine.add_strategy(BuyOnceStrategy, ["000001.SZSE"])
        engine.run()
        assert len(engine.trades) == 0
        # 委托应仍挂在未成交队列里，而不是被静默丢弃
        assert len(engine.pending_orders) == 1

    def test_commission_deducted(self):
        bars = make_bars([10, 10, 10])
        engine = BacktestEngine(initial_capital=100_000, cost=CostConfig())
        engine.load_data(bars)
        engine.add_strategy(BuyOnceStrategy, ["000001.SZSE"])
        engine.run()
        assert engine.trades[0].commission > 0
        # 买入再卖出，价格不变，净结果必然亏掉双边手续费
        total_fee = sum(t.commission for t in engine.trades)
        assert engine.cash == pytest.approx(100_000 - total_fee, abs=2.0)
        assert engine.cash < 100_000


class TestPerformance:
    def test_stats_on_flat_curve(self):
        eq = pd.Series([100.0] * 10,
                       index=pd.bdate_range("2023-01-02", periods=10))
        stats = calculate_stats(eq, [], 100.0)
        assert stats.total_return == pytest.approx(0)
        assert stats.max_drawdown == pytest.approx(0)
        assert stats.trading_days == 10

    def test_stats_drawdown(self):
        eq = pd.Series([100, 120, 90, 110],
                       index=pd.bdate_range("2023-01-02", periods=4), dtype=float)
        stats = calculate_stats(eq, [], 100.0)
        # 从 120 跌到 90 = -25%
        assert stats.max_drawdown == pytest.approx(-0.25)
        assert stats.total_return == pytest.approx(0.10)

    def test_empty_equity(self):
        stats = calculate_stats(pd.Series(dtype=float), [], 100.0)
        assert stats.trading_days == 0


class TestPriceLimitByBoard:
    """涨跌停必须按板块区分。

    真实 bug：引擎曾硬编码全局 10%，而沪深300 中 53 只（18%）是 20% 的
    创业板/科创板，它们涨跌超 10% 的交易日被误判为涨跌停而拒单。
    """

    def _run(self, symbol, exchange, closes, opens, ratio=None):
        engine = BacktestEngine(initial_capital=1_000_000,
                                price_limit_ratio=ratio)
        engine.load_data(make_bars(closes, symbol=symbol,
                                   exchange=exchange, opens=opens))
        # 限价必须高于次日开盘，否则会先被「限价低于开盘价」拒掉，
        # 测不到涨跌停判断本身
        engine.add_strategy(BuyHighLimitStrategy, [f"{symbol}.{exchange.value}"])
        engine.run()
        return engine

    def test_chinext_allows_15pct_gap(self):
        """创业板 20% 涨跌停，次日开盘涨 15% 应可成交"""
        engine = self._run("300750", Exchange.SZSE,
                           closes=[10.0, 11.5], opens=[10.0, 11.5])
        assert engine.trades, "创业板涨 15% 未触及 20% 涨停，应能成交"

    def test_main_board_blocks_15pct_gap(self):
        """主板 10% 涨跌停，同样涨 15% 必须被拒"""
        engine = self._run("600000", Exchange.SSE,
                           closes=[10.0, 11.5], opens=[10.0, 11.5])
        assert not engine.trades
        assert any("涨停" in o.message for o in engine.orders)

    def test_star_market_uses_20pct(self):
        engine = self._run("688981", Exchange.SSE,
                           closes=[10.0, 11.5], opens=[10.0, 11.5])
        assert engine.trades, "科创板同样是 20% 涨跌停"

    def test_explicit_ratio_overrides_board(self):
        """显式指定时压制板块判定，用于对照实验"""
        engine = self._run("300750", Exchange.SZSE,
                           closes=[10.0, 11.5], opens=[10.0, 11.5], ratio=0.10)
        assert not engine.trades, "显式传 10% 时创业板也应按 10% 判定"

    def test_limit_ratio_lookup(self):
        engine = BacktestEngine()
        assert engine._limit_ratio("600519.SSE") == 0.10
        assert engine._limit_ratio("300750.SZSE") == 0.20
        assert engine._limit_ratio("688981.SSE") == 0.20
        assert engine._limit_ratio("830799.BSE") == 0.30
