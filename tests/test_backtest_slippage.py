"""回测引擎的滑点测试。

## 为什么单开一个文件

回测引擎的滑点**从来没有任何测试覆盖**。把 `slip` 直接改成 0，
832 条测试全部通过 —— 也就是说「成交价等于开盘价」和「成交价含滑点」
在测试眼里是同一件事。这个盲区让一个真实的错误活了很久：

    slip = self.cost.slippage_tick * 0.01

本地行情是**后复权价**，而且锚在 IPO，复权因子差异极大：

    平安银行  真实 11.91 元 / 后复权 1241.60 元 / 因子 104
    盛屯矿业  真实 11.79 元 / 后复权  114.18 元 / 因子 9.7
    贵州茅台  真实 1326 元  / 后复权 8288 元    / 因子 6.2
    工商银行  真实 8.15 元  / 后复权   13.18 元 / 因子 1.6

把 0.01 元加到后复权价上，对平安银行只滑了真实价的百分之一个 tick。
低估倍数因股而异，所以不同标的的回测成本互相不可比 ——
这比「统一低估」更难发现，因为总量看着不离谱。

## 为什么改成相对滑点而不是修因子

按 tick 换算需要真实价，而真实价要从 turnover/volume 推。
但 volume 的单位（股还是手）在 BarData 契约里没有规定：
仓库的 parquet 用手，测试 fixture 用股。而且恰恰在因子最大的标的上，
两种解释都能落进「合理」区间 —— 平安银行按股算因子 1.04、按手算 104，
两个都说得通，猜不出来。

相对滑点绕开了整件事：同一个比例施加在后复权价和真实价上效果相同。

模拟撮合网关（sim_gateway）仍按 tick，因为它拿到的是实时行情的真实价，
不存在复权问题，按 tick 更贴近真实盘口。这个区别是有意的。
"""
from __future__ import annotations

import pandas as pd
import pytest

from qmtquant.config import CostConfig
from qmtquant.core.constants import Direction, Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.backtest_engine import BacktestEngine
from qmtquant.strategy.ma_cross import MaCrossStrategy

SYMBOL = "600519.SSE"
SETTING = {"fast_window": 3, "slow_window": 5}


def _v_shape(down: int = 8, up: int = 14, down_step: float = 0.03,
             up_step: float = 0.02, start: float = 20.0) -> list[float]:
    prices = [start * (1 - down_step) ** i for i in range(down)]
    prices += [prices[-1] * (1 + up_step) ** i for i in range(1, up + 1)]
    return prices


def _bars(closes: list[float], scale: float = 1.0) -> list[BarData]:
    """scale 模拟复权因子：把价格整体放大，真实价不变。"""
    dates = pd.bdate_range("2023-01-02", periods=len(closes))
    return [BarData(symbol="600519", exchange=Exchange.SSE,
                    datetime=d.to_pydatetime(), interval=Interval.DAILY,
                    open_price=c * scale, high_price=c * scale,
                    low_price=c * scale, close_price=c * scale,
                    volume=1_000_000, turnover=c * 1_000_000)
            for d, c in zip(dates, closes)]


def _run(closes, cost: CostConfig | None = None, scale: float = 1.0):
    engine = BacktestEngine(initial_capital=1_000_000, cost=cost)
    engine.load_data(_bars(closes, scale))
    engine.add_strategy(MaCrossStrategy, [SYMBOL], dict(SETTING))
    engine.run()
    return engine


def _first_buy(engine):
    buys = [t for t in engine.trades if t.direction == Direction.LONG]
    assert buys, "测试前提：这组数据应当产生买入"
    return buys[0]


class TestSlippageIsApplied:
    """成交价必须对自己不利。这几条是上面那个盲区的直接补位。"""

    def test_buy_pays_above_open(self):
        e = _run(_v_shape(), CostConfig(slippage_rate=0.001))
        t = _first_buy(e)
        bar = next(b for b in _bars(_v_shape())
                   if b.datetime.date() == t.datetime.date())
        assert t.price > bar.open_price, "买入成交价必须高于开盘价"

    def test_zero_rate_means_fill_at_open(self):
        e = _run(_v_shape(), CostConfig(slippage_rate=0.0))
        t = _first_buy(e)
        bar = next(b for b in _bars(_v_shape())
                   if b.datetime.date() == t.datetime.date())
        assert t.price == pytest.approx(bar.open_price)

    def test_larger_rate_costs_more(self):
        cheap = _first_buy(_run(_v_shape(), CostConfig(slippage_rate=0.0005)))
        dear = _first_buy(_run(_v_shape(), CostConfig(slippage_rate=0.005)))
        assert dear.price > cheap.price

    def test_slippage_is_proportional(self):
        """滑点是相对量：费率翻十倍，偏离开盘价的幅度也翻十倍。"""
        base = _first_buy(_run(_v_shape(), CostConfig(slippage_rate=0.0)))
        one = _first_buy(_run(_v_shape(), CostConfig(slippage_rate=0.001)))
        ten = _first_buy(_run(_v_shape(), CostConfig(slippage_rate=0.01)))
        d1, d10 = one.price - base.price, ten.price - base.price
        assert d10 == pytest.approx(d1 * 10, rel=1e-6)


class TestAdjustmentInvariance:
    """同一只票，后复权因子不同不应改变滑点的**相对**大小。

    这是相对滑点相对于绝对 tick 的核心优势，也是原实现最大的问题：
    按 tick 算时，因子 104 的平安银行和因子 1.6 的工商银行，
    实际承担的滑点差 65 倍，而两者的真实盘口都是一个 tick。
    """

    @pytest.mark.parametrize("scale", [1.0, 6.2, 9.7, 104.0])
    def test_relative_slippage_survives_adjustment(self, scale):
        cost = CostConfig(slippage_rate=0.001)
        base = _first_buy(_run(_v_shape(), CostConfig(slippage_rate=0.0),
                               scale=scale))
        slipped = _first_buy(_run(_v_shape(), cost, scale=scale))
        rel = slipped.price / base.price - 1
        assert rel == pytest.approx(0.001, rel=1e-6), \
            f"复权因子 {scale} 下相对滑点应恒为 0.1%，实际 {rel:.6%}"

    def test_absolute_tick_would_not_survive(self):
        """反证：如果按绝对 tick 算，复权因子会把相对滑点冲垮。

        这条不测生产代码，只把「为什么不能用 tick」钉成可执行的事实 ——
        否则以后有人看到 CostConfig 里还留着 slippage_tick，
        很容易顺手把引擎改回去。
        """
        tick = 0.01
        real_price = 11.91
        for factor, expect_rel in ((1.0, tick / real_price),
                                   (104.0, tick / (real_price * 104))):
            rel = tick / (real_price * factor)
            assert rel == pytest.approx(expect_rel)
        # 因子 104 时，一个 tick 的相对大小只有真实盘口的 1/104
        assert (tick / (real_price * 104)) < (tick / real_price) / 100


class TestCostConfigContract:
    def test_engine_uses_rate_not_tick(self):
        """引擎必须读 slippage_rate。改回 slippage_tick 这条会红。"""
        e1 = _first_buy(_run(_v_shape(),
                             CostConfig(slippage_rate=0.0, slippage_tick=99)))
        e2 = _first_buy(_run(_v_shape(),
                             CostConfig(slippage_rate=0.0, slippage_tick=0)))
        assert e1.price == pytest.approx(e2.price), \
            "引擎的成交价不该受 slippage_tick 影响 —— 那是模拟网关用的"

    def test_gateway_still_uses_tick(self):
        """模拟网关按 tick，因为它拿到的是实时真实价。这是有意的区分。"""
        import inspect

        from qmtquant.gateway import sim_gateway
        src = inspect.getsource(sim_gateway.SimGateway._match)
        assert "slippage_tick" in src, \
            "模拟网关应当按 tick 施加滑点（实时行情是真实价）"
