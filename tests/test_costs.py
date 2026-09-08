"""交易成本模型的测试。

这些测试存在的理由，来自一次审计发现的四类问题：

1. 成本常数散在 8 个文件里各写各的，佣金出现三个版本（万2.5/万3/万0.854）
2. 最低佣金 5 元只有引擎实现了，所有研究脚本都用固定费率
3. 印花税全部写死 0.001，是 2023-08-28 减半之前的旧值
4. 模拟撮合网关的滑点从未施加 —— `traded_price = market_price`，
   而 `CostConfig.slippage_tick` 在那个文件里根本没被引用过

第 4 条能存在这么久，正是因为没有一条测试检查过成交价与市价的关系。
所以这里钉住的是**行为**（方向、分段、单调性），不是具体费率数字 ——
费率会变，行为不该变。
"""
from __future__ import annotations

import pytest

from qmtquant.config import CostConfig
from qmtquant.core.costs import DEFAULT_COST, CostModel
from qmtquant.core.constants import Direction, Exchange
from qmtquant.event.engine import EventEngine
from qmtquant.core.objects import OrderRequest
from qmtquant.gateway.sim_gateway import SimGateway, calc_cost


class TestCostModel:
    def test_buy_has_no_stamp_tax(self):
        m = CostModel()
        amount = 1_000_000.0
        assert m.fee(amount, True) - m.fee(amount, False) == pytest.approx(
            amount * m.stamp_tax_rate)

    def test_min_commission_below_threshold(self):
        m = CostModel()
        thr = m.min_commission_threshold()
        assert m.commission(thr * 0.5) == pytest.approx(m.commission_min)
        assert m.commission(thr * 2) == pytest.approx(
            thr * 2 * m.commission_rate)

    def test_threshold_is_the_crossover(self):
        """临界点两侧的佣金应当连续 —— 差一分钱不该跳变。"""
        m = CostModel()
        thr = m.min_commission_threshold()
        assert m.commission(thr) == pytest.approx(m.commission_min)
        assert m.commission(thr * 1.0001) == pytest.approx(
            m.commission_min, rel=1e-3)

    def test_effective_rate_monotonic(self):
        """单笔越小，实际费率越高。做 T 拆单必然踩在这条曲线上。"""
        m = CostModel()
        amts = [200_000, 100_000, 50_000, 25_000, 10_000, 5_000]
        rates = [m.round_trip_rate(a) for a in amts]
        assert rates == sorted(rates), "往返费率应随单笔金额减小而单调上升"
        assert rates[-1] > rates[0] * 2

    def test_nominal_is_the_large_order_limit(self):
        """名义费率就是金额足够大时的极限。"""
        m = CostModel()
        assert m.round_trip_rate(50_000_000) == pytest.approx(
            m.round_trip_rate_nominal(), rel=1e-4)

    def test_nominal_composition(self):
        """名义往返 = 双边佣金 + 双边过户费 + 单边印花税 + 双边滑点。"""
        m = CostModel()
        assert m.round_trip_rate_nominal() == pytest.approx(
            2 * m.commission_rate + 2 * m.transfer_fee_rate
            + m.stamp_tax_rate + 2 * m.slippage_rate)

    def test_stamp_tax_is_current_rate(self):
        """印花税 2023-08-28 起减半为万 5。

        写死 0.001 会让每一笔卖出的成本高估一倍。这条是刻意钉死数值的
        少数例外 —— 它是法定费率，不该被谁随手改回去。
        """
        assert CostModel().stamp_tax_rate == 0.0005

    def test_zero_and_negative_amounts(self):
        m = CostModel()
        assert m.fee(0, False) == 0.0
        assert m.fee(-100, True) == 0.0
        assert m.round_trip_rate(0) == 0.0

    def test_slippage_from_tick_depends_on_price(self):
        """A 股最小变动价位固定 0.01 元，所以同一个 tick 对低价股更贵。"""
        m = CostModel()
        assert m.slippage_from_tick(10.0) == pytest.approx(0.001)
        assert m.slippage_from_tick(50.0) == pytest.approx(0.0002)
        assert m.slippage_from_tick(10.0) > m.slippage_from_tick(50.0)

    def test_from_config_roundtrips(self):
        cfg = CostConfig()
        m = CostModel.from_config(cfg)
        assert m.commission_rate == cfg.commission_rate
        assert m.commission_min == cfg.commission_min
        assert m.stamp_tax_rate == cfg.stamp_tax_rate
        assert m.transfer_fee_rate == cfg.transfer_fee_rate

    def test_describe_exposes_the_segments(self):
        """summary.json 里只记名义费率会掩盖最低佣金的分段效应。"""
        d = DEFAULT_COST.describe()
        assert "round_trip_by_amount" in d
        assert d["round_trip_by_amount"]["10000"] > d["round_trip_nominal"]
        assert d["min_commission_threshold"] == pytest.approx(58548.01, abs=1)

    def test_config_and_model_agree(self):
        """CostConfig 与 CostModel 的默认值必须一致，否则引擎和研究
        脚本会用不同的成本，而且不会有任何报错。"""
        cfg, m = CostConfig(), CostModel()
        assert cfg.commission_rate == m.commission_rate
        assert cfg.commission_min == m.commission_min
        assert cfg.stamp_tax_rate == m.stamp_tax_rate
        assert cfg.transfer_fee_rate == m.transfer_fee_rate


class TestCalcCostMatchesModel:
    """引擎用的 calc_cost 与研究脚本用的 CostModel.fee 必须给出同一个数。
    两条路径算出不同成本，是回测与实盘对不上的经典来源。"""

    @pytest.mark.parametrize("price,volume", [
        (10.0, 100), (10.0, 10000), (10.8, 2600), (500.0, 100),
    ])
    @pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
    def test_same_as_model(self, price, volume, direction):
        cfg = CostConfig()
        m = CostModel.from_config(cfg)
        engine = calc_cost(price, volume, direction, cfg)
        model = m.fee(price * volume, direction == Direction.SHORT)
        assert engine == pytest.approx(model)


class TestSimGatewaySlippage:
    """成交价必须对自己不利。

    此前 _match 里是 `traded_price = market_price`，零滑点成交，
    模拟盘因此优于任何真实执行 —— 而且没有一条测试发现它。

    这些测试走真实撮合路径（send_order -> _match_all），
    从 on_trade 回报里读成交价，不复算一遍公式 ——
    复算公式的测试只会证明我会算术，不会证明代码做了这件事。
    """

    def _gw(self, tick: int = 1) -> tuple[SimGateway, list]:
        gw = SimGateway(EventEngine(), initial_capital=1_000_000,
                        cost=CostConfig(slippage_tick=tick))
        fills: list = []
        gw.on_trade = fills.append          # 截住成交回报
        gw.on_order = lambda o: None
        gw.on_account = lambda a: None
        return gw, fills

    def _fill(self, tick: int, direction: Direction,
              limit: float, market: float, volume: float = 100):
        gw, fills = self._gw(tick)
        if direction == Direction.SHORT:
            # 卖出要先有可卖持仓，否则会被「可卖数量不足」拒单
            buy = OrderRequest(symbol="600711", exchange=Exchange.SSE,
                               direction=Direction.LONG,
                               price=market * 2, volume=volume)
            gw.send_order(buy)
            gw._match_all("600711.SSE", market)
            gw.settle()                     # 解锁 T+1
            fills.clear()

        req = OrderRequest(symbol="600711", exchange=Exchange.SSE,
                           direction=direction, price=limit, volume=volume)
        gw.send_order(req)
        gw._match_all("600711.SSE", market)
        assert fills, "订单应当成交"
        return fills[-1]

    def test_buy_pays_above_market(self):
        t = self._fill(1, Direction.LONG, limit=10.50, market=10.00)
        assert t.price > 10.00, "买入成交价必须高于市价"
        assert t.price == pytest.approx(10.01)

    def test_sell_receives_below_market(self):
        t = self._fill(1, Direction.SHORT, limit=9.50, market=10.00)
        assert t.price < 10.00, "卖出成交价必须低于市价"
        assert t.price == pytest.approx(9.99)

    def test_slippage_never_crosses_the_limit_price(self):
        """限价单的成交价不会差于委托价 —— 滑点不能把你推过限价。"""
        t = self._fill(5, Direction.LONG, limit=10.02, market=10.00)
        assert t.price == pytest.approx(10.02)

    def test_zero_tick_means_no_slippage(self):
        t = self._fill(0, Direction.LONG, limit=10.50, market=10.00)
        assert t.price == pytest.approx(10.00)

    def test_larger_tick_costs_more(self):
        cheap = self._fill(1, Direction.LONG, limit=99.0, market=10.00)
        dear = self._fill(3, Direction.LONG, limit=99.0, market=10.00)
        assert dear.price > cheap.price

    def test_fill_reports_the_fee(self):
        """成交回报里的 commission 应当就是 calc_cost 的结果。"""
        t = self._fill(1, Direction.LONG, limit=10.50, market=10.00)
        expect = calc_cost(t.price, t.volume, Direction.LONG, CostConfig())
        assert t.commission == pytest.approx(expect)
        assert t.commission >= CostConfig().commission_min
