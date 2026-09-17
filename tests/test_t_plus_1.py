"""T+1 卖出闸门。

## 起因

实盘上这个策略每分钟发卖单、每分钟被券商以「可卖数量不足」拒掉，
包括一笔 -2.40% 的止损单 —— 日志刷满拒单记录，实际一股也没卖出去。

根因是 A 股 T+1：当日买入的股票当日不可卖，而策略按**持仓量**下卖单，
没有区分「持有」和「可卖」。声称的「尾盘强平、不留隔夜」从来没做到过。

## 测什么

sellable() 是所有卖出路径（止损 / 信号 / 尾盘）的唯一闸门，
所以只要它对，三条路径就都对。这里测它的边界。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategies.intraday_gbm.strategy import IntradayGBMStrategy  # noqa: E402


class FakeEngine:
    """只实现策略用到的那几个查询接口。"""

    def __init__(self, pos=None, available=None, cost=None):
        self._pos = pos or {}
        self._avail = available or {}
        self._cost = cost or {}
        self.orders = []

    def get_pos(self, vt):
        return self._pos.get(vt, 0.0)

    def get_available(self, vt):
        return self._avail.get(vt, 0.0)

    def get_cost_price(self, vt):
        return self._cost.get(vt, 0.0)

    def get_cash(self):
        return 1e9

    def send_order(self, *a, **kw):
        self.orders.append((a, kw))
        return "id"


class EngineWithoutT1(FakeEngine):
    """模拟撮合 / T+0 品种：引擎不实现可卖量查询。"""

    get_available = None
    get_cost_price = None


def _make(engine, **setting):
    s = IntradayGBMStrategy(engine, "T", ["600000.SSE"], setting)
    s.pos = dict(engine._pos)
    return s


class TestSellable:
    def test_today_buy_is_not_sellable(self):
        """当日买入：持仓 1000，可卖 0 —— 一股都不能卖。"""
        eng = FakeEngine(pos={"600000.SSE": 1000}, available={"600000.SSE": 0})
        s = _make(eng)
        assert s.sellable("600000.SSE") == 0

    def test_yesterday_position_is_sellable(self):
        eng = FakeEngine(pos={"600000.SSE": 1000},
                         available={"600000.SSE": 1000})
        s = _make(eng)
        assert s.sellable("600000.SSE") == 1000

    def test_partial_available_caps_at_available(self):
        """昨仓 400 + 今日买入 600：只能卖 400。"""
        eng = FakeEngine(pos={"600000.SSE": 1000},
                         available={"600000.SSE": 400})
        s = _make(eng)
        assert s.sellable("600000.SSE") == 400

    def test_never_exceeds_position(self):
        """券商可卖量大于策略自记持仓时，以持仓为准，不能超卖。"""
        eng = FakeEngine(pos={"600000.SSE": 300},
                         available={"600000.SSE": 1000})
        s = _make(eng)
        assert s.sellable("600000.SSE") == 300

    def test_no_position_returns_zero(self):
        eng = FakeEngine(pos={}, available={"600000.SSE": 500})
        s = _make(eng)
        assert s.sellable("600000.SSE") == 0

    def test_t_plus_1_off_ignores_available(self):
        """ETF / 可转债是 T+0，关掉开关后按持仓量卖。"""
        eng = FakeEngine(pos={"600000.SSE": 1000},
                         available={"600000.SSE": 0})
        s = _make(eng, t_plus_1=False)
        assert s.sellable("600000.SSE") == 1000

    def test_engine_without_available_degrades_to_position(self):
        """引擎没实现 get_available 时退化为持仓量，而不是静默变成 0。

        退化成 0 会让策略在模拟撮合里一单都卖不出去，且不报错 ——
        又一个「看起来正常其实什么都没做」的失效形态。
        """
        eng = EngineWithoutT1(pos={"600000.SSE": 1000})
        s = _make(eng)
        assert s.sellable("600000.SSE") == 1000


class TestStopRefPrice:
    def test_prefers_broker_cost_price(self):
        eng = FakeEngine(pos={"600000.SSE": 100}, cost={"600000.SSE": 12.5})
        s = _make(eng)
        s.entry_prices["600000.SSE"] = 99.0
        assert s.stop_ref_price("600000.SSE") == 12.5

    def test_falls_back_to_entry_price(self):
        """券商成本价取不到时用自记入场价。"""
        eng = FakeEngine(pos={"600000.SSE": 100})
        s = _make(eng)
        s.entry_prices["600000.SSE"] = 10.0
        assert s.stop_ref_price("600000.SSE") == 10.0

    def test_survives_restart_via_cost_price(self):
        """重启后 entry_prices 为空，隔夜仓仍要有止损参考 ——
        这正是 on_start 不再清空、且优先用成本价的原因。"""
        eng = FakeEngine(pos={"600000.SSE": 100}, cost={"600000.SSE": 20.0})
        s = _make(eng)
        s.on_start()
        assert s.stop_ref_price("600000.SSE") == 20.0

    def test_no_reference_returns_zero(self):
        eng = FakeEngine(pos={"600000.SSE": 100})
        s = _make(eng)
        assert s.stop_ref_price("600000.SSE") == 0.0
