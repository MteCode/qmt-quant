"""日内做 T 的档位与约束。

## 为什么重点测这些

做 T 的两个致命错误都不会报错，只会静默做错事：

1. **卖超可卖量** —— 当日买回的部分 T+1 冻结。实盘上忽略这点的表现是
   每分钟发单、每分钟被券商以「可卖数量不足」拒掉，日志刷满而一股没卖出
2. **档位记错** —— 卖了 2 档却记成 1 档，尾盘就只买回 1 档，净持仓悄悄
   少了一截，第二天才发现底仓被做没了

所以这里测的不是"能不能跑"，是这两件事在边界上对不对。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qmtquant.core.objects import BarData  # noqa: E402
from strategies.intraday_t0.strategy import IntradayT0Strategy  # noqa: E402

SYM = "600000.SSE"


class FakeEngine:
    def __init__(self, available=0.0, cash=1e7):
        self._avail = available
        self._cash = cash
        self.orders = []

    def get_pos(self, vt):
        return 0.0

    def get_available(self, vt):
        return self._avail

    def get_cost_price(self, vt):
        return 0.0

    def get_cash(self):
        return self._cash

    def send_order(self, strategy_name, vt_symbol, direction, price,
                   volume, order_type=None):
        """签名必须与 StrategyBase._send_order 的调用完全一致。

        第一版我多加了个 offset 参数，导致参数整体错位 —— direction 收到的
        其实是 vt_symbol。后果是所有「断言没有卖单」的测试恒真通过，
        测了个寂寞。测试替身的签名对不上，比没有测试更危险。
        """
        self.orders.append({"vt": vt_symbol, "dir": str(direction),
                            "price": price, "volume": volume})
        return "oid"


def _bar(price, vol=10000.0, t="10:00"):
    hh, mm = t.split(":")
    b = BarData(
        symbol="600000", exchange=None, datetime=datetime(2026, 9, 18,
                                                          int(hh), int(mm)),
        gateway_name="test",
    )
    b.close_price = price
    b.open_price = price
    b.high_price = price
    b.low_price = price
    b.volume = vol
    b.turnover = price * vol
    return b


def _make(engine, **setting):
    s = IntradayT0Strategy(engine, "T0", [SYM], setting)
    s.trading = True
    s.on_start()
    return s


def _feed(s, prices, t="10:00"):
    """按顺序喂 bar，返回最后一次的委托列表。"""
    for p in prices:
        s.on_bars({SYM: _bar(p, t=t)})


class TestLayering:
    def test_no_action_near_vwap(self):
        """价格贴着 VWAP 时不该动作 —— 没有偏离就没有做 T 空间。"""
        eng = FakeEngine(available=10000)
        s = _make(eng, grid=0.008, max_layers=3)
        _feed(s, [10.0, 10.0, 10.0])
        assert eng.orders == []

    def test_sells_one_layer_on_first_grid(self):
        """涨过一格卖一档。"""
        eng = FakeEngine(available=10000)
        s = _make(eng, grid=0.01, max_layers=3, trade_value=30000)
        _feed(s, [10.0, 10.0, 10.3])      # VWAP≈10.1, dev≈+2% -> 2 档
        assert eng.orders, "应当有卖出"
        assert s.layers[SYM] >= 1

    def test_layers_capped_at_max(self):
        """涨再多也不超过 max_layers —— 否则会把底仓全卖光。"""
        eng = FakeEngine(available=1e6)
        s = _make(eng, grid=0.005, max_layers=2, trade_value=20000)
        _feed(s, [10.0, 12.0, 15.0, 20.0])
        assert s.layers[SYM] <= 2

    def test_buys_back_when_price_returns(self):
        """价格回到 VWAP 附近，卖出的档位要买回来。"""
        eng = FakeEngine(available=1e6)
        s = _make(eng, grid=0.01, max_layers=3, trade_value=30000)
        _feed(s, [10.0, 10.0, 10.5])       # 先卖
        sold = s.layers[SYM]
        assert sold >= 1
        _feed(s, [9.6, 9.5])               # 价格回落
        assert s.layers[SYM] < sold, "回落后应当买回"

    def test_buys_below_vwap(self):
        """跌破 VWAP 要加买 —— 这是对称版的核心，旧版把档位压成非负，
        只做「高抛-回补」不做「低吸-高抛」，结果损益与当日涨跌幅正相关
        0.40，单边上涨日靠底仓错配赚钱、震荡日反复付成本。"""
        eng = FakeEngine(available=1e6)
        s = _make(eng, grid=0.01, max_layers=3, trade_value=30000)
        _feed(s, [10.0, 10.0, 9.5])        # 跌破 VWAP
        assert s.layers[SYM] < 0, "低于 VWAP 应当加买（档位为负）"
        buys = [o for o in eng.orders if "LONG" in o["dir"].upper()]
        assert buys, "应当发出买单"

    def test_sells_back_when_price_recovers(self):
        """低位加买后价格回升，要把加买的卖掉回到底仓。"""
        eng = FakeEngine(available=1e6)
        s = _make(eng, grid=0.01, max_layers=3, trade_value=30000)
        _feed(s, [10.0, 10.0, 9.5])
        assert s.layers[SYM] < 0
        _feed(s, [10.4, 10.6])
        assert s.layers[SYM] >= 0, "回升后应当卖出加买的档位"

    def test_layers_symmetric_bounds(self):
        """档位上下界对称，都不超过 max_layers。"""
        eng = FakeEngine(available=1e6)
        s = _make(eng, grid=0.005, max_layers=2, trade_value=20000)
        _feed(s, [10.0, 6.0, 5.0, 4.0])
        assert s.layers[SYM] >= -2

    def test_closing_flattens_both_directions(self):
        """尾盘两个方向都要平回 0 档。"""
        eng = FakeEngine(available=1e6)
        s = _make(eng, grid=0.01, max_layers=3, trade_value=30000,
                  exit_time="14:45")
        _feed(s, [10.0, 10.0, 9.4])
        assert s.layers[SYM] < 0
        _feed(s, [9.4], t="14:50")
        assert s.layers[SYM] == 0

    def test_closing_buys_back_all(self):
        """尾盘必须把卖掉的全买回，否则净持仓少一截。"""
        eng = FakeEngine(available=1e6)
        s = _make(eng, grid=0.01, max_layers=3, trade_value=30000,
                  exit_time="14:45")
        _feed(s, [10.0, 10.0, 10.5])
        assert s.layers[SYM] >= 1
        _feed(s, [10.5], t="14:50")        # 过了 exit_time
        assert s.layers[SYM] == 0


class TestSellConstraint:
    def test_never_sells_more_than_available(self):
        """可卖量 0（全是当日买入）时一股都不能卖。"""
        eng = FakeEngine(available=0)
        s = _make(eng, grid=0.005, max_layers=3, trade_value=30000)
        _feed(s, [10.0, 10.0, 11.0])
        sells = [o for o in eng.orders if "SHORT" in o["dir"].upper()
                 or "卖" in o["dir"]]
        assert not sells, "无可卖量时不该发卖单"

    def test_caps_sell_at_available(self):
        """可卖量小于想卖的量时，按可卖量卖且取整到 100 股。"""
        eng = FakeEngine(available=350)
        s = _make(eng, grid=0.005, max_layers=3, trade_value=300000)
        _feed(s, [10.0, 10.0, 11.0])
        for o in eng.orders:
            assert o["volume"] <= 350
            assert o["volume"] % 100 == 0


class TestLimitGuard:
    def test_does_not_buy_at_limit_up(self):
        eng = FakeEngine(available=1e6)
        s = _make(eng, grid=0.01, max_layers=3,
                  limit_prices={SYM: (11.0, 9.0)})
        _feed(s, [10.0, 10.0, 10.5])
        eng.orders.clear()
        _feed(s, [11.0], t="14:50")        # 尾盘回补，但已涨停
        assert eng.orders == [], "涨停不该发买单"

    def test_does_not_sell_at_limit_down(self):
        eng = FakeEngine(available=1e6)
        s = _make(eng, grid=0.005, max_layers=3,
                  limit_prices={SYM: (11.0, 9.0)})
        # 跌停价必须等于将要成交的价格才算触及。上一版设成 10.9 却喂 11.0，
        # 根本没跌停，测的是「不该拦的没拦」，白白通过
        s.limit_prices[SYM] = (12.0, 11.0)
        _feed(s, [10.0, 10.0, 11.0])       # 价高于 VWAP 想卖，但已跌停
        sells = [o for o in eng.orders if "SHORT" in o["dir"].upper()]
        assert not sells, "跌停不该发卖单"


class TestDayReset:
    def test_layers_reset_across_days(self):
        """换日必须清零档位 —— 隔夜后昨仓构成已变，沿用会算错买回量。"""
        eng = FakeEngine(available=1e6)
        s = _make(eng, grid=0.01, max_layers=3, trade_value=30000)
        _feed(s, [10.0, 10.0, 10.5])
        assert s.layers.get(SYM, 0) >= 1
        nxt = _bar(10.0)
        nxt.datetime = datetime(2026, 9, 19, 10, 0)
        s.on_bars({SYM: nxt})
        assert s.layers.get(SYM, 0) == 0
