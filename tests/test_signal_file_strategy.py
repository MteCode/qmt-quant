"""SignalFileStrategy —— 把离线选股接进实盘引擎的那座桥。

## 它解决什么

项目里长期有两个平行世界：引擎世界（事件驱动、风控、订单状态机齐全，
但只跑演示策略）与脚本世界（真实策略都在，但每个脚本自己搭一套，
跑完就退）。不通的技术原因之一是 `PortfolioStrategy.on_bars` 依赖
`engine.get_universe()`，而实盘引擎没有这个方法。

## 这些用例守什么

- 信号过期不执行（审计发现「下单任务不校验信号新鲜度」）
- 同一份信号不重复调仓（挂单未成交时重算差异会双倍建仓）
- 按信号权重而非等权
- 停牌不下单、碎单不下单
"""
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from qmtquant.core.constants import Direction, Exchange
from qmtquant.core.objects import BarData
from qmtquant.strategy.signal_file import SignalFileStrategy


class _StubEngine:
    """记录下单调用，不连任何东西。"""

    def __init__(self, cash=1_000_000):
        self.orders: list[tuple] = []
        self._cash = cash

    def send_order(self, strategy_name, vt_symbol, direction, price,
                   volume, order_type=None, **kw):
        self.orders.append((vt_symbol, direction, price, volume))
        return "stub"

    def get_cash(self):
        return self._cash

    def get_pos(self, vt_symbol):
        return 0

    def write_log(self, msg, strategy=None):
        pass

    def load_bars(self, *a, **k):
        pass


def _write_signal(path: Path, rows: list[tuple], age_days: float = 0):
    """写信号文件。age_days>0 时把 mtime 往前拨，模拟过期信号。"""
    lines = ["vt_symbol,score,weight,target_value"]
    for sym, w in rows:
        lines.append(f"{sym},0.5,{w},{w * 1_000_000:.2f}")
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    if age_days:
        t = time.time() - age_days * 86400
        os.utime(path, (t, t))


def _bars(symbols, price=10.0, suspended=False):
    out = {}
    for s in symbols:
        code, ex = s.split(".")
        out[s] = BarData(
            symbol=code, exchange=Exchange[ex], datetime=datetime.now(),
            open_price=price, high_price=price, low_price=price,
            close_price=price, volume=10000, turnover=price * 10000)
        out[s].suspended = suspended
    return out


@pytest.fixture
def strat(tmp_path):
    sig = tmp_path / "target_latest.csv"
    eng = _StubEngine()
    syms = ["000001.SZSE", "000002.SZSE", "000003.SZSE"]
    s = SignalFileStrategy(eng, "test", syms, {
        "signal_file": str(sig),
        "cash_buffer": 0.0,
        "min_order_value": 1000.0,
    })
    s.trading = True
    return s, eng, sig


# ------------------------------------------------------------ 新鲜度

def test_过期信号不执行(strat):
    """隔了 5 天的选股结果拿来今天下单，等于用过期判断做决策。"""
    s, eng, sig = strat
    _write_signal(sig, [("000001.SZSE", 0.5)], age_days=5)
    s.on_bars(_bars(["000001.SZSE"]))
    assert eng.orders == []


def test_新鲜信号正常执行(strat):
    s, eng, sig = strat
    _write_signal(sig, [("000001.SZSE", 0.5)])
    s.on_bars(_bars(["000001.SZSE"]))
    assert len(eng.orders) == 1


def test_过期阈值可配(strat):
    s, eng, sig = strat
    s.max_signal_age_days = 10
    _write_signal(sig, [("000001.SZSE", 0.5)], age_days=5)
    s.on_bars(_bars(["000001.SZSE"]))
    assert len(eng.orders) == 1


def test_信号文件不存在时不下单(strat):
    s, eng, sig = strat
    s.on_bars(_bars(["000001.SZSE"]))
    assert eng.orders == []


def test_空信号文件不下单(strat):
    s, eng, sig = strat
    sig.write_text("vt_symbol,score,weight\n", encoding="utf-8-sig")
    s.on_bars(_bars(["000001.SZSE"]))
    assert eng.orders == []


# ------------------------------------------------------------ 幂等

def test_同一份信号不重复调仓(strat):
    """挂单未成交时持仓不变，重算差异会得到同样结果，再下一遍就是双倍建仓。"""
    s, eng, sig = strat
    _write_signal(sig, [("000001.SZSE", 0.5)])
    bars = _bars(["000001.SZSE"])
    s.on_bars(bars)
    n = len(eng.orders)
    s.on_bars(bars)
    s.on_bars(bars)
    assert len(eng.orders) == n, "同一份信号被重复执行"


def test_信号更新后重新调仓(strat):
    s, eng, sig = strat
    _write_signal(sig, [("000001.SZSE", 0.5)])
    s.on_bars(_bars(["000001.SZSE"]))
    n = len(eng.orders)

    time.sleep(1.05)          # mtime 精度到秒，确保标识变化
    _write_signal(sig, [("000002.SZSE", 0.5)])
    s.on_bars(_bars(["000002.SZSE"]))
    assert len(eng.orders) > n


# ------------------------------------------------------------ 权重

def test_按信号权重下单而非等权(strat):
    """权重 0.6/0.2 的两只，下单金额应成 3:1，而不是各一半。"""
    s, eng, sig = strat
    _write_signal(sig, [("000001.SZSE", 0.6), ("000002.SZSE", 0.2)])
    s.on_bars(_bars(["000001.SZSE", "000002.SZSE"], price=10.0))
    amounts = {o[0]: o[2] * o[3] for o in eng.orders}
    assert amounts["000001.SZSE"] == pytest.approx(
        amounts["000002.SZSE"] * 3, rel=0.02)


def test_无weight列时退化为等权(strat):
    s, eng, sig = strat
    sig.write_text("vt_symbol,score\n000001.SZSE,0.5\n000002.SZSE,0.4\n",
                   encoding="utf-8-sig")
    s.on_bars(_bars(["000001.SZSE", "000002.SZSE"]))
    amounts = [o[2] * o[3] for o in eng.orders]
    assert len(amounts) == 2
    assert amounts[0] == pytest.approx(amounts[1], rel=0.02)


# ------------------------------------------------------------ 边界

def test_停牌不下单(strat):
    s, eng, sig = strat
    _write_signal(sig, [("000001.SZSE", 0.5)])
    s.on_bars(_bars(["000001.SZSE"], suspended=True))
    assert eng.orders == []


def test_碎单不下(strat):
    """几百块的委托付的手续费比赚的多。"""
    s, eng, sig = strat
    s.min_order_value = 50_000
    _write_signal(sig, [("000001.SZSE", 0.001)])   # 目标仅 1000 元
    s.on_bars(_bars(["000001.SZSE"]))
    assert eng.orders == []


def test_未启动交易时不下单(strat):
    s, eng, sig = strat
    s.trading = False
    _write_signal(sig, [("000001.SZSE", 0.5)])
    s.on_bars(_bars(["000001.SZSE"]))
    assert eng.orders == []


def test_不依赖engine的get_universe(strat):
    """基类 on_bars 依赖 engine.get_universe()，而实盘引擎没有这个方法 ——
    这正是策略此前接不进引擎的技术原因之一。"""
    s, eng, sig = strat
    assert not hasattr(eng, "get_universe")
    _write_signal(sig, [("000001.SZSE", 0.5)])
    s.on_bars(_bars(["000001.SZSE"]))      # 不应 AttributeError
    assert len(eng.orders) == 1
