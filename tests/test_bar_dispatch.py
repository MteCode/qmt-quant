"""Bar 分发 —— 补上「引擎只有 tick、没有 bar」的缺口。

## 这些用例守什么

LiveEngine 原先只注册 EVENT_TICK，`strategy.on_bars()` 只在预热阶段
被调用一次且 trading=False。后果是所有 bar 驱动的策略
（选股类、日内 GBM）的 on_bars **在实盘永远不会被调用** ——
表现为引擎正常启动、策略正常加载、对账正常通过，然后一笔单都没有，
日志里没有任何异常。这是最难查的那种「正常」。
"""
from datetime import datetime, timedelta

import pytest

from qmtquant.core.constants import Exchange, Interval
from qmtquant.core.objects import BarData, TickData
from qmtquant.engine.bar_dispatch import BarAggregator, SectionDispatcher


def _tick(code, dt, price, vol=1000, tov=0.0):
    return TickData(symbol=code, exchange=Exchange.SZSE, datetime=dt,
                    last_price=price, volume=vol, turnover=tov,
                    gateway_name="T")


def _bar(code, dt, ex=Exchange.SZSE, close=10.0):
    return BarData(symbol=code, exchange=ex, datetime=dt,
                   interval=Interval.MINUTE, open_price=close,
                   high_price=close, low_price=close, close_price=close,
                   volume=100, turnover=1000, gateway_name="T")


# ------------------------------------------------------------ tick 聚合

def test_同一分钟内不收口():
    agg = BarAggregator()
    base = datetime(2026, 9, 8, 10, 0, 0)
    assert agg.update(_tick("000001", base, 10.0)) is None
    assert agg.update(_tick("000001", base.replace(second=30), 10.5)) is None


def test_跨分钟收口上一根():
    agg = BarAggregator()
    base = datetime(2026, 9, 8, 10, 0, 0)
    agg.update(_tick("000001", base, 10.0))
    agg.update(_tick("000001", base.replace(second=30), 10.5))
    done = agg.update(_tick("000001", base + timedelta(minutes=1), 10.2))
    assert done is not None
    assert done.datetime == base
    assert done.open_price == 10.0
    assert done.close_price == 10.5


def test_高低价正确记录():
    agg = BarAggregator()
    base = datetime(2026, 9, 8, 10, 0, 0)
    for i, px in enumerate([10.0, 11.0, 9.0, 10.5]):
        agg.update(_tick("000001", base.replace(second=i * 10), px))
    done = agg.update(_tick("000001", base + timedelta(minutes=1), 10.0))
    assert done.high_price == 11.0
    assert done.low_price == 9.0


def test_成交量取增量而非累计():
    """tick 里的 volume 是当日累计。直接当增量用会大出几个数量级，
    量能类特征（vol_z）会全线失真。"""
    agg = BarAggregator()
    base = datetime(2026, 9, 8, 10, 0, 0)
    agg.update(_tick("000001", base, 10.0, vol=5000))
    agg.update(_tick("000001", base.replace(second=30), 10.0, vol=5800))
    done = agg.update(_tick("000001", base + timedelta(minutes=1),
                            10.0, vol=6000))
    assert done.volume == pytest.approx(800), "应为增量 5800-5000"


def test_多标的互不干扰():
    agg = BarAggregator()
    base = datetime(2026, 9, 8, 10, 0, 0)
    agg.update(_tick("000001", base, 10.0))
    agg.update(_tick("000002", base, 20.0))
    d1 = agg.update(_tick("000001", base + timedelta(minutes=1), 10.5))
    assert d1.symbol == "000001"
    assert d1.close_price == 10.0


def test_无效tick被忽略():
    agg = BarAggregator()
    assert agg.update(_tick("000001", None, 10.0)) is None
    assert agg.update(_tick("000001", datetime.now(), 0)) is None


# ------------------------------------------------------------ 横截面分发

def test_跨分钟触发上一时刻的横截面():
    got = []
    d = SectionDispatcher(lambda dt, sec: got.append((dt, len(sec))))
    t0 = datetime(2026, 9, 8, 10, 0)
    d.add(_bar("000001", t0))
    d.add(_bar("000002", t0))
    assert got == [], "同一时刻还没到齐，不该发"
    d.add(_bar("000001", t0 + timedelta(minutes=1)))
    assert got == [(t0, 2)], "收到下一分钟应触发上一分钟"


def test_达到预期数量立刻触发():
    got = []
    d = SectionDispatcher(lambda dt, sec: got.append(len(sec)), expected=2)
    t0 = datetime(2026, 9, 8, 10, 0)
    d.add(_bar("000001", t0))
    d.add(_bar("000002", t0))
    assert got == [2]


def test_超时强制发出():
    """没有这一步，最后一分钟的 bar 会一直攒着不发 ——
    而收盘前那根往往正是要平仓的那根。"""
    got = []
    d = SectionDispatcher(lambda dt, sec: got.append(len(sec)), timeout=20)
    t0 = datetime(2026, 9, 8, 10, 0)
    d.add(_bar("000001", t0))
    d.check_timeout(t0 + timedelta(minutes=1, seconds=10))
    assert got == [], "还没超时"
    d.check_timeout(t0 + timedelta(minutes=1, seconds=30))
    assert got == [1], "超时后应发出"


def test_迟到的bar不触发重复分发():
    """同一时刻发两次会让策略对同一根 bar 重复决策 —— 可能重复下单。"""
    got = []
    d = SectionDispatcher(lambda dt, sec: got.append((dt, len(sec))))
    t0 = datetime(2026, 9, 8, 10, 0)
    d.add(_bar("000001", t0))
    d.add(_bar("000001", t0 + timedelta(minutes=1)))   # 触发 t0
    assert len(got) == 1
    d.add(_bar("000002", t0))                          # 迟到的 t0
    d.add(_bar("000003", t0 + timedelta(minutes=2)))
    assert [g[0] for g in got].count(t0) == 1, "t0 被重复分发"


def test_乱序时先发更早的时刻():
    got = []
    d = SectionDispatcher(lambda dt, sec: got.append(dt))
    t0 = datetime(2026, 9, 8, 10, 0)
    d.add(_bar("000001", t0))
    d.add(_bar("000001", t0 + timedelta(minutes=1)))
    d.add(_bar("000001", t0 + timedelta(minutes=2)))
    assert got == [t0, t0 + timedelta(minutes=1)]


def test_回调异常不中断分发():
    """一个策略抛异常不该让其他策略收不到行情。"""
    calls = []

    def _boom(dt, sec):
        calls.append(dt)
        raise ValueError("策略炸了")

    d = SectionDispatcher(_boom)
    t0 = datetime(2026, 9, 8, 10, 0)
    d.add(_bar("000001", t0))
    d.add(_bar("000001", t0 + timedelta(minutes=1)))
    d.add(_bar("000001", t0 + timedelta(minutes=2)))
    assert len(calls) == 2, "异常后仍应继续分发后续时刻"


def test_flush发出全部挂起():
    got = []
    d = SectionDispatcher(lambda dt, sec: got.append(dt))
    t0 = datetime(2026, 9, 8, 10, 0)
    d.add(_bar("000001", t0))
    d.flush()
    assert got == [t0]


def test_已发时刻集合不无限增长():
    """跑一整天有 240 个时刻，多日不清理会持续占内存。"""
    d = SectionDispatcher(lambda dt, sec: None)
    t0 = datetime(2026, 9, 8, 9, 30)
    for i in range(900):
        d.add(_bar("000001", t0 + timedelta(minutes=i)))
    assert len(d._done) <= 600
