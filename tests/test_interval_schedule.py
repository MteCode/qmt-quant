"""间隔重复型计划 —— 盘中反复执行的调度。

## 为什么需要

原调度只支持「每日一次」，按 last_run_date 判重。持仓快照因此只在
盘前盘后各跑一次，**盘中页面上看到的永远是上一次快照的持仓**，
而人会拿它当实时数据看 —— 这比没有数据更危险。

## 判重方式完全不同

每日一次按**日期**判重；间隔重复必须按**时间戳**判重 ——
只看日期的话，当天第一次跑完就再也不会触发了。
"""
from datetime import datetime, timedelta

import pytest

from webui import scheduler as sch


def _row(**kw):
    base = {
        "id": "t", "name": "t", "task_id": "snapshot_positions",
        "time": "09:30", "end_time": "15:00", "interval_minutes": 5,
        "weekdays": [0, 1, 2, 3, 4], "enabled": True,
        "skip_lunch": True, "last_run_at": "",
    }
    base.update(kw)
    return base


def _weekday_at(hh, mm):
    """取一个工作日的指定时刻，避免用例受运行当天是周几影响。"""
    d = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ------------------------------------------------------------ 时间窗

def test_窗口内且从未跑过则触发():
    assert sch._interval_due(_row(), _weekday_at(10, 0)) is True


def test_开盘前不触发():
    assert sch._interval_due(_row(), _weekday_at(9, 0)) is False


def test_收盘后不触发():
    assert sch._interval_due(_row(), _weekday_at(15, 30)) is False


def test_边界时刻包含在内():
    assert sch._interval_due(_row(), _weekday_at(9, 30)) is True
    assert sch._interval_due(_row(), _weekday_at(15, 0)) is True


def test_周末不触发():
    sat = datetime.now()
    while sat.weekday() != 5:
        sat += timedelta(days=1)
    assert sch._interval_due(_row(), sat.replace(hour=10, minute=0)) is False


# ------------------------------------------------------------ 午休

def test_午休跳过():
    """A 股午休持仓不变，照常轮询只是白白打扰 QMT。"""
    assert sch._interval_due(_row(), _weekday_at(12, 0)) is False


def test_午休边界():
    assert sch._interval_due(_row(), _weekday_at(11, 29)) is True
    assert sch._interval_due(_row(), _weekday_at(11, 30)) is False
    assert sch._interval_due(_row(), _weekday_at(12, 59)) is False
    assert sch._interval_due(_row(), _weekday_at(13, 0)) is True


def test_可关闭午休跳过():
    r = _row(skip_lunch=False)
    assert sch._interval_due(r, _weekday_at(12, 0)) is True


# ------------------------------------------------------------ 间隔判重

def test_间隔未到不重复触发():
    now = _weekday_at(10, 0)
    r = _row(last_run_at=(now - timedelta(minutes=2)).isoformat())
    assert sch._interval_due(r, now) is False


def test_间隔已到再次触发():
    now = _weekday_at(10, 0)
    r = _row(last_run_at=(now - timedelta(minutes=6)).isoformat())
    assert sch._interval_due(r, now) is True


def test_恰好到点触发():
    now = _weekday_at(10, 0)
    r = _row(last_run_at=(now - timedelta(minutes=5)).isoformat())
    assert sch._interval_due(r, now) is True


def test_按时间戳而非日期判重():
    """这是与「每日一次」的关键区别 —— 只看日期的话，
    当天第一次跑完就再也不会触发了。"""
    now = _weekday_at(14, 0)
    r = _row(last_run_at=(now - timedelta(minutes=30)).isoformat(),
             last_run_date=now.date().isoformat())
    assert sch._interval_due(r, now) is True, "同一天内应继续触发"


def test_时间戳损坏时保守触发():
    """宁可多跑一次，也不要因为脏数据永久停摆。"""
    r = _row(last_run_at="not-a-timestamp")
    assert sch._interval_due(r, _weekday_at(10, 0)) is True


# ------------------------------------------------------------ 配置校验

def test_间隔为零不走间隔逻辑():
    """每日一次的计划不该被间隔逻辑接管。"""
    assert sch._interval_due(_row(interval_minutes=0),
                             _weekday_at(10, 0)) is False


def test_缺结束时间不触发():
    """配置不全时宁可不跑，也不要在未知窗口里反复执行。"""
    assert sch._interval_due(_row(end_time=""), _weekday_at(10, 0)) is False


# ------------------------------------------------------------ 预设

def test_盘中快照预设存在且合理():
    p = next((s for s in sch.PRESETS if s.id == "intraday_snapshot"), None)
    assert p is not None, "缺少盘中刷新持仓的预设"
    assert p.task_id == "snapshot_positions"
    assert p.interval_minutes > 0
    assert sch._hhmm(p.time) < sch._hhmm(p.end_time)
    assert p.enabled is True, "只读任务，默认开启是安全的"


def test_下次触发说明可读():
    r = _row(last_run_at=(_weekday_at(10, 0)
                          - timedelta(minutes=2)).isoformat())
    hint = sch._next_hint(r)
    assert isinstance(hint, str) and hint, "说明文字不该为空"
