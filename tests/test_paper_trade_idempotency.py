"""paper_trade 的下单幂等与执行记录。

## 这些用例守的是什么

调仓差异是按「券商持仓 vs 目标持仓」算的，而挂单未成交时持仓不变。
09:31 买了 20 只还没成交，任何原因再跑一次（手点 webui、
scheduled_trade --now、超时重试）会算出同样的差异，再下一遍 ——
**直接双倍建仓**。

更糟的是原实现把执行记录写在所有委托发出之后，而 scheduled_trade 给
subprocess 的超时只有 120 秒：超时被 kill 时委托已经发出去了，
执行记录一行都没写 —— 事后既查不到下过什么单，重跑还会再下一遍。
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

import pytest

S = Path(__file__).resolve().parents[1] / "strategies" / "alstm_ppo_csi1000"
sys.path.insert(0, str(S))

import paper_trade as pt  # noqa: E402
import paths  # noqa: E402


@pytest.fixture
def exec_dir(tmp_path, monkeypatch):
    """把执行记录重定向到临时目录，避免污染真实记录。"""
    d = tmp_path / "executions"
    d.mkdir()
    monkeypatch.setattr(paths, "execution_file",
                        lambda date: d / f"exec_{date}.csv")
    monkeypatch.setattr(paths, "ensure_dirs", lambda: None)
    return d


def _today_file(exec_dir):
    return exec_dir / f"exec_{datetime.now().strftime('%Y-%m-%d')}.csv"


def _rows(exec_dir):
    p = _today_file(exec_dir)
    if not p.exists():
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------ 逐笔落盘

def test_每笔委托立即落盘(exec_dir):
    """不能攒到最后一起写 —— 中途被 kill 就全丢了。"""
    pt.append_execution({"vt_symbol": "000001.SZSE", "result": "已提交",
                         "mode": "实盘", "volume": 100}, "093100")
    assert len(_rows(exec_dir)) == 1
    pt.append_execution({"vt_symbol": "000002.SZSE", "result": "已提交",
                         "mode": "实盘", "volume": 200}, "093100")
    assert len(_rows(exec_dir)) == 2


def test_记录包含run_id与remark(exec_dir):
    """没有唯一标识就无法把券商回报逐笔关联回本地意图。"""
    pt.append_execution({"vt_symbol": "000001.SZSE", "remark": "093100_001",
                         "result": "已提交", "mode": "实盘"}, "093100")
    r = _rows(exec_dir)[0]
    assert r["run_id"] == "093100"
    assert r["remark"] == "093100_001"


def test_列顺序稳定(exec_dir):
    pt.append_execution({"vt_symbol": "000001.SZSE"}, "093100")
    with open(_today_file(exec_dir), encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    assert header == pt.EXEC_COLUMNS


# ------------------------------------------------------------ 幂等检查

def test_识别今日已发出的委托(exec_dir):
    pt.append_execution({"vt_symbol": "000001.SZSE", "result": "已委托",
                         "mode": "实盘"}, "093100")
    assert len(pt.today_orders_sent()) == 1


def test_预览不算已下单(exec_dir):
    """dry-run 跑一百次也不该挡住真实执行。"""
    for i in range(3):
        pt.append_execution({"vt_symbol": f"00000{i}.SZSE",
                             "result": "已预览", "mode": "预览"}, "093100")
    assert pt.today_orders_sent() == []


def test_风控拦截不算已下单(exec_dir):
    """被拦截的单没发出去，不该挡住重试。"""
    pt.append_execution({"vt_symbol": "000001.SZSE", "result": "风控拦截",
                         "mode": "实盘", "reason": "可用资金不足"}, "093100")
    assert pt.today_orders_sent() == []


def test_已提交也算已下单(exec_dir):
    """「已提交」是发单前落的盘。进程若在此后被 kill，这笔单可能已经
    发出去了 —— 必须按已下单处理，否则重跑会重复下单。"""
    pt.append_execution({"vt_symbol": "000001.SZSE", "result": "已提交",
                         "mode": "实盘"}, "093100")
    assert len(pt.today_orders_sent()) == 1


def test_无记录时返回空(exec_dir):
    assert pt.today_orders_sent() == []


def test_混合记录只统计真实委托(exec_dir):
    pt.append_execution({"vt_symbol": "A", "result": "已预览",
                         "mode": "预览"}, "1")
    pt.append_execution({"vt_symbol": "B", "result": "风控拦截",
                         "mode": "实盘"}, "1")
    pt.append_execution({"vt_symbol": "C", "result": "已委托",
                         "mode": "实盘"}, "1")
    pt.append_execution({"vt_symbol": "D", "result": "已提交",
                         "mode": "实盘"}, "1")
    sent = pt.today_orders_sent()
    assert {r["vt_symbol"] for r in sent} == {"C", "D"}


# ------------------------------------------------------------ 回归

def test_写入失败不抛异常(exec_dir, monkeypatch):
    """记录写不进去不能连累下单流程崩掉。"""
    def _boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr("builtins.open", _boom)
    pt.append_execution({"vt_symbol": "000001.SZSE"}, "1")   # 不应抛出
