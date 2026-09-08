"""两个只在实盘长时间运行才暴露的网关缺陷。

## 1. 同步接口无超时 → 冻住整个事件循环

SDK 的 order_stock / cancel_order_stock / query_* 内部是
`future.result()` 且不带 timeout（xttrader.py:419），QMT 客户端卡死或
网络抖动时会**永久阻塞**。而 EventEngine 只有一个处理线程，
策略的 on_tick → send_order 全在这个线程上 —— 一次阻塞就冻住整个系统：
行情不再处理、成交回报进不了队列、健康检查停摆，且日志里什么都不会打。

## 2. 委托号盘中重启撞号

_local_id 是实例变量、每次进程启动归零。盘中崩溃重启后，下午第一笔
新单会和上午第一笔同名，而 _orderid_map 是直接赋值，会覆盖旧单的
券商单号 —— 之后撤单撤的是新单还是旧单取决于时序，旧单的状态推送
回来又会覆盖新单的状态。
"""
import inspect
import re

import pytest


# ------------------------------------------------------------ 超时

def test_SDK同步接口确实不带超时():
    """锚定 SDK 行为。哪天 SDK 自己加了 timeout，这条会失败，
    提醒我们重新评估是否还需要 set_timeout。"""
    xttrader = pytest.importorskip("xtquant.xttrader")
    src = inspect.getsource(xttrader.XtQuantTrader.common_op_sync_with_seq)
    assert "future.result()" in src.replace(" ", "")
    # 无 timeout 参数
    assert not re.search(r"future\.result\(\s*timeout", src)


def test_事件引擎只有一个处理线程():
    """这是「阻塞会冻住整个系统」成立的前提。
    哪天改成多线程，阻塞的影响面才会变。"""
    from qmtquant.event.engine import EventEngine
    src = inspect.getsource(EventEngine.__init__)
    assert src.count("target=self._run,") == 1, "处理线程不止一个，请重新评估"


def test_miniqmt网关连接时设置超时():
    from qmtquant.gateway import miniqmt_gateway as g
    src = inspect.getsource(g.MiniQmtGateway.connect)
    assert "set_timeout" in src, "未设置超时，卡死会冻住事件循环"


def test_xt网关连接时设置超时():
    from qmtquant.gateway import xt_gateway as g
    src = inspect.getsource(g.XtGateway.connect)
    assert "set_timeout" in src, "未设置超时，卡死会冻住事件循环"


def test_超时设置失败不阻断连接():
    """老版本 SDK 可能没有 set_timeout，不该因此连不上。"""
    from qmtquant.gateway import miniqmt_gateway as g
    src = inspect.getsource(g.MiniQmtGateway.connect)
    i = src.index("set_timeout")
    assert "try:" in src[:i], "set_timeout 未包在 try 里"


def test_超时是可配置的():
    from qmtquant.config import GatewayConfig
    cfg = GatewayConfig()
    assert hasattr(cfg, "request_timeout")
    assert cfg.request_timeout > 0


def test_run_live把超时传给网关():
    """配置项存在但没传下去等于没有。"""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "run_live.py").read_text(encoding="utf-8")
    assert "request_timeout" in src


# ------------------------------------------------------------ 委托号

def test_委托号带进程会话前缀():
    """只带日期+序号的话，盘中重启后序号从 0 重来，
    下午第一笔会和上午第一笔同名。"""
    from qmtquant.gateway.miniqmt_gateway import MiniQmtGateway
    src = inspect.getsource(MiniQmtGateway._new_orderid)
    assert "_session" in src, "委托号未带会话标识，重启后会撞号"


def test_两个会话的委托号不冲突():
    """模拟盘中重启：两个实例各自从 1 开始编号，委托号必须不同。"""
    from qmtquant.gateway.miniqmt_gateway import MiniQmtGateway

    class _Fake(MiniQmtGateway):
        def __init__(self, session):
            self._local_id = 0
            self._session = session

    morning = _Fake("093000")
    afternoon = _Fake("130500")
    ids_a = {morning._new_orderid() for _ in range(5)}
    ids_b = {afternoon._new_orderid() for _ in range(5)}
    assert not (ids_a & ids_b), f"委托号冲突: {ids_a & ids_b}"


def test_同一会话内委托号递增且唯一():
    from qmtquant.gateway.miniqmt_gateway import MiniQmtGateway

    class _Fake(MiniQmtGateway):
        def __init__(self):
            self._local_id = 0
            self._session = "093000"

    g = _Fake()
    ids = [g._new_orderid() for _ in range(100)]
    assert len(set(ids)) == 100
    assert ids == sorted(ids), "委托号应单调递增，便于按序排查"
