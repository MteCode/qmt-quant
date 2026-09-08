"""常驻服务管理。

## 这些用例守什么

管理台此前只能触发一次性任务，没有「常驻服务」的概念 ——
后端实际上什么都没在跑，页面只是在展示磁盘上的文件。

服务与任务的运维语义完全不同：
- 任务「跑完了」是正常终态；服务「跑完了」意味着出事了
- 任务重复启动是重复劳动；服务重复启动会双开，后果可能是重复下单
- 服务停了就没人干活，所以要能一眼看出死没死

其中最容易写错的是 **Windows 上的优雅退出**：
`os.kill(pid, SIGTERM)` 在 Windows 上实际是 TerminateProcess，
直接杀死、信号处理器不触发。实盘引擎因此不会执行
「撤单 → 停策略 → 断网关」，挂单会留在券商那边。
"""
import os
import signal
from datetime import datetime, timedelta

import pytest

from webui import services


# ------------------------------------------------------------ 注册表

def test_服务都有唯一id():
    ids = [s.id for s in services.SERVICES]
    assert len(ids) == len(set(ids))


def test_实盘引擎标记为危险():
    """会产生真实委托的服务必须标 dangerous，前端据此要求二次确认。"""
    assert services.BY_ID["live_qmt"].dangerous is True


def test_模拟撮合不标危险():
    """sim 网关不接券商，不该吓唬人。"""
    assert services.BY_ID["live_sim"].dangerous is False


def test_服务命令指向真实脚本():
    """指向不存在的脚本会在启动时才暴露，那时已经写了状态文件。"""
    for s in services.SERVICES:
        script = next((a for a in s.args if a.endswith(".py")), None)
        assert script, f"{s.id} 没有脚本参数"
        assert (services.ROOT / script).exists(), f"{s.id} 脚本不存在: {script}"


# ------------------------------------------------------------ 状态判断

def test_未启动的服务不算运行():
    st = services.status("live_sim")
    # 本用例不启动任何东西；若环境里恰好在跑，跳过而不是误判
    if st["running"]:
        pytest.skip("live_sim 正在运行，跳过")
    assert st["healthy"] is False


def test_未知服务返回exists为假():
    assert services.status("nonexistent")["exists"] is False


def test_进程活着但心跳失联不算健康(monkeypatch, tmp_path):
    """进程活着不等于在干活 —— 实盘引擎可能卡在一次阻塞调用上，
    进程状态正常但事件循环早已冻住。只看进程会得到「一切正常」的假象。"""
    monkeypatch.setattr(services, "_pid_alive", lambda pid, s: True)
    monkeypatch.setattr(services, "_load", lambda: {
        "scheduler": {"pid": 1, "started_at": datetime.now().isoformat(),
                      "log_file": None}})
    # 心跳很旧
    monkeypatch.setattr(services, "_heartbeat_age", lambda svc: 9999.0)
    st = services.status("scheduler")
    assert st["running"] is True
    assert st["healthy"] is False, "心跳失联仍报健康"
    assert st["heartbeat_ok"] is False


def test_心跳新鲜则健康(monkeypatch):
    monkeypatch.setattr(services, "_pid_alive", lambda pid, s: True)
    monkeypatch.setattr(services, "_load", lambda: {
        "scheduler": {"pid": 1, "started_at": datetime.now().isoformat(),
                      "log_file": None}})
    monkeypatch.setattr(services, "_heartbeat_age", lambda svc: 10.0)
    st = services.status("scheduler")
    assert st["healthy"] is True


def test_pid复用不会误判存活(monkeypatch):
    """PID 会被系统复用。只比对 PID 会把无关进程认成自己的服务，
    而按 PID 停服务时误判会杀掉别人的程序。"""
    import psutil

    class _FakeProc:
        def create_time(self):
            # 与记录的启动时间差一小时，远超容忍窗口
            return (datetime.now() - timedelta(hours=1)).timestamp()

        def is_running(self):
            return True

    monkeypatch.setattr(psutil, "Process", lambda pid: _FakeProc())
    assert services._pid_alive(1, datetime.now().isoformat()) is False


# ------------------------------------------------------------ 优雅退出

def test_windows下用CTRL_BREAK而非SIGTERM():
    """os.kill(pid, SIGTERM) 在 Windows 上是 TerminateProcess ——
    直接杀死，信号处理器不触发，实盘引擎的撤单逻辑不会执行。

    进程用 CREATE_NEW_PROCESS_GROUP 启动，所以可以发 CTRL_BREAK_EVENT，
    在子进程里表现为可捕获的 SIGBREAK。
    """
    import inspect
    src = inspect.getsource(services._signal_stop)
    assert "CTRL_BREAK_EVENT" in src, "Windows 上未用 CTRL_BREAK_EVENT"
    assert 'os.name == "nt"' in src, "未区分平台"


def test_启动时用新进程组():
    """不用 CREATE_NEW_PROCESS_GROUP 就发不了 CTRL_BREAK_EVENT。"""
    import inspect
    src = inspect.getsource(services.start)
    assert "CREATE_NEW_PROCESS_GROUP" in src


@pytest.mark.parametrize("script", ["run_live.py", "run_scheduler.py"])
def test_服务脚本处理SIGBREAK(script):
    """管理台发的是 CTRL_BREAK_EVENT，脚本不处理 SIGBREAK 就等于没有优雅退出。"""
    src = (services.ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert "SIGBREAK" in src, f"{script} 未处理 SIGBREAK"


@pytest.mark.parametrize("script", ["run_live.py", "run_scheduler.py"])
def test_服务脚本写心跳(script):
    """不写心跳就只能判断进程存活，卡死时看不出来。"""
    src = (services.ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert "services.beat(" in src, f"{script} 未写心跳"


# ------------------------------------------------------------ 互斥

def test_两个实盘引擎不能同时开(monkeypatch):
    """同时跑 sim 与 miniqmt 会对同一份信号重复下单。"""
    monkeypatch.setattr(
        services, "status",
        lambda sid: {"running": sid == "live_sim", "pid": 1,
                     "exists": True, "healthy": True})
    ok, msg = services.start("live_qmt")
    assert ok is False
    assert "重复下单" in msg


# ------------------------------------------------------------ 心跳写入

def test_心跳写入与读取(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "ROOT", tmp_path)
    services.beat("unittest")
    p = tmp_path / "webui" / ".heartbeat_unittest"
    assert p.exists()
    age = services._heartbeat_age(
        services.Service(id="x", name="x", desc="", args=[],
                         heartbeat="webui/.heartbeat_unittest"))
    assert age is not None and age < 5


def test_心跳写入失败不抛异常(monkeypatch):
    """心跳写不进去不该拖垮服务本身。"""
    def _boom(*a, **k):
        raise OSError("read-only fs")
    monkeypatch.setattr("pathlib.Path.write_text", _boom)
    services.beat("unittest")       # 不应抛出
