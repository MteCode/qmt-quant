"""常驻服务管理 —— 让后端真的有东西在跑。

## 与 jobs.py 的区别

`jobs.py` 管的是**一次性任务**：跑完就退，结果落在日志和产物文件里。
本模块管的是**常驻服务**：起来就一直跑，直到被显式停掉或崩溃。

两者的运维语义完全不同：
- 任务「跑完了」是正常终态；服务「跑完了」意味着出事了
- 任务重复启动是重复劳动；服务重复启动会双开，后果可能是重复下单
- 任务失败看日志就够；服务失败要能立刻看出来，因为它停了就没人干活了

所以服务有独立的状态文件、PID 校验与心跳，而不是塞进 jobs 里。

## 状态怎么判断

三段递进，任何一段不成立就不算健康：

1. **状态文件里有记录** —— 服务被启动过
2. **PID 对应的进程还活着，且启动时间对得上** —— PID 会被系统复用，
   只比对 PID 会把无关进程认成自己的服务（`jobs.py` 里有同样的坑）
3. **心跳文件在容忍窗口内更新过** —— 进程活着不等于在干活。
   实盘引擎可能卡在一次阻塞的 order_stock 上，进程状态是 running，
   但事件循环早已冻住。只看进程会得到「一切正常」的假象。

第 3 点需要服务自己写心跳。没写的服务只做前两段判断，
并在状态里标注 `heartbeat: null` —— 说明「没有这层保障」，
而不是假装它健康。

## 为什么不用 Windows 服务 / NSSM

那套要管理员权限、安装卸载、注册表，出问题排查链条长。这里的定位是
「管理台能启停、能看出死没死」，够用且透明。真要开机自启，
外面再包一层计划任务或 NSSM 即可，两者不冲突。
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
STATE_FILE = ROOT / "webui" / "services.json"
LOG_DIR = ROOT / "webui" / "jobs"

#: PID 复用判定窗口（秒）。与 jobs.py 保持一致
PID_MATCH_WINDOW_S = 60

_lock = threading.Lock()


@dataclass
class Service:
    id: str
    name: str
    desc: str
    args: list                      # 相对 python 可执行文件的参数
    #: 心跳文件相对项目根的路径。为 None 表示该服务不写心跳
    heartbeat: str | None = None
    #: 心跳超过这个秒数没更新就判为失联
    heartbeat_timeout: int = 120
    #: 停止时是否需要额外确认（会影响真实交易的服务）
    dangerous: bool = False
    note: str = ""


SERVICES = [
    Service(
        id="scheduler",
        name="定时调度守护",
        desc="按交易日程自动跑盘前补数据、生成信号、快照持仓、盘后对账。"
             "关掉管理台也继续运行。",
        args=["-u", "scripts/run_scheduler.py"],
        heartbeat="webui/.heartbeat_scheduler",
        note="下单类计划默认关闭，需在定时任务页显式打开",
    ),
    Service(
        id="live_sim",
        name="实盘引擎（模拟撮合）",
        desc="事件驱动引擎 + 本地模拟撮合网关。不接券商，用于全链路联调。",
        args=["-u", "scripts/run_live.py", "--gateway", "sim"],
        heartbeat="webui/.heartbeat_live",
        note="安全：不会产生真实委托",
    ),
    Service(
        id="live_qmt",
        name="实盘引擎（miniQMT）",
        desc="事件驱动引擎 + miniQMT 网关。**会产生真实委托**，"
             "需 QMT 客户端已登录。",
        args=["-u", "scripts/run_live.py", "--gateway", "miniqmt"],
        heartbeat="webui/.heartbeat_live",
        dangerous=True,
        note="会下真实委托；先用「模拟撮合」跑通再切",
    ),
]

BY_ID = {s.id: s for s in SERVICES}


# ------------------------------------------------------------ 状态存取

def _load() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(d: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def _pid_alive(pid: int, started_at: str) -> bool:
    """该 PID 当前的进程是否仍是当初启动的那个。

    PID 会被系统复用。判断存活时误判只是显示错误，但**按 PID 停服务时
    误判会杀掉无关程序**，所以停之前走同一套校验。
    """
    import psutil
    try:
        p = psutil.Process(pid)
        delta = abs(p.create_time()
                    - datetime.fromisoformat(started_at).timestamp())
        return p.is_running() and delta <= PID_MATCH_WINDOW_S
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, OSError):
        return False


def _signal_stop(pid: int) -> tuple[bool, str]:
    """请求进程优雅退出。

    ## Windows 上不能用 SIGTERM

    `os.kill(pid, SIGTERM)` 在 Windows 上实际调的是 `TerminateProcess` ——
    **直接杀死，信号处理器根本不会触发**。对实盘引擎这是致命的：
    `finally` 里的「撤单 → 停策略 → 断网关」全都不会跑，
    挂单会留在券商那边，本地状态也来不及落库。

    进程是用 CREATE_NEW_PROCESS_GROUP 启的，所以可以发
    `CTRL_BREAK_EVENT` —— 这个在子进程里表现为可捕获的 SIGBREAK，
    走的是和 Ctrl+C 相同的退出路径。
    """
    if os.name == "nt":
        try:
            os.kill(pid, signal.CTRL_BREAK_EVENT)
            return True, ""
        except (OSError, AttributeError, ValueError) as e:
            return False, str(e)
    try:
        os.kill(pid, signal.SIGTERM)
        return True, ""
    except OSError as e:
        return False, str(e)


def _heartbeat_age(svc: Service) -> float | None:
    """心跳距今多少秒。服务不写心跳或文件不存在时返回 None。"""
    if not svc.heartbeat:
        return None
    p = ROOT / svc.heartbeat
    if not p.exists():
        return None
    try:
        return (datetime.now().timestamp() - p.stat().st_mtime)
    except OSError:
        return None


# ------------------------------------------------------------ 查询

def status(sid: str) -> dict:
    """单个服务的状态。"""
    svc = BY_ID.get(sid)
    if svc is None:
        return {"id": sid, "exists": False}

    rec = _load().get(sid, {})
    pid = rec.get("pid")
    started = rec.get("started_at", "")
    alive = bool(pid) and _pid_alive(pid, started)

    hb_age = _heartbeat_age(svc) if alive else None
    hb_ok = None
    if svc.heartbeat and alive:
        hb_ok = hb_age is not None and hb_age <= svc.heartbeat_timeout

    # 健康 = 进程活着 且（不写心跳 或 心跳新鲜）
    healthy = alive and (hb_ok is not False)

    return {
        "id": sid, "exists": True, "name": svc.name, "desc": svc.desc,
        "note": svc.note, "dangerous": svc.dangerous,
        "running": alive, "healthy": healthy,
        "pid": pid if alive else None,
        "started_at": started if alive else None,
        "uptime_s": ((datetime.now()
                      - datetime.fromisoformat(started)).total_seconds()
                     if alive and started else None),
        "heartbeat_age": round(hb_age, 1) if hb_age is not None else None,
        "heartbeat_ok": hb_ok,
        "has_heartbeat": bool(svc.heartbeat),
        "log_file": rec.get("log_file"),
        "stopped_at": rec.get("stopped_at"),
        "exit_note": rec.get("exit_note"),
    }


def list_status() -> list[dict]:
    return [status(s.id) for s in SERVICES]


def read_log(sid: str, tail: int = 400) -> str:
    rec = _load().get(sid, {})
    p = rec.get("log_file")
    if not p or not Path(p).exists():
        return ""
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-tail:])
    except OSError as e:
        return f"[日志读取失败] {e}"


# ------------------------------------------------------------ 启停

def start(sid: str) -> tuple[bool, str]:
    svc = BY_ID.get(sid)
    if svc is None:
        return False, f"未登记的服务: {sid}"

    with _lock:
        st = status(sid)
        if st["running"]:
            return False, f"{svc.name} 已在运行（PID {st['pid']}）"

        # 实盘引擎两个变体共用心跳文件，不能同时开 —— 会双份下单
        if sid.startswith("live_"):
            for other in ("live_sim", "live_qmt"):
                if other != sid and status(other)["running"]:
                    return False, (f"{BY_ID[other].name} 正在运行，"
                                   f"请先停止 —— 两个引擎同时跑会重复下单")

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = LOG_DIR / f"service-{sid}-{ts}.log"
        cmd = [str(PYTHON)] + list(svc.args)

        try:
            f = open(log_path, "w", encoding="utf-8", buffering=1)
            f.write(f"$ {' '.join(cmd)}\n\n")
            f.flush()
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,
                                      "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except OSError as e:
            return False, f"启动失败: {e}"

        d = _load()
        d[sid] = {
            "pid": proc.pid,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "log_file": str(log_path),
            "cmd": cmd,
            "stopped_at": None,
            "exit_note": None,
        }
        _save(d)
        return True, f"{svc.name} 已启动（PID {proc.pid}）"


def stop(sid: str) -> tuple[bool, str]:
    svc = BY_ID.get(sid)
    if svc is None:
        return False, f"未登记的服务: {sid}"

    with _lock:
        rec = _load().get(sid, {})
        pid = rec.get("pid")
        if not pid or not _pid_alive(pid, rec.get("started_at", "")):
            d = _load()
            if sid in d:
                d[sid]["stopped_at"] = datetime.now().isoformat(
                    timespec="seconds")
                _save(d)
            return False, f"{svc.name} 当前未运行"

        ok, err = _signal_stop(pid)
        if not ok:
            return False, f"停止失败: {err}"

        d = _load()
        d[sid]["stopped_at"] = datetime.now().isoformat(timespec="seconds")
        d[sid]["exit_note"] = "手动停止"
        _save(d)
        return True, f"{svc.name} 停止信号已发送（PID {pid}）"


def beat(name: str) -> None:
    """服务自己调用，写心跳。

    进程活着不等于在干活 —— 实盘引擎可能卡在一次阻塞调用上，
    进程状态正常但事件循环早已冻住。心跳是唯一能区分这两者的信号。
    """
    p = ROOT / "webui" / f".heartbeat_{name}"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(datetime.now().isoformat(timespec="seconds"),
                     encoding="utf-8")
    except OSError:
        pass        # 心跳写不进去不该拖垮服务本身
