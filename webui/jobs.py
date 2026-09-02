"""后台任务管理 —— 启动、跟踪状态、留存日志。

训练动辄一两个小时，必须能关掉浏览器再回来看。因此：

- 任务以子进程运行，日志实时写文件，不驻留在内存里
- 任务元信息落盘到 webui/jobs/index.json，服务重启后历史仍在
- 同一任务不允许并发（训练脚本会覆盖同一批产物，并发跑必然互相破坏）
"""
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .registry import ROOT, TASK_BY_ID, build_command

JOBS_DIR = ROOT / "webui" / "jobs"
INDEX_FILE = JOBS_DIR / "index.json"

_lock = threading.Lock()
#: job_id -> Popen，仅本进程存活期间有效
_procs: dict = {}


@dataclass
class Job:
    id: str
    task_id: str
    task_name: str
    cmd: list
    status: str            # running / success / failed / killed
    started_at: str
    finished_at: str = ""
    returncode: int | None = None
    pid: int | None = None
    log_file: str = ""
    params: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        end = (datetime.fromisoformat(self.finished_at) if self.finished_at
               else datetime.now())
        return (end - datetime.fromisoformat(self.started_at)).total_seconds()


def _load_index() -> list:
    if not INDEX_FILE.exists():
        return []
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_index(rows: list) -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def _reconcile(rows: list) -> list:
    """服务重启后，把已经不存在的进程从 running 改掉。

    不这样做的话，一次崩溃会让任务永远显示「运行中」，
    后续同名任务再也启动不了。
    """
    import psutil

    changed = False
    for r in rows:
        if r.get("status") != "running":
            continue
        pid = r.get("pid")
        alive = False
        if pid:
            try:
                p = psutil.Process(pid)
                # PID 会被复用，比对启动时间避免误判
                alive = (p.is_running()
                         and datetime.fromtimestamp(p.create_time())
                         >= datetime.fromisoformat(r["started_at"])
                         - __import__("datetime").timedelta(seconds=5))
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                alive = False
        if not alive and r["id"] not in _procs:
            r["status"] = "unknown"
            r["finished_at"] = r.get("finished_at") or datetime.now().isoformat(
                timespec="seconds")
            changed = True
    if changed:
        _save_index(rows)
    return rows


def list_jobs(limit: int = 50) -> list:
    with _lock:
        rows = _reconcile(_load_index())
    # 耗时在这里算好，模板里不做时间运算
    now = datetime.now()
    for r in rows[:limit]:
        try:
            start = datetime.fromisoformat(r["started_at"])
            end = (datetime.fromisoformat(r["finished_at"])
                   if r.get("finished_at") else now)
            r["duration_s"] = (end - start).total_seconds()
        except (ValueError, KeyError):
            r["duration_s"] = None
    return rows[:limit]


def get_job(job_id: str) -> dict | None:
    for r in list_jobs(limit=10_000):
        if r["id"] == job_id:
            return r
    return None


def running_jobs() -> list:
    return [r for r in list_jobs(limit=10_000) if r["status"] == "running"]


def is_task_running(task_id: str) -> bool:
    return any(r["task_id"] == task_id for r in running_jobs())


def start(task_id: str, params: dict) -> Job:
    task = TASK_BY_ID.get(task_id)
    if task is None:
        raise ValueError(f"未登记的任务: {task_id}")
    if is_task_running(task_id):
        raise RuntimeError(f"{task.name} 正在运行中，不允许并发")

    cmd = build_command(task_id, params)
    job_id = f"{task_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOBS_DIR / f"{job_id}.log"

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)
    log_f.write(f"$ {' '.join(cmd)}\n\n")
    log_f.flush()

    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env,
        stdout=log_f, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    job = Job(
        id=job_id, task_id=task_id, task_name=task.name, cmd=cmd,
        status="running", started_at=datetime.now().isoformat(timespec="seconds"),
        pid=proc.pid, log_file=str(log_path), params=params,
    )

    with _lock:
        rows = _load_index()
        rows.insert(0, asdict(job))
        _save_index(rows)
    _procs[job_id] = (proc, log_f)

    threading.Thread(target=_wait, args=(job_id,), daemon=True).start()
    return job


def _wait(job_id: str) -> None:
    entry = _procs.get(job_id)
    if not entry:
        return
    proc, log_f = entry
    rc = proc.wait()
    try:
        log_f.close()
    except OSError:
        pass
    _procs.pop(job_id, None)

    with _lock:
        rows = _load_index()
        for r in rows:
            if r["id"] == job_id:
                r["status"] = "success" if rc == 0 else "failed"
                r["returncode"] = rc
                r["finished_at"] = datetime.now().isoformat(timespec="seconds")
                break
        _save_index(rows)


def stop(job_id: str) -> bool:
    entry = _procs.get(job_id)
    if entry:
        proc, _ = entry
        try:
            proc.terminate()
        except OSError:
            return False
    else:
        row = get_job(job_id)
        if not row or not row.get("pid"):
            return False
        try:
            os.kill(row["pid"], signal.SIGTERM)
        except OSError:
            return False

    with _lock:
        rows = _load_index()
        for r in rows:
            if r["id"] == job_id:
                r["status"] = "killed"
                r["finished_at"] = datetime.now().isoformat(timespec="seconds")
                break
        _save_index(rows)
    return True


def read_log(job_id: str, tail: int = 400) -> str:
    row = get_job(job_id)
    if not row:
        return ""
    p = Path(row["log_file"])
    if not p.exists():
        return "(日志文件不存在)"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return f"(读取日志失败: {e})"
    return "\n".join(lines[-tail:])
