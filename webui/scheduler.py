"""定时任务调度 —— 盘前 / 盘后自动更新行情。

不引 APScheduler，一个后台线程每 20 秒扫一遍就够了：调度精度要求是分钟级，
任务数是个位数，引一个依赖不划算。

## 触发条件

每条计划记录 `time`(HH:MM) 与 `weekdays`。线程扫描时，满足以下全部条件才触发：

1. 今天在 `weekdays` 内
2. 当前时间已过 `time`
3. 今天尚未跑过（`last_run_date` 不是今天）
4. 距计划时间未超过 `catchup_hours`
5. 该任务当前没有实例在跑

条件 3 用日期而非时间戳判断，因此**错过会补跑**：若 08:40 的计划因关机
没执行，09:30 开机后扫描会立刻补上 —— 宁可晚一点跑，不要静默跳过。

条件 4 给补跑加了有效期，因为迟到太久的补跑没有意义甚至有害：晚上 9 点
才来跑「盘前补齐」纯属浪费，那个时点该跑的是盘后更新。因此两个预设的
窗口不同 —— 盘前 2 小时（过了开盘就失去意义），盘后 8 小时
（当晚任何时候开机补上今日数据都是有用的）。

## 交易日判定

只按周一至周五判断，**不含法定节假日**。原因是拿不到可靠的未来交易日历
（Qlib 的 day.txt 只到历史最后一天，xtdata 的日历要 QMT 在线）。

节假日误触发的后果是无害的：下载脚本取不到新数据，导出重跑一遍相同内容，
日志里会提示「日历最后一天未推进」。用假日空跑换取实现上的确定性是划算的。
"""
import json
import threading
import time as _time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

from . import jobs
from .registry import ROOT, TASK_BY_ID

SCHEDULE_FILE = ROOT / "webui" / "schedules.json"

_lock = threading.Lock()
_thread = None
_stop = threading.Event()

WEEKDAY_NAMES = ["一", "二", "三", "四", "五", "六", "日"]


@dataclass
class Schedule:
    id: str
    name: str
    task_id: str
    time: str                       # HH:MM
    params: dict = field(default_factory=dict)
    weekdays: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    enabled: bool = True
    #: 补跑有效期（小时）。超过这个时长仍未执行就放弃，等下一个交易日
    catchup_hours: float = 4.0
    last_run_date: str = ""
    last_job_id: str = ""


#: 开箱即用的预设。盘前 08:40 留足时间在 09:15 生成信号之前完成，
#: 盘后 15:20 是收盘后 20 分钟 —— 行情商的日线定稿通常需要十几分钟
PRESETS = [
    Schedule(id="pre_market", name="盘前补齐行情", task_id="update_market_data",
             time="08:40", params={"mode": "pre"},
             # 过了开盘再补「盘前」就没意义了
             catchup_hours=2.0),
    Schedule(id="post_market", name="盘后更新行情", task_id="update_market_data",
             time="15:20", params={"mode": "post"},
             # 当晚任何时候开机，补上今日数据都是有用的
             catchup_hours=8.0),
]


def _load() -> list:
    if not SCHEDULE_FILE.exists():
        return [asdict(s) for s in PRESETS]
    try:
        return json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [asdict(s) for s in PRESETS]


def _save(rows: list) -> None:
    SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                             encoding="utf-8")


def list_schedules() -> list:
    with _lock:
        rows = _load()
    today = date.today().isoformat()
    for r in rows:
        r["ran_today"] = r.get("last_run_date") == today
        r["weekday_label"] = "".join(WEEKDAY_NAMES[d] for d in r.get("weekdays", []))
        task = TASK_BY_ID.get(r["task_id"])
        r["task_name"] = task.name if task else r["task_id"]
        r["next_hint"] = _next_hint(r)
    return rows


def _overdue_minutes(row: dict, now: datetime) -> float | None:
    """今天已过计划时间多少分钟。未到点或今天不执行返回 None。"""
    if now.weekday() not in row.get("weekdays", []):
        return None
    try:
        hh, mm = (int(x) for x in row["time"].split(":"))
    except (ValueError, AttributeError):
        return None
    delta = (now.hour * 60 + now.minute) - (hh * 60 + mm)
    return delta if delta >= 0 else None


def _next_hint(row: dict) -> str:
    """下次触发的说明文字。"""
    if not row.get("enabled"):
        return "已停用"
    now = datetime.now()
    wd = now.weekday()
    hh, mm = (int(x) for x in row["time"].split(":"))

    overdue = _overdue_minutes(row, now)
    if overdue is not None:
        if row.get("last_run_date") == date.today().isoformat():
            return "今日已执行"
        window = float(row.get("catchup_hours", 4.0)) * 60
        if overdue <= window:
            return f"待补跑（已过 {int(overdue)} 分钟）"
        return "今日已错过，等下个交易日"

    if wd in row.get("weekdays", []) and (now.hour, now.minute) < (hh, mm):
        return f"今日 {row['time']}"
    for i in range(1, 8):
        nxt = (wd + i) % 7
        if nxt in row.get("weekdays", []):
            return f"周{WEEKDAY_NAMES[nxt]} {row['time']}"
    return "-"


def set_enabled(sched_id: str, enabled: bool) -> bool:
    with _lock:
        rows = _load()
        for r in rows:
            if r["id"] == sched_id:
                r["enabled"] = bool(enabled)
                _save(rows)
                return True
    return False


def update(sched_id: str, time_str: str = None, weekdays: list = None,
           catchup_hours: float = None) -> bool:
    with _lock:
        rows = _load()
        for r in rows:
            if r["id"] != sched_id:
                continue
            if time_str:
                try:
                    hh, mm = (int(x) for x in time_str.split(":"))
                    if not (0 <= hh < 24 and 0 <= mm < 60):
                        return False
                    r["time"] = f"{hh:02d}:{mm:02d}"
                except (ValueError, AttributeError):
                    return False
            if weekdays is not None:
                r["weekdays"] = sorted({int(d) for d in weekdays
                                        if 0 <= int(d) <= 6})
            if catchup_hours is not None:
                try:
                    v = float(catchup_hours)
                    if 0 <= v <= 24:
                        r["catchup_hours"] = v
                except (TypeError, ValueError):
                    pass
            _save(rows)
            return True
    return False


def trigger_now(sched_id: str) -> str | None:
    """立即手动执行一条计划，不影响其定时状态。"""
    rows = list_schedules()
    row = next((r for r in rows if r["id"] == sched_id), None)
    if row is None:
        return None
    job = jobs.start(row["task_id"], dict(row.get("params", {})))
    return job.id


def _mark_ran(sched_id: str, job_id: str) -> None:
    with _lock:
        rows = _load()
        for r in rows:
            if r["id"] == sched_id:
                r["last_run_date"] = date.today().isoformat()
                r["last_job_id"] = job_id
                break
        _save(rows)


def _tick() -> None:
    now = datetime.now()
    today = date.today().isoformat()

    with _lock:
        rows = _load()

    for r in rows:
        if not r.get("enabled"):
            continue
        if now.weekday() not in r.get("weekdays", []):
            continue
        if r.get("last_run_date") == today:
            continue

        overdue = _overdue_minutes(r, now)
        if overdue is None:
            continue
        # 迟到太久的补跑没意义 —— 晚上 9 点跑「盘前补齐」纯属浪费
        if overdue > float(r.get("catchup_hours", 4.0)) * 60:
            continue

        # 同任务已在运行时跳过本次，等下一个交易日 —— 强行并发会破坏产物
        if jobs.is_task_running(r["task_id"]):
            continue

        try:
            job = jobs.start(r["task_id"], dict(r.get("params", {})))
            _mark_ran(r["id"], job.id)
            print(f"[调度] {now:%H:%M:%S} 触发 {r['name']} -> {job.id}")
        except (RuntimeError, ValueError, OSError) as e:
            print(f"[调度] {r['name']} 启动失败: {e}")


def _loop() -> None:
    while not _stop.wait(20):
        try:
            _tick()
        except Exception as e:  # 调度线程绝不能死
            print(f"[调度] 扫描异常: {e}")


def start_scheduler() -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, daemon=True, name="scheduler")
    _thread.start()


def stop_scheduler() -> None:
    _stop.set()


def is_running() -> bool:
    return bool(_thread and _thread.is_alive())
