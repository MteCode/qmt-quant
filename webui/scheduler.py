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
    #: 每隔多少分钟重复一次。0（默认）表示每日只跑一次。
    #:
    #: 用于盘中需要反复执行的事，比如刷新持仓快照 —— 只在收盘后快照，
    #: 盘中看到的永远是昨天的持仓，而人会拿它当实时数据看。
    interval_minutes: int = 0
    #: 重复的结束时间 HH:MM。仅 interval_minutes > 0 时有效
    end_time: str = ""
    #: 重复时是否跳过午休（11:30-13:00）。A 股午休持仓不变，
    #: 照常轮询只是白白打扰 QMT
    skip_lunch: bool = True
    last_run_date: str = ""
    #: 上次执行的完整时间戳。每日一次的计划用不到，
    #: 但间隔重复必须精确到分钟，只有日期判不出「这一轮该不该跑」
    last_run_at: str = ""
    last_job_id: str = ""


#: 开箱即用的预设，按交易日时序排列。
#:
#: 此前只有两条补行情的计划，交易动作一条都没有 —— 生成信号、快照持仓、
#: 下单执行全靠人手点，这正是「这运行一下那运行一下」的由来。
#:
#: 下单类计划默认 **enabled=False**：自动下单是要花真钱的，
#: 必须由人显式打开。其余环节（补数据、生成信号、快照、对账）
#: 都是只读或只写本地文件，默认开启是安全的。
PRESETS = [
    # ---- 盘前 ----
    Schedule(id="pre_market", name="盘前补齐行情", task_id="update_market_data",
             time="08:40", params={"mode": "pre"},
             # 过了开盘再补「盘前」就没意义了
             catchup_hours=2.0),
    Schedule(id="gen_signal", name="盘前生成信号", task_id="generate_signal",
             time="09:05",
             # 信号要在开盘前算好；开盘后再算就错过了预期的建仓时点
             catchup_hours=1.0),
    Schedule(id="pre_snapshot", name="盘前快照持仓", task_id="snapshot_positions",
             time="09:15", catchup_hours=1.0),

    # ---- 盘中持续刷新 ----
    # 只在收盘后快照的话，盘中页面上看到的永远是昨天的持仓，
    # 而人会拿它当实时数据看 —— 这比没有数据更危险。
    Schedule(id="intraday_snapshot", name="盘中刷新持仓",
             task_id="snapshot_positions",
             time="09:30", end_time="15:00", interval_minutes=5,
             catchup_hours=0.2),

    # ---- 开盘后下单 ----
    # 09:30 集合竞价刚结束，价格波动最剧烈；等 5 分钟让盘口稳一稳。
    # 默认关闭 —— 打开它意味着系统会自动花钱。
    Schedule(id="execute_trade", name="开盘调仓下单", task_id="paper_trade",
             time="09:35", params={"dry_run": "false"},
             catchup_hours=1.0, enabled=False),

    # ---- 盘中 ----
    Schedule(id="risk_check", name="盘中风控巡检", task_id="risk_monitor",
             time="11:00", catchup_hours=1.0, enabled=False),

    # ---- 收盘 ----
    Schedule(id="post_market", name="盘后更新行情", task_id="update_market_data",
             time="15:20", params={"mode": "post"},
             # 当晚任何时候开机，补上今日数据都是有用的
             catchup_hours=8.0),
    Schedule(id="post_snapshot", name="盘后快照持仓", task_id="snapshot_positions",
             time="15:25", catchup_hours=6.0),
    Schedule(id="reconcile", name="盘后对账", task_id="reconcile",
             time="15:30", catchup_hours=6.0),
    Schedule(id="track_equity", name="记录实盘净值", task_id="track_equity",
             # 漏记的日子补不回来 —— 券商查不到历史净值序列，
             # 所以补跑窗口给到当晚全程
             time="15:35", catchup_hours=8.0),
]


def _load() -> list:
    """读取计划表，并把新增的预设合并进来。

    只读文件的话，老装机永远看不到后来加的预设 —— 加了「盘前生成信号」
    「盘后对账」这些计划，用户那边却什么都没变，还以为是配置没生效。
    合并策略：文件里已有的保留（用户可能改过时间或关掉了），
    文件里没有的按预设补进去。
    """
    if not SCHEDULE_FILE.exists():
        return [asdict(s) for s in PRESETS]
    try:
        rows = json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [asdict(s) for s in PRESETS]

    have = {r.get("id") for r in rows}
    added = [asdict(s) for s in PRESETS if s.id not in have]
    if added:
        rows.extend(added)
        _save(rows)
    return rows


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


def _hhmm(s: str) -> int | None:
    """"HH:MM" -> 当日第几分钟。解析不了返回 None。"""
    try:
        hh, mm = (int(x) for x in s.split(":"))
        return hh * 60 + mm
    except (ValueError, AttributeError):
        return None


def _in_lunch(minute_of_day: int) -> bool:
    """是否在午休（11:30-13:00）。A 股午休持仓不变。"""
    return 11 * 60 + 30 <= minute_of_day < 13 * 60


def _interval_due(row: dict, now: datetime) -> bool:
    """间隔重复型计划本轮是否该跑。

    与「每日一次」的判断完全不同：后者按日期判重，前者必须按时间戳 ——
    只有日期的话，当天第一次跑完就再也不会触发了。
    """
    iv = int(row.get("interval_minutes", 0) or 0)
    if iv <= 0:
        return False
    if now.weekday() not in row.get("weekdays", []):
        return False

    start = _hhmm(row.get("time", ""))
    end = _hhmm(row.get("end_time", ""))
    if start is None or end is None:
        return False

    cur = now.hour * 60 + now.minute
    if not (start <= cur <= end):
        return False
    if row.get("skip_lunch", True) and _in_lunch(cur):
        return False

    last = row.get("last_run_at")
    if not last:
        return True
    try:
        gap = (now - datetime.fromisoformat(last)).total_seconds() / 60
    except ValueError:
        return True
    return gap >= iv


def _next_hint(row: dict) -> str:
    """下次触发的说明文字。"""
    if not row.get("enabled"):
        return "已停用"
    now = datetime.now()
    wd = now.weekday()

    iv = int(row.get("interval_minutes", 0) or 0)
    if iv > 0:
        window = f"{row.get('time','')}~{row.get('end_time','')}"
        if wd not in row.get("weekdays", []):
            return f"非交易日 · 每 {iv} 分钟（{window}）"
        cur = now.hour * 60 + now.minute
        start, end = _hhmm(row.get("time", "")), _hhmm(row.get("end_time", ""))
        if start is None or end is None:
            return f"每 {iv} 分钟（时间窗配置有误）"
        if cur < start:
            return f"今日 {row['time']} 起，每 {iv} 分钟"
        if cur > end:
            return f"今日已结束 · 每 {iv} 分钟（{window}）"
        if row.get("skip_lunch", True) and _in_lunch(cur):
            return f"午休暂停 · 13:00 恢复（每 {iv} 分钟）"
        last = row.get("last_run_at")
        if last:
            try:
                gap = (now - datetime.fromisoformat(last)).total_seconds() / 60
                left = max(0, iv - gap)
                return f"运行中 · 约 {left:.0f} 分钟后下一轮"
            except ValueError:
                pass
        return f"运行中 · 每 {iv} 分钟"

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
                # 间隔重复型靠这个判重，精度必须到分钟
                r["last_run_at"] = datetime.now().isoformat(
                    timespec="seconds")
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

        if int(r.get("interval_minutes", 0) or 0) > 0:
            # 间隔重复型：按时间戳判重，不看 last_run_date
            if not _interval_due(r, now):
                continue
        else:
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
