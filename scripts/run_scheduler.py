"""定时调度守护进程 —— 让交易日程脱离管理台独立运行。

## 为什么要独立

调度器原本是 Flask 进程里的 daemon 线程（webui/scheduler.py），
后果是**关掉管理台就没有调度**。而管理台是给人看的，没理由为了
让盘前补数据、生成信号这些事发生就得一直开着浏览器后端。

这个入口把同一套调度逻辑跑成独立进程。计划表、补跑规则、任务白名单
全部复用 webui.scheduler 与 webui.registry —— 两处各写一套迟早会分叉，
而分叉的表现是「管理台上看到的计划和实际跑的不是一回事」。

## 与管理台的关系

两者读写同一份 webui/schedules.json：管理台负责改计划、看结果，
本进程负责执行。同时开着也不会重复触发 —— 调度前会检查该任务
是否已有实例在跑（jobs.running_jobs），且 last_run_date 按日期判重。

但**不要同时开两个本进程**：判重靠的是任务实例检查，两个进程在同一秒
扫描时可能都认为没在跑。用 --check 可以看当前状态。

## 用法

    python scripts/run_scheduler.py                # 前台运行，Ctrl+C 退出
    python scripts/run_scheduler.py --check        # 只打印计划表与状态，不运行
    python scripts/run_scheduler.py --run-now pre_market   # 立即触发某条计划

## 做成开机自启（Windows）

    schtasks /create /tn qmtquant-scheduler /sc onstart /rl highest ^
      /tr "D:\\qmt\\qmt\\.venv\\Scripts\\python.exe D:\\qmt\\qmt\\scripts\\run_scheduler.py"

崩溃自动重启需要额外的守护（如 NSSM）—— 单靠计划任务只保证开机启动一次。
"""
import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webui import jobs, scheduler  # noqa: E402
from webui.registry import TASK_BY_ID  # noqa: E402


def print_schedules() -> None:
    rows = scheduler.list_schedules()
    now = datetime.now()
    print(f"{'=' * 74}")
    print(f"  交易日程   {now:%Y-%m-%d %H:%M:%S}  "
          f"（周{scheduler.WEEKDAY_NAMES[now.weekday()]}）")
    print(f"{'=' * 74}")
    print(f"  {'时间':<7} {'计划':<14} {'任务':<22} {'状态':<8} {'今日'}")
    print(f"  {'-'*7} {'-'*14} {'-'*22} {'-'*8} {'-'*10}")
    for r in sorted(rows, key=lambda x: x["time"]):
        task = TASK_BY_ID.get(r["task_id"])
        tname = task.name if task else f"[未登记 {r['task_id']}]"
        state = "启用" if r.get("enabled") else "关闭"
        ran = "已跑" if r.get("ran_today") else "-"
        mark = " " if task else "!"
        print(f" {mark}{r['time']:<7} {r['name']:<14} {tname:<22} "
              f"{state:<8} {ran}")

    missing = [r for r in rows if r["task_id"] not in TASK_BY_ID]
    if missing:
        print(f"\n  [!] {len(missing)} 条计划指向未登记的任务，永远不会执行：")
        for r in missing:
            print(f"      {r['name']} -> {r['task_id']}")

    off = [r for r in rows if not r.get("enabled")]
    if off:
        print(f"\n  已关闭 {len(off)} 条（下单类计划默认关闭，"
              f"需在管理台显式打开）：")
        for r in off:
            print(f"      {r['time']} {r['name']}")

    running = jobs.running_jobs()
    if running:
        print(f"\n  正在运行 {len(running)} 个任务：")
        for j in running:
            print(f"      {j['task_id']}  started={j['started_at']}")


def main() -> int:
    p = argparse.ArgumentParser(description="qmtquant 定时调度守护")
    p.add_argument("--check", action="store_true",
                   help="只打印计划表与状态，不进入循环")
    p.add_argument("--run-now", metavar="SCHEDULE_ID",
                   help="立即触发某条计划（忽略时间与今日已跑）")
    args = p.parse_args()

    if args.check:
        print_schedules()
        return 0

    if args.run_now:
        try:
            job_id = scheduler.trigger_now(args.run_now)
        except RuntimeError as e:
            print(f"触发失败: {e}")
            return 1
        if job_id is None:
            print(f"计划不存在: {args.run_now}")
            return 1
        print(f"已触发，job_id={job_id}")
        return 0

    print_schedules()
    n = sum(1 for s in scheduler.list_schedules() if s["enabled"])
    print(f"\n{'=' * 74}")
    print(f"  调度守护已启动（{n} 条计划生效），Ctrl+C 退出")
    print(f"{'=' * 74}\n")

    scheduler.start_scheduler()

    running = True

    def _stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    # 管理台停服务时：Windows 发 CTRL_BREAK_EVENT（表现为 SIGBREAK），
    # 其他平台发 SIGTERM。两者都要走同一条优雅退出路径 ——
    # Windows 上 os.kill(pid, SIGTERM) 实际是 TerminateProcess，
    # 直接杀死、信号处理器不触发，所以那条路不能用。
    for _sig in ("SIGBREAK", "SIGTERM"):
        h = getattr(signal, _sig, None)
        if h is None:
            continue
        try:
            signal.signal(h, _stop)
        except (OSError, ValueError):
            pass

    from webui import services

    try:
        tick = 0
        while running:
            time.sleep(1)
            tick += 1
            # 每 30 秒写一次心跳：进程活着不等于调度线程还在转，
            # 只看进程会把「线程死了但进程还在」误判成健康
            if tick % 30 == 0 and scheduler.is_running():
                services.beat("scheduler")
    finally:
        print("\n正在停止调度 ...")
        scheduler.stop_scheduler()
        print("已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
