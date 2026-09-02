"""定时交易脚本 -- 开盘前更新数据生成信号，开盘后自动下单。

时间线：
  09:00  更新数据 + 导出 Qlib
  09:15  生成交易信号（ALSTM + PPO）
  09:31  自动下单（等开盘 1 分钟再下，避免集合竞价）

前置条件：
  - miniQMT 已启动并登录
  - 在交易日运行

用法::

    python scripts/scheduled_trade.py              # 正常定时执行
    python scripts/scheduled_trade.py --dry-run     # 到点只预览不下单
    python scripts/scheduled_trade.py --now         # 立即执行全部步骤（不等时间）
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PYTHON = str(Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe")
ROOT = str(Path(__file__).resolve().parent.parent)

SCHEDULE = [
    ("09:00", "update_data",    "更新行情数据"),
    ("09:10", "export_qlib",    "导出 Qlib 格式"),
    ("09:15", "generate_signal","生成交易信号"),
    ("09:31", "execute_trade",  "执行下单"),
]


def wait_until(target_time: str):
    """等待到指定时间 (HH:MM)。"""
    now = datetime.now()
    h, m = map(int, target_time.split(":"))
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        return
    diff = (target - now).total_seconds()
    print(f"\n  等待到 {target_time} ... ({diff/60:.0f} 分钟后)")
    while datetime.now() < target:
        remaining = (target - datetime.now()).total_seconds()
        mins, secs = divmod(int(remaining), 60)
        print(f"\r  倒计时 {mins:02d}:{secs:02d}  ", end="", flush=True)
        time.sleep(1)
    print()


def run_cmd(desc: str, cmd: list, timeout: int = 600) -> bool:
    print(f"\n{'='*50}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {desc}")
    print(f"{'='*50}\n")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, timeout=timeout)
        elapsed = time.time() - t0
        if r.returncode == 0:
            print(f"\n  [OK] {desc} ({elapsed:.0f}s)")
            return True
        print(f"\n  [FAIL] {desc} (exit={r.returncode}, {elapsed:.0f}s)")
        return False
    except subprocess.TimeoutExpired:
        print(f"\n  [FAIL] {desc} -- timeout ({timeout}s)")
        return False
    except Exception as e:
        print(f"\n  [FAIL] {desc} -- {e}")
        return False


def is_trading_day() -> bool:
    """简单判断：周一到周五视为交易日（不含节假日）。"""
    return datetime.now().weekday() < 5


def main():
    p = argparse.ArgumentParser(description="定时交易脚本")
    p.add_argument("--dry-run", action="store_true",
                    help="到下单时只预览，不实际执行")
    p.add_argument("--now", action="store_true",
                    help="立即执行全部步骤，不等定时")
    p.add_argument("--capital", type=float, default=500_000)
    args = p.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    print("=" * 50)
    print(f"  ALSTM + PPO 定时交易")
    print(f"  日期: {today}")
    print(f"  模式: {'DRY RUN' if args.dry_run else '实盘下单'}")
    print("=" * 50)

    if not is_trading_day():
        print(f"\n  今天是周末，非交易日，退出。")
        return 0

    now = datetime.now()
    if not args.now and now.hour >= 15:
        print(f"\n  已过 15:00 收盘，今日无法交易，退出。")
        return 0

    results = []

    for sched_time, step, desc in SCHEDULE:
        if not args.now:
            wait_until(sched_time)

        if step == "update_data":
            ok = run_cmd(desc, [
                PYTHON, "scripts/download_data.py",
                "--sector", "中证1000", "--intervals", "1d", "--resume",
            ], timeout=600)
            results.append((desc, ok))
            if not ok:
                print("\n  [WARN] 数据下载失败，尝试继续（用已有数据）...")

        elif step == "export_qlib":
            ok = run_cmd(desc, [
                PYTHON, "scripts/export_qlib.py",
                "--index", "000852.SH", "--start", "2016-01-01",
            ], timeout=600)
            results.append((desc, ok))

        elif step == "generate_signal":
            ok = run_cmd(desc, [
                PYTHON, "scripts/generate_signal.py",
                "--date", today, "--capital", str(args.capital),
            ], timeout=300)
            results.append((desc, ok))
            if not ok:
                print("\n  [FAIL] 信号生成失败，取消下单")
                break

        elif step == "execute_trade":
            if args.dry_run:
                ok = run_cmd(desc + " (DRY RUN)", [
                    PYTHON, "scripts/run_paper_trade.py", "--dry-run",
                ], timeout=120)
            else:
                ok = run_cmd(desc, [
                    PYTHON, "scripts/run_paper_trade.py",
                ], timeout=120)
            results.append((desc, ok))

    # 汇总
    print(f"\n{'='*50}")
    print(f"  执行完毕 {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")
    for desc, ok in results:
        icon = "[OK]" if ok else "[FAIL]"
        print(f"  {icon} {desc}")

    # 显示信号摘要
    signal_file = Path("signals/target_latest.csv")
    if signal_file.exists():
        import csv
        with open(signal_file, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if rows:
            total_value = sum(float(r["target_value"]) for r in rows)
            total_weight = sum(float(r["weight"]) for r in rows)
            print(f"\n  信号: {len(rows)} 只股票, 仓位 {total_weight:.1%}, 金额 {total_value:,.0f} 元")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
