"""每日一键流水线 —— 数据更新 → ALSTM 推理 → PPO 择时 → 生成信号。

收盘后运行一次，自动完成全部步骤。
下单仍需手动确认（安全起见不自动执行）。

用法::

    python scripts/daily_pipeline.py
    python scripts/daily_pipeline.py --skip-download   # 跳过数据下载（已更新时）
    python scripts/daily_pipeline.py --auto-trade       # 自动下单（慎用）
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON = str(Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe")
ROOT = str(Path(__file__).resolve().parent.parent)


def run_step(name: str, cmd: list, timeout: int = 1800) -> bool:
    """运行一个步骤，返回是否成功。"""
    print(f"\n{'='*60}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {name}")
    print(f"{'='*60}\n")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, timeout=timeout,
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"\n  [OK] {name} 完成 ({elapsed:.0f}s)")
            return True
        else:
            print(f"\n  [FAIL] {name} 失败 (exit={result.returncode}, {elapsed:.0f}s)")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n  [FAIL] {name} 超时 ({timeout}s)")
        return False
    except Exception as e:
        print(f"\n  [FAIL] {name} 异常: {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="每日一键流水线")
    p.add_argument("--skip-download", action="store_true",
                    help="跳过数据下载（Qlib 数据已是最新时）")
    p.add_argument("--retrain-alstm", action="store_true",
                    help="重新训练 ALSTM（默认使用已有分数，无需每天重训）")
    p.add_argument("--auto-trade", action="store_true",
                    help="自动执行下单（默认只生成信号）")
    p.add_argument("--capital", type=float, default=500_000)
    args = p.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    t_start = time.time()

    print("=" * 60)
    print(f"  ALSTM + PPO 每日流水线")
    print(f"  日期: {today}")
    print(f"  本金: {args.capital:,.0f} 元")
    print("=" * 60)

    steps_ok = []

    # 步骤 1：下载最新行情
    if not args.skip_download:
        ok = run_step(
            "1/5 下载最新行情（QMT）",
            [PYTHON, "scripts/download_data.py",
             "--sector", "中证1000", "--intervals", "1d", "--resume"],
            timeout=600,
        )
        steps_ok.append(("下载行情", ok))
        if not ok:
            print("\n[WARN] 行情下载失败，可能 QMT 未启动。")
            print("  如果数据已是最新，可用 --skip-download 跳过")
            return 1

        # 步骤 2：导出 Qlib 格式
        ok = run_step(
            "2/5 导出 Qlib 格式",
            [PYTHON, "scripts/export_qlib.py",
             "--index", "000852.SH", "--start", "2016-01-01"],
            timeout=600,
        )
        steps_ok.append(("导出Qlib", ok))
        if not ok:
            return 1
    else:
        print("\n  跳过数据下载（--skip-download）")
        steps_ok.append(("下载行情", "跳过"))
        steps_ok.append(("导出Qlib", "跳过"))

    # 步骤 3：ALSTM 推理
    # 默认跳过重新训练，直接使用已有 scores.parquet。
    # 只有传 --retrain-alstm 才会重新训练（耗时 10-20 分钟且结果有随机性）。
    if args.retrain_alstm:
        ok = run_step(
            "3/5 ALSTM 重新训练+推理",
            [PYTHON, "scripts/run_alstm.py",
             "--market", "csi1000", "--index", "000852.SH"],
            timeout=1800,
        )
        steps_ok.append(("ALSTM推理", ok))
        if not ok:
            print("\n[WARN] ALSTM 推理失败，将使用上次的分数")
    else:
        scores_path = Path("reports/alstm_csi1000/scores.parquet")
        if scores_path.exists():
            import pandas as pd
            scores = pd.read_parquet(scores_path)
            latest = scores.index[-1].date()
            print(f"\n  使用已有 ALSTM 分数（最新: {latest}）")
            print(f"  如需重新训练，加 --retrain-alstm")
        else:
            print("\n  [WARN] 分数文件不存在，需要先训练一次")
            print(f"  运行: {PYTHON} scripts/run_alstm.py --market csi1000 --index 000852.SH")
            return 1
        steps_ok.append(("ALSTM推理", "使用缓存"))

    # 步骤 4：生成信号
    ok = run_step(
        "4/5 生成交易信号（ALSTM选股 + PPO择时）",
        [PYTHON, "scripts/generate_signal.py",
         "--date", today, "--capital", str(args.capital)],
        timeout=300,
    )
    steps_ok.append(("生成信号", ok))
    if not ok:
        print("\n[FAIL] 信号生成失败")
        return 1

    # 步骤 5：下单
    if args.auto_trade:
        ok = run_step(
            "5/5 执行下单",
            [PYTHON, "scripts/run_paper_trade.py"],
            timeout=120,
        )
        steps_ok.append(("执行下单", ok))
    else:
        print(f"\n{'='*60}")
        print(f"  5/5 信号已就绪，请手动执行下单：")
        print(f"{'='*60}")
        print(f"\n  预览:  {PYTHON} scripts/run_paper_trade.py --dry-run")
        print(f"  执行:  {PYTHON} scripts/run_paper_trade.py")
        steps_ok.append(("执行下单", "待手动"))

    # 汇总
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  流水线完成  耗时 {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")
    for name, status in steps_ok:
        if status is True:
            icon = "[OK]"
        elif status is False:
            icon = "[FAIL]"
        else:
            icon = "[--]"
        print(f"  {icon} {name}: {status}")

    # 显示信号摘要
    signal_file = Path("signals/target_latest.csv")
    if signal_file.exists():
        import csv
        with open(signal_file, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if rows:
            total_value = sum(float(r["target_value"]) for r in rows)
            total_weight = sum(float(r["weight"]) for r in rows)
            print(f"\n  今日信号: {len(rows)} 只股票")
            print(f"  总仓位: {total_weight:.1%}")
            print(f"  总金额: Y{total_value:,.0f}")
            print(f"  前 5 只:")
            for r in rows[:5]:
                print(f"    {r['vt_symbol']:>12s}  "
                      f"分数 {float(r['score']):+.4f}  "
                      f"Y{float(r['target_value']):,.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
