"""行情更新 —— 下载最新日线并导出为 Qlib 格式。

下载与导出必须成对执行：下载只更新 `data/1d/` 下的原始 parquet，
模型读的是 `data/qlib_data/` 下的 bin 文件，不导出等于没更新。

## 盘前 / 盘后的区别

两种模式做的是同一件事，差别在**预期**与**校验**：

- ``--mode post``（盘后，收盘后运行）：当日日线已定稿，期望能取到今天的数据。
  取不到会明确报警 —— 多半是 QMT 没启动或当天休市。
- ``--mode pre``（盘前，开盘前运行）：补齐昨夜遗漏。若昨天盘后任务失败，
  这一次是最后的补救机会，因此**即使数据看起来是最新的也照常执行**。

## 数据一致性

`export_qlib.py --index` 只重写该指数的历史成分股，但 `features/` 下可能
存有以前导出其它指数留下的标的。日历会按本次导出的标的重建，
没被重写的标的就会落后于新日历。

本脚本跑完会检查这种撕裂状态并报告落后标的数。落后不影响只用该指数成分的
训练，但跨指数使用时会读到过期数据。用 ``--all`` 可全量导出消除撕裂，
代价是耗时显著增加。

用法::

    python scripts/update_market_data.py --mode post
    python scripts/update_market_data.py --mode pre
    python scripts/update_market_data.py --mode post --all
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def run(desc: str, cmd: list, timeout: int) -> bool:
    print(f"\n{'=' * 58}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {desc}")
    print(f"{'=' * 58}\n")
    t0 = time.time()
    try:
        import os
        r = subprocess.run(
            cmd, cwd=str(ROOT), timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        dt = time.time() - t0
        if r.returncode == 0:
            print(f"\n  [OK] {desc}（{dt:.0f}s）")
            return True
        print(f"\n  [FAIL] {desc} exit={r.returncode}（{dt:.0f}s）")
        return False
    except subprocess.TimeoutExpired:
        print(f"\n  [FAIL] {desc} 超时（{timeout}s）")
        return False
    except OSError as e:
        print(f"\n  [FAIL] {desc}: {e}")
        return False


def check_freshness(index: str) -> dict:
    """检查 Qlib 数据的新鲜度与一致性。"""
    import numpy as np

    from qmtquant.config import get_config

    cfg = get_config()
    qdir = Path(cfg.data.store_dir) / "qlib_data"
    cal_file = qdir / "calendars" / "day.txt"
    feat_dir = qdir / "features"
    if not cal_file.exists() or not feat_dir.exists():
        return {}

    cal = cal_file.read_text(encoding="utf-8").split()
    n_cal = len(cal)
    stale = fresh = total = 0
    for d in feat_dir.iterdir():
        p = d / "close.day.bin"
        if not p.exists():
            continue
        total += 1
        n = (p.stat().st_size - 4) // 4
        start = int(np.fromfile(p, dtype=np.float32, count=1)[0])
        if start + n - 1 >= n_cal - 1:
            fresh += 1
        else:
            stale += 1

    return {"last_date": cal[-1] if cal else "?", "days": n_cal,
            "total": total, "fresh": fresh, "stale": stale}


def main() -> int:
    p = argparse.ArgumentParser(description="行情更新")
    p.add_argument("--mode", choices=["pre", "post"], default="post",
                    help="pre=盘前补齐，post=盘后更新")
    p.add_argument("--sector", default="中证1000")
    p.add_argument("--index", default="000852.SH")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--all", action="store_true",
                    help="导出本地全部标的，消除跨指数的数据撕裂（较慢）")
    p.add_argument("--skip-download", action="store_true",
                    help="只重新导出，不下载")
    args = p.parse_args()

    sys.path.insert(0, str(ROOT))

    label = "盘前补齐" if args.mode == "pre" else "盘后更新"
    print("=" * 58)
    print(f"  行情更新 · {label}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  板块 {args.sector} · 指数 {args.index}")
    print("=" * 58)

    before = check_freshness(args.index)
    if before:
        print(f"\n更新前: 日历至 {before['last_date']}，"
              f"{before['fresh']}/{before['total']} 个标的已跟上")

    ok = True
    if not args.skip_download:
        ok = run("下载最新日线（QMT）",
                 [str(PYTHON), "scripts/download_data.py",
                  "--sector", args.sector, "--intervals", "1d",
                  "--resume", "--yes"],
                 timeout=1800)
        if not ok:
            print("\n下载失败。常见原因：miniQMT 未启动，或当日休市无数据。")
            if args.mode == "post":
                return 1
            print("盘前模式下继续尝试导出已有数据。")

    # 清洗必须紧跟下载。跳过这一步，脏数据会一路进到训练集 ——
    # 曾有 25000 多行非正价格因此进了模型
    if not run("清洗数据", [str(PYTHON), "scripts/clean_data.py"],
               timeout=1800):
        print("\n清洗失败。导出会回退到原始层就地清洗，结果仍是干净的，"
              "但每次导出都要重清。")

    export_cmd = [str(PYTHON), "scripts/export_qlib.py",
                  "--start", args.start]
    export_cmd += ["--all"] if args.all else ["--index", args.index]
    if not run("导出 Qlib 格式", export_cmd, timeout=1800):
        return 1

    after = check_freshness(args.index)
    if after:
        print(f"\n{'=' * 58}")
        print("  数据状态")
        print(f"{'=' * 58}")
        print(f"  日历最后一天 : {after['last_date']}"
              f"（共 {after['days']} 个交易日）")
        print(f"  标的总数     : {after['total']}")
        print(f"  已跟上日历   : {after['fresh']}")
        if after["stale"]:
            print(f"  落后于日历   : {after['stale']}  <- 注意")
            print(f"\n  这些标的多半是以前导出其它指数留下的，本次未被重写。")
            print(f"  只用 {args.index} 成分训练不受影响；")
            print(f"  跨指数使用会读到过期数据，可用 --all 全量导出消除。")
        else:
            print(f"  落后于日历   : 0  全部一致")

        if before and after["last_date"] == before["last_date"]:
            if args.mode == "post":
                print(f"\n  [注意] 日历最后一天未推进，今日可能休市或数据未更新")

    print(f"\n完成 {datetime.now().strftime('%H:%M:%S')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
