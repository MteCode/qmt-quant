"""下载中证1000全部历史成分股的日线数据。

从 index_weight_000852.SH.csv 提取成分股并集，
跳过本地已有日线的标的，调用 download_data 的核心逻辑下载缺失的。

用法::

    python scripts/download_csi1000.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from qmtquant.config import LOG_DIR, get_config
from qmtquant.core.constants import Interval
from qmtquant.utils.logger import setup_logging


def main() -> int:
    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    store = Path(cfg.data.store_dir)

    weight_csv = store / "universe" / "index_weight_000852.SH.csv"
    if not weight_csv.exists():
        print(f"缺少 {weight_csv}")
        return 1

    w = pd.read_csv(weight_csv, parse_dates=["date"])
    all_symbols = sorted(w["symbol"].unique())
    print(f"中证1000 历史成分并集: {len(all_symbols)} 只")

    missing = []
    for s in all_symbols:
        code, _, ex = s.rpartition(".")
        if not (store / "1d" / ex / f"{code}.parquet").exists():
            missing.append(s)

    print(f"本地已有: {len(all_symbols) - len(missing)} 只")
    print(f"需要下载: {len(missing)} 只")

    if not missing:
        print("全部已有，无需下载")
        return 0

    from qmtquant.datafeed.xt_feed import XtDataFeed
    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)

    start = "2015-01-01"
    end = "2030-12-31"
    interval = Interval.DAILY

    print(f"\n开始下载 {len(missing)} 只标的日线 ({start} ~)")
    print("=" * 60)

    t0 = time.time()
    batch_size = 200
    total_ok = 0
    total_fail = 0

    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(missing) + batch_size - 1) // batch_size
        print(f"\n批次 {batch_num}/{total_batches}  ({len(batch)} 只)")

        def progress(done, total, symbol):
            filled = int(28 * done / total)
            elapsed = time.time() - t0
            sys.stdout.write(
                f"\r  [{'#' * filled}{'-' * (28 - filled)}] "
                f"{done}/{total} {symbol:<14}")
            sys.stdout.flush()
            if done == total:
                sys.stdout.write("\n")

        try:
            result = feed.download_history(
                batch, start, end, interval,
                skip_existing=True, progress=progress)
            total_ok += len(result["ok"])
            total_fail += len(result.get("failed", []))
            if result.get("failed"):
                print(f"  失败 {len(result['failed'])} 只: "
                      f"{result['failed'][:5]}")
        except KeyboardInterrupt:
            print("\n\n已中断。重新运行即可续传（自动跳过已有的）。")
            return 130

    elapsed = time.time() - t0
    print(f"\n完成。成功 {total_ok} 只，失败 {total_fail} 只，"
          f"耗时 {elapsed / 60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
