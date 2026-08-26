"""下载已退市股票的历史行情（akshare）。

为什么单独一个脚本：退市股在 QMT 中完全不存在，只能走 akshare，
而 akshare 速度不可控、偶发卡死，需要独立的超时与续传处理。

数据落到与 QMT 相同的目录结构，回测时两者可无缝混用。

用法：
    python scripts/download_delisted.py
    python scripts/download_delisted.py --resume     # 跳过已下载
    python scripts/download_delisted.py --since 2018-01-01
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import Interval  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402


def make_progress():
    start = time.time()

    def _p(done: int, total: int, symbol: str) -> None:
        elapsed = time.time() - start
        eta = (total - done) / (done / elapsed) if done and elapsed else 0
        filled = int(30 * done / total) if total else 0
        sys.stdout.write(f"\r  [{'#'*filled}{'-'*(30-filled)}] {done}/{total} "
                         f"{symbol:<14} ETA {eta/60:5.1f}min ")
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\n")
    return _p


def main() -> int:
    parser = argparse.ArgumentParser(description="下载退市股行情")
    parser.add_argument("--universe", default="universe_full.parquet")
    parser.add_argument("--since", default="2015-01-01",
                        help="只下载该日期之后退市的标的（更早的对近期回测无用）")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--interval", default="1d", choices=["1d", "1w", "1mon"])
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    uni_path = Path(cfg.data.store_dir) / "universe" / args.universe
    if not uni_path.exists():
        print(f"缺少标的池: {uni_path}")
        print("请先运行: python scripts/build_history_universe.py")
        return 1

    full = pd.read_parquet(uni_path)
    delisted = full[full["status"] == "delisted"].copy()
    if args.since:
        cutoff = pd.Timestamp(args.since)
        before = len(delisted)
        delisted = delisted[delisted["delist_date"] >= cutoff]
        print(f"按退市日 >= {args.since} 过滤：{before} -> {len(delisted)} 只")

    if delisted.empty:
        print("没有需要下载的退市股")
        return 0

    symbols = delisted["vt_symbol"].tolist()
    interval = Interval(args.interval)

    from qmtquant.datafeed.ak_feed import AkshareDataFeed
    feed = AkshareDataFeed(cfg.data.store_dir, cfg.data.dividend_type)

    print("=" * 60)
    print(f"退市股数量 : {len(symbols)}")
    print(f"周期       : {interval.value}")
    print(f"区间       : {args.start} ~ {args.end}")
    print(f"数据源     : akshare（QMT 无退市股数据）")
    print(f"预计耗时   : 约 {len(symbols)*1.2/60:.0f} 分钟（含限流间隔）")
    print("=" * 60)

    t0 = time.time()
    result = feed.download_history(symbols, args.start, args.end, interval,
                                   skip_existing=args.resume,
                                   progress=make_progress())

    print(f"\n完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"  成功 {len(result['ok'])} | 跳过 {len(result['skipped'])} "
          f"| 失败 {len(result['failed'])}")
    if result["failed"]:
        preview = ", ".join(result["failed"][:10])
        print(f"\n失败标的（可加 --resume 重跑）: {preview}"
              + (" ..." if len(result["failed"]) > 10 else ""))
        print("常见原因：退市过久上游已无数据、代码变更、接口临时不可用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
