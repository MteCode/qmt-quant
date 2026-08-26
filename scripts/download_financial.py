"""下载财务数据（QMT 数据源，本地取数不限流）。

数据落到 data/financial/{报表}/{交易所}/{代码}.parquet。

⚠ 所有查询请走 `FinancialStore.get_asof()` / `get_panel()`，它们按**公告日**过滤。
直接用报告期取数会引入巨大前视偏差：实测年报公告日比报告期晚 90~110 天。

用法：
    python scripts/download_financial.py --sector 沪深300
    python scripts/download_financial.py --symbols 600519.SH --tables Income,Balance
    python scripts/download_financial.py --summary
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.datafeed.financial import (  # noqa: E402
    DEFAULT_TABLES,
    FinancialStore,
)
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize  # noqa: E402


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
    parser = argparse.ArgumentParser(description="下载财务数据")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--sector", default=None, help="板块名，如 沪深300")
    src.add_argument("--symbols", default=None, help="逗号分隔的标的代码")
    parser.add_argument("--tables", default=None,
                        help=f"逗号分隔的报表名，默认 {','.join(DEFAULT_TABLES)}")
    parser.add_argument("--resume", action="store_true", help="跳过已下载的标的")
    parser.add_argument("--summary", action="store_true", help="只看本地库存")
    parser.add_argument("--yes", "-y", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    store = FinancialStore(cfg.data.store_dir)

    if args.summary:
        df = store.summary()
        if df.empty:
            print("本地暂无财务数据")
        else:
            print(f"数据目录: {store.root.resolve()}\n")
            print(df.to_string(index=False))
        return 0

    tables = ([t.strip() for t in args.tables.split(",") if t.strip()]
              if args.tables else DEFAULT_TABLES)

    if args.sector:
        from qmtquant.datafeed.xt_feed import XtDataFeed
        feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
        vt_symbols = feed.get_sector_stocks(args.sector)
        if not vt_symbols:
            print(f"板块「{args.sector}」未取到成分股")
            return 1
    elif args.symbols:
        vt_symbols = [normalize(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        print("请用 --sector 或 --symbols 指定标的")
        return 1

    print("=" * 60)
    print(f"标的数量 : {len(vt_symbols)}"
          + (f"（板块：{args.sector}）" if args.sector else ""))
    print(f"报表     : {', '.join(tables)}")
    print(f"存储目录 : {store.root.resolve()}")
    print("数据源   : QMT（本地取数，不受限流影响）")
    print("=" * 60)

    if not args.yes:
        try:
            if input("确认开始？[y/N] ").strip().lower() not in ("y", "yes"):
                print("已取消")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n已取消")
            return 0

    def dl_progress(finished: int, dl_total: int) -> None:
        if dl_total:
            pct = finished / dl_total
            filled = int(30 * pct)
            sys.stdout.write(f"\r  下载 [{'#'*filled}{'-'*(30-filled)}] "
                             f"{finished}/{dl_total} ")
            sys.stdout.flush()
            if finished >= dl_total:
                sys.stdout.write("\n")

    t0 = time.time()
    try:
        result = store.download(vt_symbols, tables, skip_existing=args.resume,
                                progress=make_progress(),
                                download_progress=dl_progress)
    except KeyboardInterrupt:
        print("\n已中断，下次加 --resume 续传")
        return 130

    print(f"\n完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"  成功 {len(result['ok'])} | 跳过 {len(result['skipped'])} "
          f"| 失败 {len(result['failed'])}")

    df = store.summary()
    if not df.empty:
        print()
        print(df.to_string(index=False))

    if result["failed"]:
        preview = ", ".join(result["failed"][:10])
        print(f"\n失败标的: {preview}"
              + (" ..." if len(result["failed"]) > 10 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
