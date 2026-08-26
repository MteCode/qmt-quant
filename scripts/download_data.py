"""历史数据下载入口。

需要 QMT 客户端已启动并登录，且 xtquant 可导入（见 docs/IMPLEMENTATION.md 1.2 节）。
数据落到 data/{周期}/{交易所}/{代码}.parquet，之后回测不再依赖客户端。

常用：
    # 沪深300 全部成分股，日线 + 周线 + 1分钟线
    python scripts/download_data.py --sector 沪深300 --intervals 1d,1w,1m --start 2020-01-01

    # 指定标的
    python scripts/download_data.py --symbols 000001.SZ,600519.SH --intervals 1d

    # 中断后续传（跳过已下载的）
    python scripts/download_data.py --sector 沪深300 --intervals 1m --resume

    # 只看本地库存
    python scripts/download_data.py --summary
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import MINUTE_INTERVALS, Interval  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize  # noqa: E402

#: 每只股票每种周期的粗略磁盘占用（MB），用于下载前预警
SIZE_HINT_MB = {
    Interval.MINUTE: 12.0,      # 1分钟线 5 年约 12MB/只
    Interval.MINUTE_5: 2.5,
    Interval.MINUTE_15: 0.9,
    Interval.MINUTE_30: 0.5,
    Interval.HOUR: 0.3,
    Interval.DAILY: 0.05,
    Interval.WEEKLY: 0.01,
    Interval.MONTHLY: 0.005,
}


def make_progress(interval: Interval):
    """单行刷新的进度条"""
    start_ts = time.time()

    def _progress(done: int, total: int, symbol: str) -> None:
        elapsed = time.time() - start_ts
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        bar_len = 30
        filled = int(bar_len * done / total) if total else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(
            f"\r  [{interval.value}] [{bar}] {done}/{total} "
            f"{symbol:<14} ETA {eta/60:5.1f}min "
        )
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\n")

    return _progress


def estimate_disk(symbols: int, intervals: list[Interval], years: float) -> float:
    """估算磁盘占用（MB）"""
    scale = years / 5.0
    return sum(SIZE_HINT_MB.get(iv, 0.1) * symbols * max(scale, 0.2) for iv in intervals)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="下载历史行情到本地",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--symbols", help="逗号分隔的标的代码")
    src.add_argument("--sector", help="板块名，如 沪深300 / 中证500 / 上证50 / 沪深A股")
    parser.add_argument("--intervals", default="1d",
                        help="逗号分隔的周期：1m,5m,15m,30m,1h,1d,1w,1mon")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2030-12-31")
    parser.add_argument("--resume", action="store_true", help="跳过已下载的标的")
    parser.add_argument("--summary", action="store_true", help="只显示本地库存并退出")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认提示")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    from qmtquant.datafeed.xt_feed import XtDataFeed
    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)

    # ---- 只看库存
    if args.summary:
        df = feed.summary()
        if df.empty:
            print("本地暂无数据")
        else:
            print(f"数据目录: {Path(cfg.data.store_dir).resolve()}\n")
            print(df.to_string(index=False))
        return 0

    # ---- 解析周期
    try:
        intervals = [Interval(s.strip()) for s in args.intervals.split(",") if s.strip()]
    except ValueError as e:
        print(f"无效的周期: {e}")
        return 1

    # 周线/月线依赖日线合成，自动把日线排到前面
    needs_daily = any(i in (Interval.WEEKLY, Interval.MONTHLY) for i in intervals)
    if needs_daily and Interval.DAILY not in intervals:
        print("[i] 周线/月线由日线合成，已自动加入日线下载")
        intervals.insert(0, Interval.DAILY)
    intervals.sort(key=lambda i: 0 if i == Interval.DAILY else 1)

    # ---- 解析标的
    if args.sector:
        try:
            vt_symbols = feed.get_sector_stocks(args.sector)
        except ImportError:
            print("未安装 xtquant，无法获取板块成分股。")
            print("请先按 docs/IMPLEMENTATION.md 1.2 节把 xtquant 装好。")
            return 1
        if not vt_symbols:
            print(f"板块「{args.sector}」未取到成分股。")
            print("请在 QMT 客户端中确认已下载该板块数据，且客户端处于登录状态。")
            return 1
    elif args.symbols:
        vt_symbols = [normalize(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        print("请用 --sector 或 --symbols 指定要下载的标的")
        return 1

    # ---- 下载前预警
    years = max((pd_years(args.start, args.end)), 0.2)
    est_mb = estimate_disk(len(vt_symbols), intervals, years)
    print("=" * 60)
    print(f"标的数量  : {len(vt_symbols)}"
          + (f"（板块：{args.sector}）" if args.sector else ""))
    print(f"周期      : {', '.join(i.value for i in intervals)}")
    print(f"时间区间  : {args.start} ~ {args.end}（约 {years:.1f} 年）")
    print(f"复权方式  : {cfg.data.dividend_type}")
    print(f"存储目录  : {Path(cfg.data.store_dir).resolve()}")
    print(f"预估占用  : 约 {est_mb/1024:.1f} GB" if est_mb > 1024
          else f"预估占用  : 约 {est_mb:.0f} MB")
    if any(i in MINUTE_INTERVALS for i in intervals):
        print()
        print("[!] 包含分钟级数据，注意：")
        print("    - 300 只股票的 1 分钟线下载通常需要 30~90 分钟，取决于网络与券商限速")
        print("    - QMT 客户端必须全程保持登录，中途掉线会导致部分标的失败")
        print("    - 中断后可加 --resume 续传")
    print("=" * 60)

    if not args.yes:
        try:
            if input("确认开始下载？[y/N] ").strip().lower() not in ("y", "yes"):
                print("已取消")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n已取消")
            return 0

    # ---- 逐周期下载
    all_failed: dict[str, list[str]] = {}
    t0 = time.time()
    for interval in intervals:
        print(f"\n>>> {interval.value}")
        try:
            result = feed.download_history(
                vt_symbols, args.start, args.end, interval,
                skip_existing=args.resume, progress=make_progress(interval),
            )
        except KeyboardInterrupt:
            print("\n\n已中断。下次加 --resume 可跳过已完成的标的。")
            return 130

        print(f"    成功 {len(result['ok'])} | 跳过 {len(result['skipped'])} "
              f"| 失败 {len(result['failed'])}")
        if result["failed"]:
            all_failed[interval.value] = result["failed"]

    # ---- 汇总
    print("\n" + "=" * 60)
    print(f"全部完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
    df = feed.summary()
    if not df.empty:
        print()
        print(df.to_string(index=False))

    if all_failed:
        print("\n以下标的下载失败：")
        for period, symbols in all_failed.items():
            preview = ", ".join(symbols[:8])
            more = f" ... 等 {len(symbols)} 只" if len(symbols) > 8 else ""
            print(f"  [{period}] {preview}{more}")
        print("\n失败常见原因：新股上市时间晚于起始日、已退市、当日停牌无数据。")
        print("可重跑本命令（不加 --resume）重试失败项。")
    return 0


def pd_years(start: str, end: str) -> float:
    """区间跨度（年），end 超过今天则按今天算"""
    import pandas as pd
    s = pd.Timestamp(start)
    e = min(pd.Timestamp(end), pd.Timestamp.today())
    return max((e - s).days / 365.0, 0.0)


if __name__ == "__main__":
    raise SystemExit(main())
