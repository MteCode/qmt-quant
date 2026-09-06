"""盘中行情增量更新 —— 全市场 1m/5m K 线实时追加。

## 与 update_market_data.py 的区别

update_market_data.py 是**盘后**的全量同步，跑一次几十分钟，
更新日线并导出 Qlib 格式，供隔夜训练和次日选股用。

本脚本是**盘中**的增量追加，每次只拉最新几根 bar，几秒到一两分钟跑完，
供日内策略实时计算特征和信号。

## 数据流

    xtdata（QMT 实时行情）
        │  update_bars()：增量拉取，追加到已有 parquet
        ▼
    data/1m/{SSE,SZSE}/     原始层，每标的一个 parquet
        │  clean_data.py --intervals 1m：增量清洗
        ▼
    data/clean/1m/{SSE,SZSE}/  清洗层，日内策略读这里

## 用法

    # 全市场 1m 增量更新（盘中每分钟跑一次）
    python scripts/update_intraday.py

    # 同时更新 1m 和 5m
    python scripts/update_intraday.py --intervals 1m,5m

    # 只更新指定标的
    python scripts/update_intraday.py --symbols 600519.SH 000001.SZ

    # 更新后自动清洗
    python scripts/update_intraday.py --clean

    # 持续运行模式：每隔 N 秒自动刷新，盘中自动启停
    python scripts/update_intraday.py --loop 60

## 持续运行模式

``--loop N`` 让脚本常驻，每 N 秒执行一轮更新。只在交易时段
（09:15~11:32, 12:58~15:02）工作，其余时间休眠。
适合挂在后台配合日内策略使用。

注意：持续模式下 miniQMT 必须保持运行。如果 QMT 客户端断线，
脚本会报错但不会退出，等恢复后继续。
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def get_symbols(args) -> list[str]:
    """确定要更新的标的列表。"""
    if args.symbols:
        from qmtquant.utils.symbol import normalize
        return [normalize(s) for s in args.symbols]

    # 全市场：优先 QMT 板块，兜底 market.db
    try:
        from xtquant import xtdata
        codes = xtdata.get_stock_list_in_sector(args.sector) or []
    except ImportError:
        codes = []

    if codes:
        from qmtquant.utils.symbol import from_xt_symbol
        symbols = []
        for c in codes:
            try:
                symbols.append(from_xt_symbol(c))
            except (KeyError, ValueError):
                pass
        return sorted(symbols)

    # 兜底：从本地 1m 目录取已有标的
    from qmtquant.config import get_config
    store = Path(get_config().data.store_dir)
    symbols = []
    for ex_dir in (store / "1m").iterdir():
        if ex_dir.is_dir():
            for f in ex_dir.glob("*.parquet"):
                symbols.append(f"{f.stem}.{ex_dir.name}")
    return sorted(symbols)


def is_trading_time() -> bool:
    """当前是否在交易时段（含集合竞价前后缓冲）。"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    # 09:15~11:32（上午含集合竞价和收盘缓冲）
    # 12:58~15:02（下午含开盘前和收盘后缓冲）
    return (915 <= t <= 1132) or (1258 <= t <= 1502)


def run_update(symbols: list[str], intervals: list, feed, clean: bool,
               store: Path) -> dict:
    """执行一轮增量更新。"""
    from qmtquant.core.constants import Interval

    summary = {}
    t0 = time.time()

    for interval in intervals:
        r = feed.update_bars(symbols, interval=interval)
        ok = len(r.get("ok", []))
        fail = len(r.get("failed", []))
        skip = len(r.get("skipped", []))
        summary[interval.value] = {"ok": ok, "failed": fail, "skipped": skip}
        if ok or fail:
            print(f"  [{interval.value}] 更新 {ok}，跳过 {skip}，失败 {fail}")
        elif skip:
            print(f"  [{interval.value}] 无新数据（{skip} 只均已最新）")

    elapsed = time.time() - t0

    if clean and any(s["ok"] > 0 for s in summary.values()):
        clean_intervals = [iv.value for iv in intervals
                           if summary.get(iv.value, {}).get("ok", 0) > 0]
        for iv in clean_intervals:
            print(f"  清洗 {iv}...")
            try:
                subprocess.run(
                    [str(PYTHON), "scripts/clean_data.py",
                     "--intervals", iv, "--types", "bars"],
                    cwd=str(ROOT), timeout=600,
                    capture_output=True)
            except (subprocess.TimeoutExpired, OSError) as e:
                print(f"  清洗 {iv} 失败: {e}")

    summary["elapsed"] = round(elapsed, 1)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="盘中行情增量更新")
    p.add_argument("--sector", default="沪深A股",
                   help="板块名称（默认沪深A股）")
    p.add_argument("--intervals", default="1m",
                   help="逗号分隔的周期：1m,5m,15m（默认 1m）")
    p.add_argument("--symbols", nargs="*", default=[],
                   help="只更新指定标的，例如 600519.SH 000001.SZ")
    p.add_argument("--clean", action="store_true",
                   help="更新后自动清洗到 data/clean/")
    p.add_argument("--loop", type=int, default=0,
                   help="持续运行，每 N 秒执行一轮（0=单次运行）")
    args = p.parse_args()

    from qmtquant.config import get_config
    from qmtquant.core.constants import Interval
    from qmtquant.datafeed.xt_feed import XtDataFeed

    try:
        intervals = [Interval(s.strip()) for s in args.intervals.split(",")
                     if s.strip()]
    except ValueError as e:
        print(f"无效的周期: {e}")
        return 1

    cfg = get_config()
    store = Path(cfg.data.store_dir)
    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)

    print("=" * 58)
    print("  盘中行情增量更新")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  周期 {args.intervals}")
    if args.loop:
        print(f"  持续模式：每 {args.loop} 秒一轮")
    print("=" * 58)

    symbols = get_symbols(args)
    if not symbols:
        print("无标的可更新 —— 请确认 miniQMT 已启动或 data/1m/ 下有数据")
        return 1
    print(f"\n标的 {len(symbols)} 只")

    if not args.loop:
        # 单次运行
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始更新")
        summary = run_update(symbols, intervals, feed, args.clean, store)
        print(f"\n完成，耗时 {summary['elapsed']}s")
        return 0

    # 持续运行模式
    print(f"\n进入持续模式，Ctrl+C 退出")
    round_n = 0
    try:
        while True:
            if not is_trading_time():
                now = datetime.now()
                t = now.hour * 100 + now.minute
                if t < 915:
                    wait_msg = "等待开盘"
                elif 1132 < t < 1258:
                    wait_msg = "午休"
                else:
                    wait_msg = "已收盘"
                print(f"\r  {now.strftime('%H:%M:%S')} {wait_msg}，"
                      f"下次检查 {args.loop}s 后", end="", flush=True)
                time.sleep(args.loop)
                continue

            round_n += 1
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"\n[{ts}] 第 {round_n} 轮")
            run_update(symbols, intervals, feed, args.clean, store)
            time.sleep(args.loop)
    except KeyboardInterrupt:
        print(f"\n\n已停止，共运行 {round_n} 轮")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
