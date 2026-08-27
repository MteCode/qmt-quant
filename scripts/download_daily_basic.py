"""下载每日估值因子（Tushare daily_basic，需 2000 积分）。

## 为什么这批数据值得单独拉

财报数据有公告日滞后（茅台年报平均滞后 100 天以上），用它做因子
必须小心 point-in-time。而 `daily_basic` 是**逐日快照** ——
2024-01-15 那天的 PE 就是当天盘后算出来的，不存在前视问题。

覆盖 PE/PE_TTM/PB/PS/股息率/换手率/市值 等，是估值类因子的直接来源。

## 落盘方式

按年分文件：``data/factor/daily_basic/{年份}.parquet``。

不合成一个大文件的原因：全市场 5500 只 × 2600 个交易日 ≈ 1400 万行，
单文件读写都慢，而因子研究通常只关心某几年。

## 断点续传

2600 次请求跑十几分钟，中途断网、限流、关机都可能发生。
脚本记录已完成的交易日，``--resume`` 时跳过 —— 不加也会自动跳过
已在 parquet 里的日期，重复跑是安全的。

用法::

    python scripts/download_daily_basic.py --start 2016-01-01
    python scripts/download_daily_basic.py --start 2016-01-01 --resume
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.datafeed.tushare_feed import TushareError, TushareFeed  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402


def store_dir(cfg) -> Path:
    d = Path(cfg.data.store_dir) / "factor" / "daily_basic"
    d.mkdir(parents=True, exist_ok=True)
    return d


def existing_dates(root: Path) -> set[pd.Timestamp]:
    """已落盘的交易日。重复跑安全的关键。"""
    done: set[pd.Timestamp] = set()
    for p in sorted(root.glob("*.parquet")):
        try:
            col = pd.read_parquet(p, columns=["trade_date"])
        except Exception as e:  # noqa: BLE001 —— 损坏的分片不应中断整个任务
            print(f"  [!] {p.name} 读取失败（将重下该年）：{e}")
            continue
        done |= set(pd.to_datetime(col["trade_date"]).unique())
    return done


def write_year(root: Path, year: int, frames: list[pd.DataFrame]) -> int:
    """把某一年的数据合并写回。已有数据会被合并而非覆盖。"""
    if not frames:
        return 0
    path = root / f"{year}.parquet"
    df = pd.concat(frames, ignore_index=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = (df.drop_duplicates(subset=["trade_date", "symbol"])
            .sort_values(["trade_date", "symbol"])
            .reset_index(drop=True))
    df.to_parquet(path, index=False)
    return len(df)


def main() -> int:
    p = argparse.ArgumentParser(description="下载每日估值因子")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None, help="默认今天")
    p.add_argument("--resume", action="store_true",
                   help="跳过已下载的交易日（不加也会自动跳过）")
    p.add_argument("--flush-every", type=int, default=60,
                   help="每 N 个交易日落盘一次，防止中断丢进度")
    args = p.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    root = store_dir(cfg)
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        feed = TushareFeed(cfg.tushare)
        print("拉取交易日历...")
        dates = feed.trade_dates(args.start, end)
    except TushareError as e:
        print(f"[!] {e}")
        return 1

    if not dates:
        print("区间内没有交易日")
        return 1

    done = existing_dates(root)
    todo = [d for d in dates if d not in done]
    print(f"区间 {args.start} ~ {end}：{len(dates)} 个交易日，"
          f"已有 {len(dates) - len(todo)}，待下载 {len(todo)}")
    if not todo:
        print("已是最新，无需下载")
        return 0

    started = time.time()
    buffers: dict[int, list[pd.DataFrame]] = {}
    ok = fail = 0

    for i, d in enumerate(todo, 1):
        try:
            df = feed.daily_basic(d)
        except TushareError as e:
            print(f"  [!] {d.date()} 失败：{e}")
            fail += 1
            continue
        if df.empty:
            # 非交易日或数据未更新，不算失败但要记下来
            fail += 1
            continue
        buffers.setdefault(d.year, []).append(df)
        ok += 1

        if i % args.flush_every == 0 or i == len(todo):
            for year, frames in buffers.items():
                write_year(root, year, frames)
            buffers.clear()
            elapsed = time.time() - started
            eta = elapsed / i * (len(todo) - i)
            print(f"  {i}/{len(todo)}  {d.date()}  "
                  f"已用 {elapsed / 60:.1f}min  ETA {eta / 60:.1f}min")

    print(f"\n完成：成功 {ok}，失败/空 {fail}，"
          f"耗时 {(time.time() - started) / 60:.1f} 分钟")

    total = 0
    for path in sorted(root.glob("*.parquet")):
        n = len(pd.read_parquet(path, columns=["trade_date"]))
        total += n
        print(f"  {path.name}  {n:>9,} 行")
    print(f"  合计 {total:,} 行，{sum(f.stat().st_size for f in root.glob('*.parquet')) / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
