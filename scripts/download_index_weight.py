"""下载指数历史成分股（Tushare Pro，需 2000 积分）。

这是**消除成分股前视偏差**的关键数据。没有它，任何选股回测的收益
都由偏差主导而非策略主导 —— 实测同一组均值回归参数，
加不加纳入日过滤，总收益从 +320.33% 变成 -4.49%。

产出 ``data/universe/index_weight_{指数代码}.csv``，
格式即 ``HistoricalUniverse`` 直接可读的 ``date,symbol,weight``。

用法::

    # 沪深300 全历史（首次跑，约 10 分钟）
    python scripts/download_index_weight.py --index 000300.SH --start 2005-01-01

    # 顺带把另外两个也拉了（同一份积分，不额外花钱）
    python scripts/download_index_weight.py --index 000905.SH --index 000852.SH

    # 增量更新：只补最后一期之后的月份
    python scripts/download_index_weight.py --index 000300.SH --update
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.datafeed.tushare_feed import (  # noqa: E402
    INDEX_CODES,
    TushareError,
    TushareFeed,
)
from qmtquant.utils.logger import setup_logging  # noqa: E402


def out_path(store_dir: str, index_code: str) -> Path:
    d = Path(store_dir) / "universe"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"index_weight_{index_code}.csv"


def download_one(feed: TushareFeed, index_code: str, start: str, end: str,
                 path: Path, update: bool) -> int:
    """拉一个指数，返回新增行数"""
    existing = pd.DataFrame()
    if update and path.exists():
        existing = pd.read_csv(path, parse_dates=["date"])
        if not existing.empty:
            last = existing["date"].max()
            # 从最后一期所在月的次月开始，避免重复拉整段历史
            start = (last + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
            print(f"  增量模式：已有 {len(existing):,} 行，最后一期 "
                  f"{last.date()}，从 {start} 续拉")
            if pd.Timestamp(start) > pd.Timestamp(end):
                print("  已是最新，无需更新")
                return 0

    df = feed.index_weight(index_code, start, end)
    if df.empty:
        print("  未取到数据")
        return 0

    added = len(df)
    if not existing.empty:
        df = (pd.concat([existing, df], ignore_index=True)
                .drop_duplicates(subset=["date", "symbol"])
                .sort_values(["date", "symbol"])
                .reset_index(drop=True))
        added = len(df) - len(existing)

    df.to_csv(path, index=False, encoding="utf-8-sig")

    periods = df["date"].nunique()
    symbols = df["symbol"].nunique()
    print(f"  {len(df):,} 行（新增 {added:,}） · {periods} 期名单 · "
          f"历史出现过 {symbols} 只标的")
    print(f"  区间 {df['date'].min().date()} ~ {df['date'].max().date()}")
    # 这个数字就是偏差的规模：远大于成分数，说明调进调出很频繁
    latest = df[df["date"] == df["date"].max()]
    print(f"  最新一期 {len(latest)} 只，历史累计比它多 "
          f"{symbols - len(latest)} 只 —— 这些正是当前快照会漏掉的")
    print(f"  已写入 {path}")
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="下载指数历史成分股")
    parser.add_argument("--index", action="append", default=None,
                        help="指数代码，可重复。默认 000300.SH")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None, help="默认今天")
    parser.add_argument("--update", action="store_true",
                        help="增量模式，只补最后一期之后的月份")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    indexes = args.index or ["000300.SH"]
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        feed = TushareFeed(cfg.tushare)
    except TushareError as e:
        print(f"[!] {e}")
        return 1

    total = 0
    for code in indexes:
        name = INDEX_CODES.get(code, code)
        print(f"\n=== {code} {name} ===")
        try:
            total += download_one(feed, code, args.start, end,
                                  out_path(cfg.data.store_dir, code),
                                  args.update)
        except TushareError as e:
            print(f"  [!] 失败：{e}")
            return 1

    print(f"\n完成，共新增 {total:,} 行。")
    print("接下来可以用它跑无偏回测：")
    print("  python scripts/run_portfolio_backtest.py --universe-csv "
          f"{out_path(cfg.data.store_dir, indexes[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
