"""历史数据下载入口。

需要 QMT 客户端已启动登录，且 xtquant 可导入。
数据落到 data/{周期}/{交易所}/{代码}.parquet，之后回测不再依赖客户端。

用法：
    python scripts/download_data.py --symbols 000001.SZ,600519.SH --start 2020-01-01
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import Interval  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="下载历史行情到本地")
    parser.add_argument("--symbols", required=True, help="逗号分隔的标的代码")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2030-12-31")
    parser.add_argument("--interval", default="1d",
                        choices=[i.value for i in Interval if i != Interval.TICK])
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    try:
        from qmtquant.datafeed.xt_feed import XtDataFeed
    except ImportError as e:
        print(f"导入数据源失败: {e}")
        return 1

    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    vt_symbols = [normalize(s.strip()) for s in args.symbols.split(",") if s.strip()]

    print(f"下载 {len(vt_symbols)} 个标的，周期={args.interval}，"
          f"区间={args.start}~{args.end}，复权={cfg.data.dividend_type}")
    feed.download_history(vt_symbols, args.start, args.end, Interval(args.interval))
    print(f"完成，数据目录: {Path(cfg.data.store_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
