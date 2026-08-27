"""下载基准指数（用于回测对标）。

没有基准的净值曲线意义有限 —— 策略赚 10% 到底是本事还是大盘涨了 15%，
必须有对照才知道。

用法：
    python scripts/download_index.py
    python scripts/download_index.py --codes 000300.SH,000905.SH
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.datafeed.xt_feed import BENCHMARKS, IndexFeed  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="下载基准指数")
    parser.add_argument("--codes", default=None,
                        help=f"逗号分隔，默认 {','.join(BENCHMARKS)}")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2030-12-31")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    codes = ([c.strip() for c in args.codes.split(",") if c.strip()]
             if args.codes else list(BENCHMARKS))
    feed = IndexFeed(cfg.data.store_dir)

    print(f"下载 {len(codes)} 个指数：{args.start} ~ {args.end}")
    result = feed.download(codes, args.start, args.end)

    for code in result["ok"]:
        s = feed.load_close(code)
        name = BENCHMARKS.get(code, "")
        print(f"  OK   {code} {name:<8} {len(s)} 条  "
              f"{s.index[0].date()} ~ {s.index[-1].date()}  末值 {s.iloc[-1]:.2f}")
    for code in result["failed"]:
        print(f"  FAIL {code}")
    print(f"\n存储目录: {feed.dir.resolve()}")
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
