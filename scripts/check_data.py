"""数据质量检查。

对本地已下载的行情与财务数据跑一遍校验规则，产出问题清单。

**只报告，不修改数据。** 自动"修复"比脏数据更危险 —— 你不知道它改了什么。

用法：
    python scripts/check_data.py --sector 沪深300
    python scripts/check_data.py --sector 沪深300 --interval 1d,1w
    python scripts/check_data.py --symbols 600519.SH --verbose
    python scripts/check_data.py --sector 沪深300 --financial
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import Interval  # noqa: E402
from qmtquant.datafeed.financial import DEFAULT_TABLES, FinancialStore  # noqa: E402
from qmtquant.datafeed.validator import (  # noqa: E402
    BarValidator,
    FinancialValidator,
    Severity,
    summarize,
)
from qmtquant.datafeed.xt_feed import XtDataFeed  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize  # noqa: E402


def load_symbols(args, cfg, feed: XtDataFeed) -> list[str]:
    if args.symbols:
        return [normalize(s.strip()) for s in args.symbols.split(",") if s.strip()]

    # 优先用本地元数据，避免依赖 QMT 客户端
    meta = Path(cfg.data.store_dir) / "universe" / f"universe_{args.sector}.parquet"
    if meta.exists():
        return pd.read_parquet(meta)["vt_symbol"].tolist()
    return feed.get_sector_stocks(args.sector)


def check_bars(feed: XtDataFeed, symbols: list[str], intervals: list[Interval],
               validator: BarValidator, verbose: bool) -> list:
    all_issues = []
    for interval in intervals:
        print(f"\n>>> 检查 {interval.value} 行情（{len(symbols)} 只）")
        checked = 0
        for i, vt_symbol in enumerate(symbols, 1):
            if i % 50 == 0:
                sys.stdout.write(f"\r  {i}/{len(symbols)}")
                sys.stdout.flush()

            path = feed._path(vt_symbol, interval)
            if not path.exists():
                continue
            checked += 1
            df = feed._normalize_df(pd.read_parquet(path))
            issues = validator.validate(df, vt_symbol)
            all_issues += issues
            if verbose:
                for issue in issues:
                    print(f"\r  {issue}")
        sys.stdout.write(f"\r  已检查 {checked} 只，无数据 {len(symbols)-checked} 只\n")
    return all_issues


def check_financial(store: FinancialStore, symbols: list[str],
                    validator: FinancialValidator, verbose: bool) -> list:
    all_issues = []
    print(f"\n>>> 检查财务数据（{len(symbols)} 只 × {len(DEFAULT_TABLES)} 表）")
    for i, vt_symbol in enumerate(symbols, 1):
        if i % 50 == 0:
            sys.stdout.write(f"\r  {i}/{len(symbols)}")
            sys.stdout.flush()
        for table in DEFAULT_TABLES:
            df = store.load(vt_symbol, table)
            if df.empty:
                continue
            issues = validator.validate(df, vt_symbol, table)
            all_issues += issues
            if verbose:
                for issue in issues:
                    print(f"\r  {issue}")
    sys.stdout.write("\r" + " " * 40 + "\r")
    return all_issues


def main() -> int:
    parser = argparse.ArgumentParser(description="数据质量检查")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--sector", default="沪深300")
    src.add_argument("--symbols", default=None)
    parser.add_argument("--interval", default="1d",
                        help="逗号分隔的周期，如 1d,1w,1m")
    parser.add_argument("--financial", action="store_true", help="同时检查财务数据")
    parser.add_argument("--verbose", "-v", action="store_true", help="逐条打印问题")
    parser.add_argument("--tolerance", type=float, default=1.5,
                        help="涨跌幅容忍倍数，超过 涨跌停×该值 判为异常")
    parser.add_argument("--out", default=None, help="问题清单导出 CSV 路径")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    symbols = load_symbols(args, cfg, feed)
    if not symbols:
        print("没有可检查的标的")
        return 1

    intervals = [Interval(s.strip()) for s in args.interval.split(",") if s.strip()]
    bar_validator = BarValidator(price_limit_tolerance=args.tolerance)

    print("=" * 62)
    print(f"数据质量检查  标的 {len(symbols)} 只  周期 "
          f"{', '.join(i.value for i in intervals)}")
    print("=" * 62)

    issues = check_bars(feed, symbols, intervals, bar_validator, args.verbose)

    if args.financial:
        store = FinancialStore(cfg.data.store_dir)
        issues += check_financial(store, symbols, FinancialValidator(), args.verbose)

    # ---- 汇总
    print("\n" + "=" * 62)
    if not issues:
        print("未发现问题")
        return 0

    df = summarize(issues)
    print(df.to_string(index=False))
    print("=" * 62)

    errors = [i for i in issues if i.severity is Severity.ERROR]
    warnings = [i for i in issues if i.severity is Severity.WARNING]
    print(f"\n错误 {len(errors)} 条 | 可疑 {len(warnings)} 条")

    if errors:
        print("\n--- 错误明细（前 10 条）---")
        for issue in errors[:10]:
            print(f"  {issue}")
        print("\n[!] 存在数据错误，用这些数据得出的回测结论不可信")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "规则": i.rule, "级别": i.severity.value, "标的": i.vt_symbol,
            "数量": i.count, "说明": i.detail, "样例": "; ".join(map(str, i.samples)),
        } for i in issues]).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n完整清单已导出: {out.resolve()}")

    # 有错误时返回非 0，方便接入自动化流程
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
