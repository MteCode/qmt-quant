"""导出标的元数据（上市日、名称、ST 标记等）到本地。

回测需要按上市日做 point-in-time 过滤，但 get_instrument_detail 要连客户端。
先导出成 parquet，之后回测就不依赖 QMT 了。

用法：
    python scripts/build_universe.py --sector 沪深300
    python scripts/build_universe.py --sector 沪深A股 --out universe_all.parquet
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import from_xt_symbol, to_xt_symbol  # noqa: E402

#: ExpireDate 为该值表示尚未退市
NOT_EXPIRED = "99999999"


def main() -> int:
    parser = argparse.ArgumentParser(description="导出标的元数据")
    parser.add_argument("--sector", default="沪深300")
    parser.add_argument("--out", default=None, help="输出文件名，默认 universe_{板块}.parquet")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    from xtquant import xtdata

    from qmtquant.datafeed.xt_feed import XtDataFeed

    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    vt_symbols = feed.get_sector_stocks(args.sector)
    if not vt_symbols:
        print(f"板块「{args.sector}」未取到成分股")
        return 1

    rows = []
    for i, vt_symbol in enumerate(vt_symbols, 1):
        sys.stdout.write(f"\r  {i}/{len(vt_symbols)} {vt_symbol:<14}")
        sys.stdout.flush()
        try:
            d = xtdata.get_instrument_detail(to_xt_symbol(vt_symbol), iscomplete=True) or {}
        except Exception:
            d = {}

        open_date = str(d.get("OpenDate") or "")
        expire = str(d.get("ExpireDate") or "")
        name = d.get("InstrumentName") or ""
        rows.append({
            "vt_symbol": vt_symbol,
            "name": name,
            # OpenDate 为 'YYYYMMDD'，0 或空表示取不到
            "listing_date": _parse_date(open_date),
            "delist_date": None if expire in ("", "0", NOT_EXPIRED) else _parse_date(expire),
            # 名称里带 ST/退 的当前处于风险警示状态。
            # 注意这是**当前**状态，不是 point-in-time，不能直接用于历史过滤
            "is_st_now": ("ST" in name.upper()) or ("退" in name),
        })
    sys.stdout.write("\n")

    df = pd.DataFrame(rows)
    out_dir = Path(cfg.data.store_dir) / "universe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (args.out or f"universe_{args.sector}.parquet")
    df.to_parquet(out_path)

    missing = int(df["listing_date"].isna().sum())
    print(f"\n已导出 {len(df)} 只标的 -> {out_path}")
    print(f"  上市日缺失   : {missing}")
    print(f"  当前 ST/退市 : {int(df['is_st_now'].sum())}")
    if not df["listing_date"].isna().all():
        print(f"  最早上市日   : {df['listing_date'].min().date()}")
        print(f"  最晚上市日   : {df['listing_date'].max().date()}")
    if missing:
        print(f"\n[!] {missing} 只标的取不到上市日，"
              "PointInTimeUniverse 会保守地把它们排除在外")
    return 0


def _parse_date(s: str):
    """'20010827' -> Timestamp；无效值返回 None"""
    if not s or s in ("0", NOT_EXPIRED) or len(s) != 8:
        return None
    try:
        return pd.Timestamp(s)
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
