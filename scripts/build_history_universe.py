"""构建无幸存者偏差的历史标的池。

思路：不依赖指数成分（拿不到历史成分，且会引入成分股前视偏差），
改用**全市场 point-in-time 标的池** —— 每个调仓日只看「当时确实在市」的股票。

需要两份数据：
1. 在市股票 + 上市日   —— QMT（快）或 akshare
2. 已退市股票 + 上市日 + 退市日 —— **只有 akshare 有**，QMT 中退市股完全不存在

输出 data/universe/universe_full.parquet，供 PointInTimeUniverse 使用。

用法：
    python scripts/build_history_universe.py
    python scripts/build_history_universe.py --no-qmt   # 全部走 akshare
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize  # noqa: E402

NOT_EXPIRED = "99999999"


def fetch_listed_from_qmt() -> pd.DataFrame:
    """从 QMT 取在市 A 股与上市日（快，但不含退市股）"""
    from xtquant import xtdata

    from qmtquant.utils.symbol import from_xt_symbol, to_xt_symbol

    xtdata.enable_hello = False
    codes = xtdata.get_stock_list_in_sector("沪深A股")
    if not codes:
        xtdata.download_sector_data()
        codes = xtdata.get_stock_list_in_sector("沪深A股")

    rows = []
    for i, code in enumerate(codes, 1):
        if i % 500 == 0:
            sys.stdout.write(f"\r  QMT {i}/{len(codes)}")
            sys.stdout.flush()
        try:
            vt_symbol = from_xt_symbol(code)
        except (KeyError, ValueError):
            continue
        d = xtdata.get_instrument_detail(to_xt_symbol(vt_symbol), iscomplete=True) or {}
        open_date = str(d.get("OpenDate") or "")
        rows.append({
            "vt_symbol": vt_symbol,
            "name": d.get("InstrumentName") or "",
            "listing_date": _parse(open_date),
            "delist_date": pd.NaT,
            "status": "listed",
        })
    sys.stdout.write("\n")
    return pd.DataFrame(rows)


def fetch_listed_from_akshare() -> pd.DataFrame:
    """从 akshare 取在市 A 股名单（无上市日，需另行补齐）"""
    import akshare as ak

    df = ak.stock_info_a_code_name()
    rows = [{
        "vt_symbol": normalize(str(c).zfill(6)),
        "name": n,
        "listing_date": pd.NaT,
        "delist_date": pd.NaT,
        "status": "listed",
    } for c, n in zip(df["code"], df["name"])]
    return pd.DataFrame(rows)


def _parse(s: str):
    if not s or s in ("0", NOT_EXPIRED) or len(s) != 8:
        return pd.NaT
    try:
        return pd.Timestamp(s)
    except ValueError:
        return pd.NaT


def main() -> int:
    parser = argparse.ArgumentParser(description="构建无幸存者偏差的标的池")
    parser.add_argument("--no-qmt", action="store_true", help="不用 QMT，全部走 akshare")
    parser.add_argument("--out", default="universe_full.parquet")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    # ---- 在市股票
    print(">>> 获取在市 A 股")
    if args.no_qmt:
        listed = fetch_listed_from_akshare()
    else:
        try:
            listed = fetch_listed_from_qmt()
        except ImportError:
            print("  xtquant 不可用，改用 akshare")
            listed = fetch_listed_from_akshare()
    print(f"    在市 {len(listed)} 只，其中有上市日的 "
          f"{int(listed['listing_date'].notna().sum())} 只")

    # ---- 退市股票（关键：QMT 完全没有这部分）
    print("\n>>> 获取已退市股票（akshare）")
    from qmtquant.datafeed.ak_feed import fetch_delisted_stocks

    delisted = fetch_delisted_stocks()
    if delisted.empty:
        print("    [!] 未取到退市名单，幸存者偏差无法消除")
    else:
        delisted["status"] = "delisted"
        print(f"    退市 {len(delisted)} 只，"
              f"退市日范围 {delisted['delist_date'].min()} ~ {delisted['delist_date'].max()}")

    # ---- 合并
    full = pd.concat([listed, delisted], ignore_index=True)
    full = full.drop_duplicates(subset=["vt_symbol"], keep="first")
    full = full.sort_values("vt_symbol").reset_index(drop=True)

    out_dir = Path(cfg.data.store_dir) / "universe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out
    full.to_parquet(out_path)

    print("\n" + "=" * 56)
    print(f"标的池已输出 -> {out_path}")
    print(f"  总计        : {len(full)}")
    print(f"  在市        : {int((full['status']=='listed').sum())}")
    print(f"  已退市      : {int((full['status']=='delisted').sum())}")
    print(f"  有上市日    : {int(full['listing_date'].notna().sum())}")
    print(f"  缺上市日    : {int(full['listing_date'].isna().sum())}")
    print("=" * 56)
    print("\n下一步：下载退市股行情（QMT 没有，只能走 akshare）")
    print("  python scripts/download_delisted.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
