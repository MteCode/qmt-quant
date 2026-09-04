"""下载不复权价并推导复权因子。

## 为什么需要

本地存的是**后复权价**，它比真实价高出很多（茅台库中 8125 元，真实约 1400，
抬高 5.4 倍）。这在两处造成实质错误：

1. **回测的整手约束失真** —— 一手的名义成本被同比例放大。
   实测 100 万本金 / 10 只持仓时，沪深300 有 36 只取整后为 0 股，
   等于把这些标的悄悄剔出了标的池。价格越高的股票越容易被误剔，
   而它们往往正是大盘蓝筹。

2. **实盘下单量算错** —— 按后复权价算出的股数与真实可买数不同。

后复权价适合算收益率（连续可比），但**不适合算「能买几股」**。
两者都需要，所以把不复权价单独存一份，并推导出复权因子：

    复权因子 = 后复权价 / 不复权价
    真实价   = 后复权价 / 复权因子

## 存储

不写进 `data/1d/`（那里是后复权，混在一起会分不清），
单独存 `data/1d_raw/`，结构与之相同。复权因子由两者相除得到，
不单独存 —— 存了就要维护一致性，除法很便宜。

用法::

    python scripts/download_adj_factor.py --sector 中证1000
    python scripts/download_adj_factor.py --symbols 600519.SSE,000001.SZSE
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_DIRNAME = "1d_raw"


def to_xt_code(vt_symbol: str) -> str:
    code, _, ex = vt_symbol.rpartition(".")
    return f"{code}.{'SH' if ex == 'SSE' else 'SZ'}"


def download(symbols: list, store: Path, start: str, end: str,
             resume: bool = True) -> dict:
    from xtquant import xtdata

    out_root = store / RAW_DIRNAME
    ok = skipped = failed = 0
    t0 = time.time()

    for i, vt in enumerate(symbols, 1):
        code, _, ex = vt.rpartition(".")
        dst = out_root / ex / f"{code}.parquet"
        if resume and dst.exists():
            skipped += 1
            continue

        xt_code = to_xt_code(vt)
        try:
            xtdata.download_history_data(xt_code, period="1d",
                                         start_time=start, end_time=end)
            df = xtdata.get_market_data_ex(
                field_list=[], stock_list=[xt_code], period="1d",
                start_time=start, end_time=end,
                # 关键：不复权。data/1d/ 存的是后复权，两者相除即复权因子
                dividend_type="none", fill_data=False).get(xt_code)
        except Exception:
            failed += 1
            continue

        if df is None or df.empty:
            failed += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst)
        ok += 1

        if i % 200 == 0:
            print(f"  {i}/{len(symbols)}  成功 {ok} 跳过 {skipped} "
                  f"失败 {failed}  ({time.time() - t0:.0f}s)")

    return {"ok": ok, "skipped": skipped, "failed": failed,
            "elapsed": time.time() - t0}


def verify(store: Path, samples: int = 5) -> None:
    """抽样核对：复权因子是否合理、真实价是否落在常识区间。"""
    import pandas as pd

    raw_root = store / RAW_DIRNAME
    adj_root = store / "1d"
    files = sorted(raw_root.rglob("*.parquet"))[:samples * 20]
    if not files:
        print("  无不复权数据可核对")
        return

    print(f"\n{'标的':<14s}{'后复权':>10s}{'不复权':>10s}"
          f"{'复权因子':>10s}{'最后日期':>12s}")
    print("-" * 58)
    shown = 0
    for p in files:
        vt = f"{p.stem}.{p.parent.name}"
        adj_p = adj_root / p.parent.name / p.name
        if not adj_p.exists():
            continue
        try:
            raw = pd.read_parquet(p)
            adj = pd.read_parquet(adj_p)
        except (OSError, ValueError):
            continue
        common = raw.index.intersection(adj.index)
        if len(common) == 0:
            continue
        d = common[-1]
        r, a = float(raw.loc[d, "close"]), float(adj.loc[d, "close"])
        if r <= 0:
            continue
        print(f"{vt:<14s}{a:>10.2f}{r:>10.2f}{a / r:>10.4f}{str(d):>12s}")
        shown += 1
        if shown >= samples:
            break


def main() -> int:
    p = argparse.ArgumentParser(description="下载不复权价（用于推导复权因子）")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--symbols", help="逗号分隔的 vt_symbol")
    src.add_argument("--sector", default="中证1000", help="板块名")
    p.add_argument("--start", default="20160101")
    p.add_argument("--end", default="20301231")
    p.add_argument("--rebuild", action="store_true", help="重新下载已有的")
    p.add_argument("--verify-only", action="store_true", help="只做抽样核对")
    args = p.parse_args()

    from qmtquant.config import get_config
    store = Path(get_config().data.store_dir)

    if args.verify_only:
        verify(store)
        return 0

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        from xtquant import xtdata
        codes = xtdata.get_stock_list_in_sector(args.sector) or []
        symbols = [f"{c.split('.')[0]}."
                   f"{'SSE' if c.endswith('.SH') else 'SZSE'}" for c in codes]

    print("=" * 58)
    print("下载不复权价")
    print(f"  标的 {len(symbols)} 只   {args.start} ~ {args.end}")
    print(f"  输出 {store / RAW_DIRNAME}")
    print("=" * 58)
    if not symbols:
        print("没有标的 —— 请确认 miniQMT 已启动")
        return 1

    r = download(symbols, store, args.start, args.end,
                 resume=not args.rebuild)
    print(f"\n成功 {r['ok']}  跳过 {r['skipped']}  失败 {r['failed']}"
          f"  耗时 {r['elapsed'] / 60:.1f} 分钟")

    verify(store)
    print(f"\n复权因子 = 后复权价 / 不复权价")
    print(f"真实价   = 后复权价 / 复权因子")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
