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


def verify(store: Path, samples: int = 5) -> dict:
    """核对：复权因子分布是否合理、末根日期是否对齐。

    末根日期对齐是关键 —— 引擎用「后复权末根收盘 / 不复权末根收盘」求因子
    （backtest_engine._adj_factor 路径 1 取 .iloc[-1]）。两根若跨了除权日，
    相除得到的因子就是错的。抽 5 只看不出这个问题，必须全量扫。

    返回统计 dict，便于自动化断言。
    """
    import numpy as np
    import pandas as pd

    raw_root = store / RAW_DIRNAME
    adj_root = store / "1d"
    files = sorted(raw_root.rglob("*.parquet"))
    if not files:
        print("  无不复权数据可核对")
        return {"total": 0}

    factors: list[float] = []
    raw_newer = 0                                  # 不复权末根晚于后复权（正常）
    raw_stale: list[tuple[str, str, str]] = []     # 不复权落后（危险）
    missing_adj = 0
    shown = 0

    print(f"\n{'标的':<14s}{'后复权':>10s}{'不复权':>10s}"
          f"{'复权因子':>10s}{'最后日期':>12s}")
    print("-" * 58)
    for p in files:
        vt = f"{p.stem}.{p.parent.name}"
        adj_p = adj_root / p.parent.name / p.name
        if not adj_p.exists():
            missing_adj += 1
            continue
        try:
            raw = pd.read_parquet(p)
            adj = pd.read_parquet(adj_p)
        except (OSError, ValueError):
            continue
        if raw.empty or adj.empty:
            continue
        common = raw.index.intersection(adj.index)
        if len(common) == 0:
            continue
        d = common[-1]
        r, a = float(raw.loc[d, "close"]), float(adj.loc[d, "close"])
        if r <= 0:
            continue
        f = a / r
        factors.append(f)

        # 不复权通常比后复权**新**（今日仍在下，后复权滞后一两天）——
        # 引擎按日期对齐取值，这是正常状态，不是问题。
        # 真正危险的是反过来：不复权落后，引擎只能取更早的价，可能跨除权日。
        raw_last, adj_last = str(raw.index[-1]), str(adj.index[-1])
        if raw_last > adj_last:
            raw_newer += 1
        elif raw_last < adj_last:
            raw_stale.append((vt, raw_last, adj_last))

        if shown < samples:
            print(f"{vt:<14s}{a:>10.2f}{r:>10.2f}{f:>10.4f}{str(d):>12s}")
            shown += 1

    if not factors:
        print("  无可核对的因子（数据为空或缺失）")
        return {"total": 0}

    arr = np.array(factors)
    bad = int(((arr < 1.0) | (arr > 500.0)).sum())
    print(f"\n  复权因子：{len(arr)} 只  "
          f"min={arr.min():.2f}  p50={np.median(arr):.2f}  max={arr.max():.2f}")
    print(f"  落在 [1, 500] 之外：{bad} 只（{bad / len(arr) * 100:.1f}%）")
    if raw_newer:
        print(f"  不复权末根晚于后复权：{raw_newer} 只"
              f"（正常，引擎按日期对齐取值）")
    if raw_stale:
        print(f"  ! 不复权数据落后于后复权：{len(raw_stale)} 只 —— "
              f"引擎会取更早的不复权价，可能跨除权日。示例：")
        for vt, rd, ad in raw_stale[:3]:
            print(f"      {vt}  不复权末根 {rd}  /  后复权末根 {ad}")
    if missing_adj:
        print(f"  ! {missing_adj} 只无对应的后复权文件，无法核对")

    return {"total": len(arr), "bad": bad,
            "raw_newer": raw_newer, "raw_stale": len(raw_stale),
            "missing_adj": missing_adj,
            "min": float(arr.min()), "max": float(arr.max())}


def main() -> int:
    p = argparse.ArgumentParser(description="下载不复权价（用于推导复权因子）")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--symbols", help="逗号分隔的 vt_symbol")
    src.add_argument("--sector", help="板块名（如 中证1000）")
    src.add_argument("--from-store", action="store_true",
                     help="下载 data/1d/ 里已有行情的全部标的（回测实际用到的）")
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
    elif args.from_store:
        # 回测用的历史成分股远超当前板块 —— 实测中证1000 历史成分 2839 只，
        # 而"中证1000"板块只含当前 1000 只，按板块下载会漏掉 1839 只。
        # 直接按 data/1d/ 实有行情下，覆盖任意回测标的。
        adj_root = store / "1d"
        symbols = sorted(f"{q.stem}.{q.parent.name}"
                         for q in adj_root.rglob("*.parquet"))
        print(f"  从 data/1d/ 取到 {len(symbols)} 只标的")
    else:
        from xtquant import xtdata
        codes = xtdata.get_stock_list_in_sector(args.sector or "中证1000") or []
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
