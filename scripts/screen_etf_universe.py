"""按流动性筛出可交易的 ETF 候选池，并按跟踪标的去重。

## 为什么要先筛

沪深两市 1700 多只 ETF，大部分是近两年新发的迷你品种，日均成交额只有
几百万甚至几十万。20 万本金做全仓轮动，单笔就是 20 万砸进去 —— 这些
标的回测里能成交，实盘会被冲击成本吃掉。

把它们留在训练集里，后果和「回测交易科创板/高价股」是同一类：结果虚高，
而且高得没法验证，因为实盘根本复现不了。

## 为什么要按相关性去重

同一个指数常有多家基金公司各发一只 ETF（券商 ETF 就有好几只）。它们
日收益几乎完全同步，留在池子里只会让「轮动」在近似重复品之间空转，
还会让回测误以为分散度比实际高。

去重不按名称匹配 —— 名称五花八门（证券ETF / 券商ETF / 证券保险ETF），
而跟踪同一指数的品种日收益相关性必然在 0.99 以上。用相关性判定客观，
且不依赖券商是否提供跟踪指数字段。

## 输出

    data/universe/etf_candidates.csv

用法::

    python scripts/screen_etf_universe.py
    python scripts/screen_etf_universe.py --min-amount 500000000
    python scripts/screen_etf_universe.py --dedupe-corr 0.98
    python scripts/screen_etf_universe.py --no-dedupe
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DAILY_DIR = ROOT / "data" / "1d"
OUT_PATH = ROOT / "data" / "universe" / "etf_candidates.csv"

#: xtdata 板块 -> 轮动用的分类标签。一只 ETF 可能同时属于多个板块，
#: 按这个顺序取第一个命中的，越具体的排越前
#: 不参与行业轮动的类别 —— 只作空仓时的防御腿。它们是机构现金管理工具，
#: 成交额天然比行业 ETF 高一个数量级，不限额会把候选池淹掉。
DEFENSIVE = ("债券", "货币")

SECTOR_TAGS = [
    ("ETF货币型", "货币"),
    ("ETF债券型", "债券"),
    ("ETF商品型", "商品"),
    ("ETF跨境型", "跨境"),
    ("ETF行业指数", "行业"),
    ("ETF主题指数", "主题"),
    ("TFG宽基ETF", "宽基"),
]


def load_sector_map() -> dict[str, str]:
    """xt_code -> 分类标签。"""
    from xtquant import xtdata

    tag: dict[str, str] = {}
    for sector, label in SECTOR_TAGS:
        try:
            codes = xtdata.get_stock_list_in_sector(sector) or []
        except Exception as e:                       # noqa: BLE001
            print(f"  [WARN] 板块 {sector} 取不到: {e}")
            continue
        for c in codes:
            tag.setdefault(c, label)
        print(f"  {sector:<14} {len(codes):>5} 只")
    return tag


def load_names(xt_codes: list[str]) -> dict[str, str]:
    from xtquant import xtdata

    out = {}
    for c in xt_codes:
        try:
            d = xtdata.get_instrument_detail(c) or {}
            out[c] = d.get("InstrumentName") or ""
        except Exception:                            # noqa: BLE001
            out[c] = ""
    return out


def to_xt(vt: str) -> str:
    code, _, ex = vt.rpartition(".")
    return f"{code}.{'SH' if ex == 'SSE' else 'SZ'}"


def _path_of(vt: str) -> Path:
    code, _, ex = vt.rpartition(".")
    return DAILY_DIR / ex / f"{code}.parquet"


def scan(lookback: int, min_days: int) -> list[dict]:
    """逐个 ETF 日线文件算流动性。"""
    import pandas as pd

    rows = []
    for ex, prefixes in (("SSE", ("50", "51", "52", "56", "58")),
                         ("SZSE", ("15", "16"))):
        d = DAILY_DIR / ex
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.parquet")):
            if not f.stem.startswith(prefixes):
                continue
            try:
                df = pd.read_parquet(f, columns=["close", "volume", "amount"])
            except (OSError, ValueError, KeyError):
                continue
            if len(df) < min_days:
                continue
            tail = df.tail(lookback)
            amt = float(tail["amount"].mean())
            # 停牌日 amount=0，会把均值拉低；单独记录有效交易日占比
            active = float((tail["amount"] > 0).mean())
            rows.append({
                "vt_symbol": f"{f.stem}.{ex}",
                "days": len(df),
                "first_date": str(df.index[0]),
                "last_date": str(df.index[-1]),
                "last_close": round(float(df["close"].iloc[-1]), 4),
                "avg_amount": round(amt, 0),
                "active_ratio": round(active, 3),
            })
    return rows


def dedupe_by_correlation(kept, window: int, threshold: float):
    """同一跟踪标的只留成交额最大的一只。

    贪心：按成交额从大到小取代表，凡与已选代表日收益相关性超过阈值的
    一律并入该代表。返回 (代表行, 被并明细)。
    """
    import pandas as pd

    closes = {}
    for vt in kept["vt_symbol"]:
        try:
            s = pd.read_parquet(_path_of(vt), columns=["close"])["close"]
        except (OSError, ValueError, KeyError):
            continue
        closes[vt] = s.tail(window)

    if len(closes) < 2:
        return kept, []

    px = pd.DataFrame(closes).astype(float).sort_index()
    rets = px.pct_change().dropna(how="all")
    corr = rets.corr()

    order = kept.sort_values("avg_amount", ascending=False)["vt_symbol"].tolist()
    reps: list[str] = []
    merged: list[dict] = []
    taken: set[str] = set()

    for vt in order:
        if vt in taken:
            continue
        if vt not in corr.columns:
            reps.append(vt)
            continue
        reps.append(vt)
        taken.add(vt)
        for other in order:
            if other in taken or other not in corr.columns:
                continue
            c = corr.at[vt, other]
            if pd.notna(c) and c >= threshold:
                taken.add(other)
                merged.append({"rep": vt, "dropped": other, "corr": round(float(c), 4)})

    return kept[kept["vt_symbol"].isin(reps)].copy(), merged


def main() -> int:
    p = argparse.ArgumentParser(description="按流动性筛 ETF 候选池并去重")
    p.add_argument("--min-amount", type=float, default=200_000_000,
                   help="近 N 日均成交额下限，默认 2 亿")
    p.add_argument("--lookback", type=int, default=60, help="流动性回看交易日数")
    p.add_argument("--min-days", type=int, default=250,
                   help="最少历史交易日，滤掉新上市品种")
    p.add_argument("--min-active", type=float, default=0.9,
                   help="回看期内有成交的交易日占比下限")
    p.add_argument("--dedupe-corr", type=float, default=0.98,
                   help="日收益相关性超过此值视为同一跟踪标的")
    p.add_argument("--defensive-cap", type=int, default=2,
                   help="债券/货币各保留几只。它们不参与行业轮动，"
                        "只作空仓时的防御腿，留太多会稀释池子")
    p.add_argument("--corr-window", type=int, default=250,
                   help="算相关性用的交易日数")
    p.add_argument("--no-dedupe", action="store_true", help="不做相关性去重")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    import pandas as pd

    print("=" * 66)
    print("ETF 候选池筛选")
    print("=" * 66)
    print(f"  成交额下限: {args.min_amount/1e8:,.1f} 亿（近 {args.lookback} 日均）")
    print(f"  最少历史  : {args.min_days} 个交易日")
    if not args.no_dedupe:
        print(f"  去重      : 日收益相关性 >= {args.dedupe_corr} "
              f"（{args.corr_window} 日窗口）")
    print()

    print("读取板块分类...")
    tags = load_sector_map()

    print("\n扫描日线...")
    rows = scan(args.lookback, args.min_days)
    if not rows:
        print("  没扫到 ETF 日线，确认 data/1d/ 下已有数据")
        return 1
    print(f"  有足够历史的 ETF: {len(rows)} 只")

    df = pd.DataFrame(rows)
    df["xt_code"] = df["vt_symbol"].map(to_xt)
    df["category"] = df["xt_code"].map(tags).fillna("其他")

    kept = df[(df["avg_amount"] >= args.min_amount)
              & (df["active_ratio"] >= args.min_active)].copy()
    print(f"  过流动性门槛  : {len(kept)} 只")
    if kept.empty:
        print("  门槛过高，没有标的入选。试试调低 --min-amount")
        return 1

    merged: list[dict] = []
    if not args.no_dedupe:
        print("\n按日收益相关性去重...")
        before = len(kept)
        kept, merged = dedupe_by_correlation(
            kept, args.corr_window, args.dedupe_corr)
        print(f"  {before} -> {len(kept)} 只（并掉 {len(merged)} 只重复品）")

    kept = kept.sort_values("avg_amount", ascending=False)

    if args.defensive_cap > 0:
        before = len(kept)
        parts = [g.head(args.defensive_cap) if cat in DEFENSIVE else g
                 for cat, g in kept.groupby("category", sort=False)]
        kept = pd.concat(parts).sort_values("avg_amount", ascending=False)
        if len(kept) != before:
            print(f"\n防御腿限额: 债券/货币各留 {args.defensive_cap} 只"
                  f"（{before} -> {len(kept)}）")

    print("\n取标的名称...")
    names = load_names(kept["xt_code"].tolist())
    kept["name"] = kept["xt_code"].map(names)

    cols = ["vt_symbol", "xt_code", "name", "category", "avg_amount",
            "last_close", "days", "active_ratio", "first_date", "last_date"]
    kept = kept[cols]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, index=False, encoding="utf-8-sig")

    if merged:
        name_of = load_names([to_xt(m["dropped"]) for m in merged])
        rep_names = load_names([to_xt(m["rep"]) for m in merged])
        mp = out.with_name("etf_merged.csv")
        pd.DataFrame([{
            "rep": m["rep"], "rep_name": rep_names.get(to_xt(m["rep"]), ""),
            "dropped": m["dropped"],
            "dropped_name": name_of.get(to_xt(m["dropped"]), ""),
            "corr": m["corr"],
        } for m in merged]).to_csv(mp, index=False, encoding="utf-8-sig")
        print("\n" + "-" * 66)
        print("被并掉的重复品（前 15）")
        print("-" * 66)
        for m in merged[:15]:
            print(f"  {m['dropped']:<14} {str(name_of.get(to_xt(m['dropped']),''))[:12]:<14}"
                  f" -> {m['rep']:<14} "
                  f"{str(rep_names.get(to_xt(m['rep']),''))[:12]:<14} corr={m['corr']}")
        print(f"  完整明细: {mp.relative_to(ROOT)}")

    print("\n" + "-" * 66)
    print("按分类统计")
    print("-" * 66)
    for cat, g in kept.groupby("category", sort=False):
        print(f"  {cat:<6} {len(g):>4} 只   "
              f"日均额中位数 {g['avg_amount'].median()/1e8:>6.2f} 亿")

    print("\n" + "-" * 66)
    print("最终候选池")
    print("-" * 66)
    for _, r in kept.iterrows():
        print(f"  {r['vt_symbol']:<14} {str(r['name'])[:16]:<18} "
              f"{r['category']:<4} {r['avg_amount']/1e8:>7.2f} 亿")

    print(f"\n已写入: {out.relative_to(ROOT)}  （{len(kept)} 只）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
