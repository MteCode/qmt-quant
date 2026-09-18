"""筛选适合日内做 T 的标的。

## 做 T 要的是什么

做 T = 持有底仓，日内卖昨仓、低位买回，净持仓不变，赚日内波动差价。
所以选股标准和动量/趋势策略**正好相反**：

- 趋势策略要「单边走」—— 买了一路涨
- 做 T 要「来回震荡」—— 同一天里既有高点也有低点，且反复出现

一只票当天从开盘一路涨到收盘，振幅再大也做不了 T：你卖在半路，它继续
涨，回不来。真正能做 T 的是那种早盘冲高、盘中回落、尾盘又拉起来的。

## 三个核心指标

1. **日均振幅** = (high - low) / preClose
   没有振幅就没有空间。低于 2% 基本无利可图 —— 扣掉双边手续费和滑点
   就剩不下什么。

2. **非趋势度** = 1 - |close - open| / (high - low)
   衡量「日内走完后回到了哪」。收盘价接近开盘价、但当天高低点拉得开，
   说明是来回震荡（=1 最理想）；收盘就在最高/最低点，说明是单边（=0）。
   **这一项是做 T 与趋势选股的分水岭。**

3. **做 T 空间** = 日均振幅 × 非趋势度
   两者相乘才有意义：振幅 3% 但单边走完，实际可回转空间接近 0。

## 排除什么

- 涨跌停日占比高的：封板时买不进也卖不出，做 T 无从谈起
- 流动性不足的：20 万本金单笔进出，成交额太小会被冲击成本吃掉
- 硬性规则：北交所、科创板、ST、股价 >500 元

用法::

    python scripts/screen_t0_candidates.py
    python scripts/screen_t0_candidates.py --days 120 --min-amount 200000000
    python scripts/screen_t0_candidates.py --min-amplitude 0.025
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DAILY_DIR = ROOT / "data" / "clean" / "1d"
OUT_PATH = ROOT / "data" / "universe" / "t0_candidates.csv"


def scan(days: int, min_rows: int) -> list[dict]:
    """逐个标的算做 T 相关指标。"""
    import numpy as np
    import pandas as pd

    rows = []
    for ex in ("SSE", "SZSE"):          # BSE 硬性排除
        d = DAILY_DIR / ex
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.parquet")):
            code = f.stem
            # 硬性规则：科创板 688、北交所 43/83/87/92
            if code.startswith("688") or code[:2] in ("43", "83", "87", "92"):
                continue
            # ETF/基金不在此列（本脚本只看股票）
            if code[:2] in ("51", "56", "58", "50", "15", "16"):
                continue
            try:
                df = pd.read_parquet(
                    f, columns=["open", "high", "low", "close",
                                "amount", "preClose", "suspendFlag"])
            except (OSError, ValueError, KeyError):
                continue
            df = df.tail(days)
            if len(df) < min_rows:
                continue

            # 停牌日剔除：振幅恒为 0，会把均值拉低
            act = df[(df["suspendFlag"] == 0) & (df["amount"] > 0)]
            if len(act) < min_rows:
                continue

            pc = act["preClose"].to_numpy(dtype=float)
            hi = act["high"].to_numpy(dtype=float)
            lo = act["low"].to_numpy(dtype=float)
            op = act["open"].to_numpy(dtype=float)
            cl = act["close"].to_numpy(dtype=float)
            ok = pc > 0
            if ok.sum() < min_rows:
                continue
            pc, hi, lo, op, cl = pc[ok], hi[ok], lo[ok], op[ok], cl[ok]

            amp = (hi - lo) / pc                       # 日振幅
            rng = hi - lo
            with np.errstate(divide="ignore", invalid="ignore"):
                # 非趋势度：收盘离开盘越近、而高低点拉得越开，越适合做 T
                osc = np.where(rng > 0, 1.0 - np.abs(cl - op) / rng, 0.0)
            osc = np.clip(osc, 0.0, 1.0)

            # 涨跌停：当日涨跌幅绝对值 ≥9.8% 视为触及（近似，够用）
            chg = np.abs(cl / pc - 1.0)
            limit_ratio = float((chg >= 0.098).mean())

            rows.append({
                "vt_symbol": f"{code}.{ex}",
                "days": int(len(amp)),
                "amplitude": float(np.mean(amp)),
                "amp_median": float(np.median(amp)),
                # 振幅的稳定性：偶尔暴动不如天天有波动
                "amp_std": float(np.std(amp)),
                "oscillation": float(np.mean(osc)),
                "t0_space": float(np.mean(amp) * np.mean(osc)),
                "avg_amount": float(np.mean(act["amount"].to_numpy(float))),
                "limit_ratio": limit_ratio,
                "last_close": float(cl[-1]),
            })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="筛选适合日内做 T 的标的")
    p.add_argument("--days", type=int, default=120, help="回看交易日数")
    p.add_argument("--min-rows", type=int, default=60,
                   help="有效交易日下限，滤掉次新与长期停牌")
    p.add_argument("--min-amplitude", type=float, default=0.02,
                   help="日均振幅下限，默认 2%%（低于此扣完成本无利可图）")
    p.add_argument("--min-oscillation", type=float, default=0.45,
                   help="非趋势度下限，默认 0.45")
    p.add_argument("--min-amount", type=float, default=100_000_000,
                   help="日均成交额下限，默认 1 亿")
    p.add_argument("--max-limit-ratio", type=float, default=0.10,
                   help="涨跌停日占比上限，默认 10%%")
    p.add_argument("--max-price", type=float, default=500.0)
    p.add_argument("--min-price", type=float, default=3.0,
                   help="股价下限。低价股一个 tick 就是很大的相对成本")
    p.add_argument("--top", type=int, default=40, help="展示前 N 只")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    import pandas as pd

    print("=" * 74)
    print("日内做 T 标的筛选")
    print("=" * 74)
    print(f"  回看 {args.days} 个交易日")
    print(f"  日均振幅 ≥ {args.min_amplitude:.1%}   非趋势度 ≥ {args.min_oscillation}")
    print(f"  日均成交额 ≥ {args.min_amount/1e8:.1f} 亿   "
          f"涨跌停日占比 ≤ {args.max_limit_ratio:.0%}")
    print()

    rows = scan(args.days, args.min_rows)
    if not rows:
        print("  没扫到数据，确认 data/clean/1d/ 下有日线")
        return 1
    df = pd.DataFrame(rows)
    print(f"  扫描 {len(df):,} 只（已排除科创板/北交所/ETF）")

    steps = [
        ("股价区间", (df["last_close"] >= args.min_price)
         & (df["last_close"] <= args.max_price)),
        ("流动性", df["avg_amount"] >= args.min_amount),
        ("振幅", df["amplitude"] >= args.min_amplitude),
        ("非趋势度", df["oscillation"] >= args.min_oscillation),
        ("涨跌停占比", df["limit_ratio"] <= args.max_limit_ratio),
    ]
    mask = pd.Series(True, index=df.index)
    for label, cond in steps:
        before = int(mask.sum())
        mask &= cond
        print(f"    {label:<10} {before:>5} -> {int(mask.sum()):>5}")

    kept = df[mask].sort_values("t0_space", ascending=False)
    if kept.empty:
        print("\n  无标的入选，试试放宽 --min-amplitude 或 --min-amount")
        return 1

    # ST 过滤放在最后：名称查询比较慢，只对入选的查
    try:
        from qmtquant.utils.symbol import to_xt_symbol
        from xtquant import xtdata
        names, drop_st = {}, []
        for vt in kept["vt_symbol"]:
            d = xtdata.get_instrument_detail(to_xt_symbol(vt)) or {}
            nm = d.get("InstrumentName") or ""
            names[vt] = nm
            if "ST" in nm.upper() or "退" in nm:
                drop_st.append(vt)
        kept = kept[~kept["vt_symbol"].isin(drop_st)].copy()
        kept["name"] = kept["vt_symbol"].map(names)
        if drop_st:
            print(f"    {'ST/退市':<10} 排除 {len(drop_st)} 只")
    except Exception as e:                           # noqa: BLE001
        print(f"    [WARN] ST 过滤跳过: {e}")
        kept["name"] = ""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["vt_symbol", "name", "t0_space", "amplitude", "amp_median",
            "oscillation", "avg_amount", "limit_ratio", "last_close", "days"]
    kept[cols].to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n  最终入选 {len(kept)} 只")
    print()
    print("-" * 74)
    print(f"  前 {min(args.top, len(kept))} 只（按做T空间 = 振幅 × 非趋势度 排序）")
    print("-" * 74)
    print(f"  {'标的':<14}{'名称':<10}{'做T空间':>8}{'振幅':>8}"
          f"{'非趋势':>8}{'日均额':>10}{'股价':>8}")
    for _, r in kept.head(args.top).iterrows():
        print(f"  {r['vt_symbol']:<14}{str(r['name'])[:8]:<10}"
              f"{r['t0_space']:>7.2%}{r['amplitude']:>8.2%}"
              f"{r['oscillation']:>8.2f}{r['avg_amount']/1e8:>9.2f}亿"
              f"{r['last_close']:>8.2f}")

    print()
    print(f"  已写入: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
