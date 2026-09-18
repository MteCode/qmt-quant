"""用分钟线筛做 T 标的 —— 算真正能落地的两个指标。

## 日线筛不出什么

日线只能给「振幅」和「收盘离开盘多远」。但同样是振幅 8%、收盘回到开盘，
可能是两种完全不同的走势：

    A: 早盘一路跌 4%，午后一路涨回来      -> 一天只有 1 次机会
    B: 全天在 ±4% 区间里来回震荡五六次    -> 一天有 5 次机会

做 T 要的是 B。日线上这两者长得一模一样，**必须用分钟线才能区分**。

## 两个核心指标

1. **日均可做 T 次数**
   用 zigzag 分解日内走势：只保留幅度 ≥ 阈值的波段（默认 0.8%，够覆盖
   双边成本还有余）。波段个数就是当天理论上的操作机会数。

2. **日均可捕捉幅度**
   所有合格波段的幅度之和。这才是做 T 的收益上限 —— 振幅只说明最高最低
   差多少，而可捕捉幅度说明「来回走了多少」。

配套的 **路径比 = 日内路径长度 / 振幅**：等于 1 说明单边走完，越大说明
来回越多。它和波段数互为印证。

## 为什么阈值是 0.8%

一个来回要付两次手续费加滑点，约 0.15%。波段太小赚不回成本。0.8% 是
留出 5 倍安全边际的经验值，可用 --swing 调。

用法::

    python scripts/screen_t0_minute.py
    python scripts/screen_t0_minute.py --swing 0.01 --top-n 200
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MIN_DIR = ROOT / "data" / "clean" / "1m"
DAILY_CANDIDATES = ROOT / "data" / "universe" / "t0_candidates.csv"
OUT_PATH = ROOT / "data" / "universe" / "t0_minute_ranked.csv"


def zigzag_swings(px, thr: float):
    """把一段价格序列分解成幅度 >= thr 的波段，返回 (波段数, 幅度合计)。

    标准 zigzag：跟踪当前方向上的极值，一旦反向回撤超过阈值就确认一个
    波段。只统计确认了的波段 —— 未确认的尾部不算，否则会把「还没走完的
    半个波段」当成机会。
    """
    n = len(px)
    if n < 3:
        return 0, 0.0

    cnt = 0
    total = 0.0
    pivot = px[0]        # 上一个已确认的转折点
    hi = lo = px[0]      # 自 pivot 以来的高低点
    direction = 0        # +1 上行, -1 下行, 0 未定

    for i in range(1, n):
        p = px[i]
        if p > hi:
            hi = p
        if p < lo:
            lo = p

        # 上行（或方向未定）：自高点回撤超过阈值 -> 确认一个上行波段
        if direction >= 0 and hi > pivot > 0 and hi > 0 \
                and (p - hi) / hi <= -thr:
            cnt += 1
            total += (hi - pivot) / pivot
            pivot, direction = hi, -1
            hi = lo = p
            continue

        # 下行（或方向未定）：自低点反弹超过阈值 -> 确认一个下行波段
        if direction <= 0 and 0 < lo < pivot and lo > 0 \
                and (p - lo) / lo >= thr:
            cnt += 1
            total += (pivot - lo) / pivot
            pivot, direction = lo, 1
            hi = lo = p

    return cnt, total


def analyse(vt: str, thr: float, max_days: int):
    import numpy as np
    import pandas as pd

    code, _, ex = vt.rpartition(".")
    f = MIN_DIR / ex / f"{code}.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f, columns=["close", "amount"])
    except (OSError, ValueError, KeyError):
        return None
    # 至少 20 个交易日。清洗层各标的的覆盖长度差异很大（实测有的只有
    # 80 多天），门槛设太高会静默筛掉大半
    if len(df) < 240 * 20:
        return None


    # 清洗层的 1m 索引是 DatetimeIndex，原始层是 YYYYMMDDHHMMSS 字符串，
    # 两种都要认 —— 按字符串切前 8 位在 DatetimeIndex 上会切出 "2026-05-"，
    # 日期分组全错且不报错，表现为「一只都筛不出来」
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        day = idx.strftime("%Y%m%d").to_numpy()
    else:
        day = idx.astype(str).str.replace("-", "", regex=False).str[:8].to_numpy()
    close = df["close"].to_numpy(dtype=float)
    amt = df["amount"].to_numpy(dtype=float)

    uniq = np.unique(day)[-max_days:]
    swings, spans, ratios, amps = [], [], [], []
    for d in uniq:
        m = day == d
        px = close[m]
        if len(px) < 60 or amt[m].sum() <= 0:
            continue
        hi, lo = px.max(), px.min()
        base = px[0]
        if base <= 0 or hi <= lo:
            continue
        amp = (hi - lo) / base
        # 日内路径长度：逐分钟绝对涨跌之和
        path = float(np.abs(np.diff(px)).sum() / base)
        c, tot = zigzag_swings(px, thr)
        swings.append(c)
        spans.append(tot)
        ratios.append(path / amp if amp > 0 else 0.0)
        amps.append(amp)

    if len(swings) < 20:
        return None
    return {
        "vt_symbol": vt,
        "days": len(swings),
        "swings_per_day": float(np.mean(swings)),
        "capturable": float(np.mean(spans)),
        "path_ratio": float(np.mean(ratios)),
        "amplitude": float(np.mean(amps)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="分钟线筛做 T 标的")
    p.add_argument("--swing", type=float, default=0.008,
                   help="波段幅度阈值，默认 0.8%%（双边成本约 0.15%%）")
    p.add_argument("--top-n", type=int, default=300,
                   help="从日线候选里取前 N 只做分钟级分析")
    p.add_argument("--max-days", type=int, default=120, help="回看交易日数")
    p.add_argument("--min-swings", type=float, default=2.0,
                   help="日均可做 T 次数下限")
    p.add_argument("--show", type=int, default=25)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    import pandas as pd

    if not DAILY_CANDIDATES.exists():
        print(f"缺少日线候选 {DAILY_CANDIDATES}，先跑 screen_t0_candidates.py")
        return 1
    cand = pd.read_csv(DAILY_CANDIDATES, encoding="utf-8-sig")
    syms = cand["vt_symbol"].astype(str).tolist()[: args.top_n]
    name_of = dict(zip(cand["vt_symbol"].astype(str),
                       cand.get("name", pd.Series(dtype=str)).astype(str)))

    print("=" * 78)
    print("分钟线做 T 筛选")
    print("=" * 78)
    print(f"  输入 {len(syms)} 只（日线候选前 {args.top_n}）")
    print(f"  波段阈值 {args.swing:.1%}   回看 {args.max_days} 日")
    print()

    rows = []
    for i, vt in enumerate(syms, 1):
        r = analyse(vt, args.swing, args.max_days)
        if r:
            rows.append(r)
        if i % 50 == 0:
            print(f"    {i}/{len(syms)} 已分析，有效 {len(rows)}")

    if not rows:
        print("  没有标的有足够分钟数据")
        return 1

    df = pd.DataFrame(rows)
    df["name"] = df["vt_symbol"].map(name_of).fillna("")
    # 收益上限 ≈ 可捕捉幅度；次数保证机会不是偶发
    df = df[df["swings_per_day"] >= args.min_swings]
    df = df.sort_values("capturable", ascending=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["vt_symbol", "name", "capturable", "swings_per_day",
            "path_ratio", "amplitude", "days"]
    df[cols].to_csv(out, index=False, encoding="utf-8-sig")

    print()
    print(f"  有效分析 {len(rows)} 只，过次数门槛 {len(df)} 只")
    print()
    print("-" * 78)
    print(f"  前 {min(args.show, len(df))} 只（按日均可捕捉幅度排序）")
    print("-" * 78)
    print(f"  {'标的':<14}{'名称':<10}{'可捕捉':>8}{'次数/日':>8}"
          f"{'路径比':>8}{'振幅':>8}")
    for _, r in df.head(args.show).iterrows():
        print(f"  {r['vt_symbol']:<14}{str(r['name'])[:8]:<10}"
              f"{r['capturable']:>7.2%}{r['swings_per_day']:>8.1f}"
              f"{r['path_ratio']:>8.2f}{r['amplitude']:>8.2%}")

    print()
    print(f"  已写入: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
