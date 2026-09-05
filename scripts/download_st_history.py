"""历史 ST 名单 —— 消除 ST 判定的前视偏差。

## 为什么需要

此前排除 ST 用的是 QMT 的**当前**状态：名称里含 "ST" 就排除。
这有两个问题：

1. **前视**：某只股票 2023 年被 ST、2025 年摘帽，现在名称正常。
   按当前状态它不会被排除，但 2023 年的回测里它本该按 ST 规则撮合。
2. **误伤**：某只股票 2026 年刚被 ST，之前一直正常。
   按当前状态它被整段排除，但 2022~2025 年的数据本是干净可用的。

Tushare 的 `namechange` 给出每次更名的**区间**（start_date / end_date），
据此可还原「某只股票在某一天是不是 ST」，这才是无前视的做法。

## ST 为什么必须区分

- **涨跌停 5%** 而非 10%。回测引擎按代码前缀判定涨跌停，
  拿不到 ST 状态时会用 10% 去撮合本该 5% 的标的，
  让本该拒单的委托成交，系统性高估可成交性。
- 流动性差，滑点远高于常规估计
- 退市风险

## 产出

`data/universe/st_history.parquet`，每行一个 ST 区间：

    vt_symbol   start_date   end_date   name        reason

`end_date` 为空表示至今仍是。配套的 `is_st(vt_symbol, date)`
供回测与选股调用。

用法::

    python scripts/download_st_history.py
    python scripts/download_st_history.py --check 600000.SSE 2023-06-01
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "universe" / "st_history.parquet"

#: 名称里出现这些标记即视为风险警示状态。
#: *ST 是退市风险警示（更严重），ST 是其他风险警示，两者涨跌停都是 5%
ST_MARKS = ("ST", "*ST")


def is_st_name(name: str) -> bool:
    """名称是否表示风险警示状态。

    不能简单用 "ST" in name，也不能只看前缀 —— 有正常公司名以这两个
    字母开头（如申通快递曾用名 "STO"）。

    A 股的风险警示标记是**独立的前缀**，后面直接跟中文简称：
    "ST传媒"、"*ST海航"、"SST前锋"、"S*ST光明"。
    因此要求 ST 之后的首个字符不是英文字母。
    """
    n = str(name).strip().upper().replace(" ", "")
    for mark in ("*ST", "SST", "S*ST", "ST"):
        if n.startswith(mark):
            rest = n[len(mark):]
            # 后面紧跟字母说明是普通公司名（STO、STAR 之类），不是风险警示
            return not (rest and rest[0].isascii() and rest[0].isalpha())
    return False


def to_vt(ts_code: str) -> str | None:
    code, _, ex = ts_code.rpartition(".")
    if ex == "SH":
        return f"{code}.SSE"
    if ex == "SZ":
        return f"{code}.SZSE"
    return None


def fetch(client, codes: list, sleep: float = 0.12) -> list:
    """逐只拉更名历史，筛出 ST 区间。"""
    import pandas as pd

    rows = []
    t0 = time.time()
    for i, ts in enumerate(codes, 1):
        try:
            df = client.pro.namechange(ts_code=ts)
        except Exception:
            time.sleep(sleep * 4)
            continue
        if df is None or df.empty:
            continue
        vt = to_vt(ts)
        if not vt:
            continue
        for r in df.itertuples():
            name = str(getattr(r, "name", ""))
            if not is_st_name(name):
                continue
            sd = getattr(r, "start_date", None)
            ed = getattr(r, "end_date", None)
            rows.append({
                "vt_symbol": vt,
                "name": name,
                "start_date": pd.to_datetime(str(sd), format="%Y%m%d",
                                             errors="coerce"),
                # 空表示至今仍是 ST
                "end_date": (pd.to_datetime(str(ed), format="%Y%m%d",
                                            errors="coerce")
                             if ed and str(ed) != "None" else pd.NaT),
                "reason": str(getattr(r, "change_reason", "")),
            })
        time.sleep(sleep)
        if i % 200 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(codes)}  已找到 {len(rows)} 个 ST 区间  "
                  f"剩约 {el / i * (len(codes) - i) / 60:.0f} 分钟")
    return rows


def load_st_history():
    """读 ST 区间表。供回测与选股调用。"""
    import pandas as pd

    if not OUT.exists():
        return None
    return pd.read_parquet(OUT)


def build_checker(df):
    """返回 is_st(vt_symbol, date) -> bool。

    预先按标的分组，避免逐次全表扫描 —— 回测里会调用几十万次。
    """
    import pandas as pd

    by_sym: dict[str, list] = {}
    for r in df.itertuples():
        by_sym.setdefault(r.vt_symbol, []).append(
            (r.start_date, r.end_date))

    def is_st(vt_symbol: str, date) -> bool:
        spans = by_sym.get(vt_symbol)
        if not spans:
            return False
        d = pd.Timestamp(date)
        for s, e in spans:
            if pd.isna(s):
                continue
            if d >= s and (pd.isna(e) or d <= e):
                return True
        return False

    return is_st


def main() -> int:
    p = argparse.ArgumentParser(description="下载历史 ST 名单")
    p.add_argument("--sector", default="沪深A股")
    p.add_argument("--sleep", type=float, default=0.12)
    p.add_argument("--check", nargs=2, metavar=("VT_SYMBOL", "DATE"),
                    help="查询某标的某日是否 ST")
    args = p.parse_args()

    import pandas as pd

    if args.check:
        df = load_st_history()
        if df is None:
            print("尚无 ST 历史，请先运行本脚本下载")
            return 1
        vt, date = args.check
        st = build_checker(df)(vt, date)
        print(f"{vt} 在 {date} {'是' if st else '不是'} ST")
        sub = df[df["vt_symbol"] == vt]
        if not sub.empty:
            print(f"\n该标的的 ST 区间:")
            for r in sub.itertuples():
                e = "至今" if pd.isna(r.end_date) else str(r.end_date.date())
                print(f"  {r.start_date.date()} ~ {e}  {r.name}  {r.reason}")
        return 0

    from xtquant import xtdata

    from qmtquant.datafeed.tushare_feed import TushareClient

    print("=" * 60)
    print("下载历史 ST 名单")
    print("=" * 60)

    codes = xtdata.get_stock_list_in_sector(args.sector) or []
    ts_codes = [c for c in codes if c.endswith((".SH", ".SZ"))]
    print(f"\n板块 {args.sector}：{len(ts_codes)} 只")
    if not ts_codes:
        print("取不到名单 —— 请确认 miniQMT 已启动")
        return 1

    client = TushareClient()
    print(f"预计耗时 {len(ts_codes) * args.sleep / 60:.0f} 分钟\n")

    rows = fetch(client, ts_codes, args.sleep)
    if not rows:
        print("未找到任何 ST 记录")
        return 1

    df = pd.DataFrame(rows).sort_values(["vt_symbol", "start_date"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    n_sym = df["vt_symbol"].nunique()
    ongoing = int(df["end_date"].isna().sum())
    print(f"\n共 {len(df)} 个 ST 区间，涉及 {n_sym} 只标的")
    print(f"  其中 {ongoing} 个至今仍是 ST")
    print(f"  最早 {df['start_date'].min().date()}，"
          f"最晚开始 {df['start_date'].max().date()}")

    # 与「当前状态」对比，量化前视偏差的影响面
    now_st = set()
    for c in ts_codes:
        try:
            d = xtdata.get_instrument_detail(c) or {}
        except Exception:
            continue
        if is_st_name(d.get("InstrumentName", "")):
            vt = to_vt(c)
            if vt:
                now_st.add(vt)
    hist = set(df["vt_symbol"])
    print(f"\n对比当前状态:")
    print(f"  当前是 ST        {len(now_st):>5d} 只")
    print(f"  历史上曾是 ST    {len(hist):>5d} 只")
    print(f"  曾是但现已摘帽    {len(hist - now_st):>5d} 只"
          f"  <- 按当前状态排除会误伤这些")
    print(f"\n已保存: {OUT}")
    print("\n用法：")
    print("  python scripts/download_st_history.py --check 600000.SSE 2023-06-01")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
