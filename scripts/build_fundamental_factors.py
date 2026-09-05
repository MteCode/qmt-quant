"""基本面因子 —— 从财报构建截面因子。

## 为什么这是当前最大的增量

已有因子全是价量与市场行为：Alpha158（价量）、龙虎榜、两融、公告。
这几天的实验反复证明这条路走到头了 —— 换模型族、加正交因子、集成，
都撞在同一堵墙上。基本面是唯一还没碰的信息维度。

## 三个必须处理对的地方

### 1. 前视：报告期 ≠ 公告日

2025 年报的报告期是 2025-12-31，但要到 2026-03 才公告。用报告期对齐价格
就是开天眼。本模块一律按 **announce_date** 对齐。

同一报告期还会被**追溯重述**（多年后重新披露）。取数规则是：
在日期 d，取所有 `announce_date <= d` 的记录里 report_date 最大的；
同一 report_date 有多版时取公告日最晚的（当时已知的最新修订版）。

### 2. 累计口径：EPS 每年归零

实测 000001 的 s_fa_eps_basic：

    2025-12-31  2.07     <- 全年
    2026-03-31  0.67     <- 一季度，归零重算

每年 4~8 月，已披露一季报的股票显示 0.67，尚未披露的还挂着上一年的 2.07。
直接做截面排序，等于拿一个季度的盈利去比一整年的 —— 排出来的不是
「谁更赚钱」，而是「谁还没发财报」。

因此**流量型**指标（EPS、每股经营现金流、ROE）一律转 TTM：

    TTM(t) = 累计(t) + 上年全年 - 上年同期累计

四季度报本身就是全年，直接用。**存量型**（每股净资产）和**比率型**
（毛利率、增长率）不需要转。

### 3. 估值分母要用真实价

每股指标按真实股本计算，配后复权价会得到无意义的比值 ——
后复权把历史价格整体抬高，越早的数据失真越大。

## 因子

    估值    bp   = 每股净资产 / 股价     （账面市值比，越高越便宜）
            ep   = 每股收益TTM / 股价    （盈利收益率）
            cfp  = 每股经营现金流TTM / 股价
    盈利    roe  = 净资产收益率TTM
            gross_margin = 销售毛利率
            net_margin   = 净利率
    成长    rev_growth    = 营收增长率
            profit_growth = 净利润增长率

用法::

    python scripts/build_fundamental_factors.py
    python scripts/build_fundamental_factors.py --start 2016-01-01
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: 流量型每股指标 -> 因子名。年内累计口径，必须转 TTM
FLOW = {
    "s_fa_eps_basic": "eps",        # 每股收益
    "s_fa_ocfps": "ocfps",          # 每股经营现金流
    "du_return_on_equity": "roe",   # 净资产收益率
}

#: 存量型与比率型 -> 因子名。不需要转 TTM
#:
#: 没有收 ``du_profit_rate``：字段名写着净利率，实测数值是
#: 98.28 -> -48.41 -> 13.47 这种量级跳变，是变化率而非利润率；
#: 且与 ``inc_net_profit_rate`` 相关系数 0.95，属近重复。
#: 对比 ``sales_gross_profit`` 稳定在 34~38%，那才是真正的margin行为。
STOCK = {
    "s_fa_bps": "bps",                       # 每股净资产（时点值）
    "sales_gross_profit": "gross_margin",    # 销售毛利率
    "inc_revenue_rate": "rev_growth",        # 营收增长率
    "inc_net_profit_rate": "profit_growth",  # 净利润增长率
}

#: 需要截面缩尾的因子。增长率在基数接近零时会飙到几百上千倍
#: （实测 1%/99% 分位是 -541 / +1241），几个极端值就能主导整个排序。
#: 树模型对此相对耐受，但缩尾不花成本，也让因子能被非树模型直接使用。
WINSOR = {"rev_growth", "profit_growth", "gross_margin", "roe", "cfp"}
WINSOR_Q = 0.01

#: 每股指标 / 股价 -> 估值因子。仅在有不复权价时构建，
#: 通常走 daily_basic 那条路（见下），这里是退路
VALUATION = [("bps", "bp"), ("eps", "ep"), ("ocfps", "cfp")]

#: Tushare daily_basic 派生的因子。这条线和上面完全不同：
#: 它是**逐日**数据，Tushare 已经处理好财报对齐，不需要 as-of，
#: 覆盖也宽得多（全市场 5500+ 只 vs 财报接口的几百只）。
#:
#: 倒数形式（1/pe 而非 pe）是必须的：pe 在盈利趋近 0 时发散到无穷，
#: 排序会被几只极端值主导；ep 则连续且可跨越零点。
DAILY_BASIC = {
    "bp": ("pb", "inv"),            # 账面市值比 = 1/市净率
    "ep": ("pe_ttm", "inv"),        # 盈利收益率 = 1/市盈率TTM
    "sp": ("ps_ttm", "inv"),        # 营收市值比 = 1/市销率TTM
    "dp": ("dv_ttm", "raw"),        # 股息率TTM，本身就是收益率形式
    "size": ("total_mv", "log"),    # 规模 = ln(总市值)
    "turnover": ("turnover_rate_f", "raw"),   # 自由流通换手率
    "vol_ratio": ("volume_ratio", "raw"),     # 量比
}

#: daily_basic 的可获得滞后（交易日）。当日指标要收盘后才算得出来
DB_LAG = 1


def to_ttm(df, col, first_by_report):
    """年内累计 -> 滚动十二个月。

    ``first_by_report`` 是 {report_date: 首次公告的值}。用**首发版本**
    而非最新修订版做减项，是因为修订可能发生在当前这行公告之后 ——
    那样又会引入前视。首发值是当时市场真正看到的数字。
    """
    import pandas as pd

    out = []
    for r, v in zip(df["report_date"], df[col]):
        if pd.isna(r) or pd.isna(v):
            out.append(float("nan"))
            continue
        if r.month == 12:          # 四季报即全年
            out.append(v)
            continue
        fy_prev = first_by_report.get(pd.Timestamp(r.year - 1, 12, 31))
        try:
            same_prev = first_by_report.get(
                pd.Timestamp(r.year - 1, r.month, r.day))
        except ValueError:         # 2 月 29 日之类
            same_prev = None
        if fy_prev is None or same_prev is None:
            out.append(float("nan"))
        else:
            out.append(v + fy_prev - same_prev)
    return pd.Series(out, index=df.index)


def asof_series(df, values, grid):
    """把不规则的财报序列铺到交易日网格上，按公告日对齐。

    做法：先按公告日排序，再对「report_date 的前缀最大值」取 argmax ——
    因为已按公告日排序，同一 report_date 的多个版本里靠后的自动胜出，
    正好是「当时已知的最新修订版」。最后用 searchsorted 把网格日期
    映射到位置。全程向量化，逐日调用 get_asof 在千万量级上跑不完。
    """
    import numpy as np
    import pandas as pd

    ann = df["announce_date"].values
    rep = df["report_date"].values.astype("datetime64[ns]").astype("int64")
    # 前缀 argmax：>= 让相同 report_date 时取靠后（公告更晚）那条
    best = np.zeros(len(rep), dtype=np.int64)
    cur = 0
    for i in range(1, len(rep)):
        if rep[i] >= rep[cur]:
            cur = i
        best[i] = cur

    pos = np.searchsorted(ann, np.asarray(grid, dtype="datetime64[ns]"),
                          side="right") - 1
    out = np.full(len(grid), np.nan)
    ok = pos >= 0
    if ok.any():
        out[ok] = np.asarray(values, dtype=float)[best[pos[ok]]]
    return pd.Series(out, index=grid)


def build_panels(root: Path, symbols: list, grid):
    """构建 {因子名: 日期 x 标的} 面板。"""
    import pandas as pd

    fields = {**FLOW, **STOCK}
    cols = ["report_date", "announce_date"] + list(fields)
    out = {name: {} for name in fields.values()}
    n_ok = n_ttm = 0
    t0 = time.time()

    for i, (vt, path) in enumerate(symbols, 1):
        try:
            df = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        have = [c for c in cols if c in df.columns]
        if "announce_date" not in have or "report_date" not in have:
            continue
        df = df[have].dropna(subset=["announce_date"])
        df = df.sort_values(["announce_date", "report_date"])
        if df.empty:
            continue
        n_ok += 1

        # 首发值表：同一 report_date 保留最早公告的那版
        first = df.drop_duplicates("report_date", keep="first")

        for col, name in fields.items():
            if col not in df.columns:
                continue
            if col in FLOW:
                fb = dict(zip(first["report_date"], first[col]))
                vals = to_ttm(df, col, fb)
                n_ttm += 1
            else:
                vals = df[col]
            # 滞后一日：asof_series 在公告当日就把值放出来了，
            # 但财报多在盘后披露，当日不可用。与 daily_basic 那条线
            # （已在 build_daily_basic 里 shift 过）保持同一口径，
            # 消费方一律按 lag=0 注册，不再重复滞后
            s = asof_series(df, vals.values, grid).shift(1)
            if s.notna().any():
                out[name][vt] = s

        if i % 500 == 0:
            print(f"  {i}/{len(symbols)}  有数据 {n_ok}  "
                  f"({time.time() - t0:.0f}s)")

    panels = {n: pd.DataFrame(d) for n, d in out.items() if d}
    return panels, n_ok


def build_daily_basic(store: Path, grid):
    """从 Tushare daily_basic 构建估值/规模/流动性因子。

    与财报那条线的关键差别：这里**不需要按公告日对齐** ——
    Tushare 每天用当时已披露的最新财报算好指标，本身就是时点值。
    只需整体滞后一日（当日指标收盘后才算得出来）。

    ## pe_ttm 缺失不是「未知」，是「不赚钱」

    2025 年 pe_ttm 有 27% 为空。这不是数据缺陷 —— 亏损公司的市盈率
    没有意义，Tushare 直接留空。这里**保留 NaN 而不填充**：
    LightGBM 原生支持缺失值，会自己学到「缺失该往哪边分」，
    比我硬填一个数更可靠。同时单独给出 ``fund_is_loss`` 显式标记，
    让这个信息不至于只藏在缺失模式里。
    """
    import numpy as np
    import pandas as pd

    root = store / "factor" / "daily_basic"
    files = sorted(root.glob("*.parquet"))
    if not files:
        return {}

    need = {"symbol", "trade_date"} | {c for c, _ in DAILY_BASIC.values()}
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        keep = [c for c in need if c in d.columns]
        frames.append(d[keep])
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    print(f"  daily_basic: {len(df):,} 行  {df['symbol'].nunique()} 只"
          f"  {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")

    out = {}
    for name, (col, how) in DAILY_BASIC.items():
        if col not in df.columns:
            continue
        panel = df.pivot_table(index="trade_date", columns="symbol",
                               values=col, aggfunc="last")
        if how == "inv":
            # 倒数前把 0 变 NaN —— 除零会产生 inf，之后所有排序都被它吃掉
            panel = 1.0 / panel.replace(0, np.nan)
        elif how == "log":
            panel = np.log(panel.where(panel > 0))
        # 滞后一日后铺到交易日网格
        out[name] = panel.sort_index().shift(DB_LAG).reindex(grid).ffill(limit=5)

    # 亏损标记：有市销率（说明是正常在市公司）但没有市盈率
    if "pe_ttm" in df.columns and "ps_ttm" in df.columns:
        pe = df.pivot_table(index="trade_date", columns="symbol",
                            values="pe_ttm", aggfunc="last").sort_index()
        ps = df.pivot_table(index="trade_date", columns="symbol",
                            values="ps_ttm", aggfunc="last").sort_index()
        loss = (pe.isna() & ps.notna()).astype(float)
        loss = loss.where(ps.notna())      # 完全没数据的日子仍是 NaN
        out["is_loss"] = loss.shift(DB_LAG).reindex(grid).ffill(limit=5)

    return out


def load_prices(store: Path, grid):
    """真实价（不复权）。估值因子的分母必须是真实股价 ——
    ``data/1d`` 是后复权，用它算出的每股/股价没有意义。

    daily_basic 里的 ``close`` 正是不复权价，且覆盖全市场，
    优先用它；``data/1d_raw`` 是早期的另一条路径，作为退路保留。
    """
    import pandas as pd

    db = store / "factor" / "daily_basic"
    files = sorted(db.glob("*.parquet"))
    if files:
        frames = []
        for f in files:
            d = pd.read_parquet(f, columns=["symbol", "trade_date", "close"])
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        panel = df.pivot_table(index="trade_date", columns="symbol",
                               values="close", aggfunc="last")
        return panel.sort_index().reindex(grid).ffill(limit=5), True

    raw = store / "1d_raw"
    cols = {}
    if raw.exists():
        for p in raw.rglob("*.parquet"):
            try:
                d = pd.read_parquet(p, columns=["close"])
            except (OSError, ValueError, KeyError):
                continue
            if not isinstance(d.index, pd.DatetimeIndex):
                d.index = pd.to_datetime(d.index.astype(str),
                                         format="%Y%m%d", errors="coerce")
            cols[f"{p.stem}.{p.parent.name}"] = d["close"]
    if cols:
        return pd.DataFrame(cols).reindex(grid).ffill(limit=5), True
    return None, False


def main() -> int:
    p = argparse.ArgumentParser(description="构建基本面因子")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args()

    import pandas as pd

    from qmtquant.config import get_config
    from qmtquant.datafeed.qlib_export import to_qlib_code

    cfg = get_config()
    store = Path(cfg.data.store_dir)

    print("=" * 62)
    print("构建基本面因子")
    print("=" * 62)

    cal_file = store / "qlib_data" / "calendars" / "day.txt"
    if not cal_file.exists():
        print("缺少交易日历，请先运行 scripts/export_qlib.py")
        return 1
    grid = pd.to_datetime(cal_file.read_text(encoding="utf-8").split())
    grid = grid[grid >= args.start]
    if args.end:
        grid = grid[grid <= args.end]

    print(f"交易日 {len(grid)} 天  {grid[0].date()} ~ {grid[-1].date()}")
    factors = {}

    # ---- 线路一：daily_basic（逐日，覆盖全市场，无需对齐）
    print("\n[1/2] Tushare daily_basic —— 估值 / 规模 / 流动性")
    db = build_daily_basic(store, grid)
    for name, panel in db.items():
        factors[f"fund_{name}"] = panel
    print(f"  得到 {len(db)} 个因子")

    # ---- 线路二：财报（按公告日 as-of + TTM，覆盖窄但含盈利质量与成长）
    fin_root = store / "financial" / "PershareIndex"
    symbols = sorted({(f"{q.stem}.{q.parent.name}", q)
                      for q in fin_root.rglob("*.parquet")}) \
        if fin_root.exists() else []

    if not symbols:
        print("\n[2/2] 无财报数据，跳过盈利质量与成长因子")
        print("      补齐：python scripts/download_financial.py --sector 沪深A股")
        panels = {}
    else:
        print(f"\n[2/2] 财报 —— 盈利质量 / 成长（{len(symbols)} 只）")
        print(f"  TTM 转换: {', '.join(FLOW.values())}"
              f"   直接使用: {', '.join(STOCK.values())}")
        panels, n_ok = build_panels(fin_root, symbols, grid)
        print(f"  {n_ok} 只有有效数据，得到 {len(panels)} 个因子")

        # daily_basic 已有的（bp/ep）不覆盖 —— 它覆盖面宽得多
        for n in list(STOCK.values()) + ["roe"]:
            if n in panels and f"fund_{n}" not in factors:
                factors[f"fund_{n}"] = panels[n]

    # 财报口径的估值因子只在 daily_basic 缺该项时才补
    missing_val = [v for _, v in VALUATION if f"fund_{v}" not in factors]
    if missing_val and panels:
        prices, _ = load_prices(store, grid)
        if prices is None:
            print(f"\n无不复权价，跳过 {'/'.join(missing_val)}")
        else:
            for per, val in VALUATION:
                if val not in missing_val or per not in panels:
                    continue
                num = panels[per]
                shared = num.columns.intersection(prices.columns)
                if len(shared) == 0:
                    continue
                factors[f"fund_{val}"] = (
                    num[shared] / prices[shared].replace(0, pd.NA))

    out_dir = store / "qlib_data" / "money_factors"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'因子':<20s}{'标的':>7s}{'日均有值':>9s}{'中位数':>11s}"
          f"{'1%':>10s}{'99%':>10s}")
    print("-" * 68)
    written = 0
    for name in sorted(factors):
        panel = factors[name]

        if name.removeprefix("fund_") in WINSOR:
            # 逐日截面缩尾。必须按日做 —— 按整体分位会把牛市整段
            # 或熊市整段裁掉，那是抹掉时序信息而不是去极值
            lo = panel.quantile(WINSOR_Q, axis=1)
            hi = panel.quantile(1 - WINSOR_Q, axis=1)
            panel = panel.clip(lo, hi, axis=0)

        ren = {}
        for c in panel.columns:
            try:
                ren[c] = to_qlib_code(str(c))
            except (ValueError, KeyError):
                pass
        panel = panel[list(ren)].rename(columns=ren)
        if panel.empty:
            continue

        s = panel.stack()
        s.index.names = ["datetime", "instrument"]
        s.name = name
        s.to_frame().to_parquet(out_dir / f"{name}.parquet")
        written += 1

        nn = panel.notna().sum(axis=1).mean()
        print(f"{name:<20s}{panel.shape[1]:>7d}{nn:>9.0f}"
              f"{s.median():>11.4f}{s.quantile(.01):>10.3f}"
              f"{s.quantile(.99):>10.3f}")

    print(f"\n写入 {written} 个因子 -> {out_dir}")
    print("\n下一步：三关检验")
    print("  python strategies/alstm_ppo_csi1000/eval_factors.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
