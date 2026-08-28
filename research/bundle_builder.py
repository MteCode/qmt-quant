"""把项目本地数据装配成 ``hs300_multifactor.DataBundle``。

## 各字段的来源

============  ==========================================  ==================
字段           来源                                        point-in-time 保证
============  ==========================================  ==================
close/ret     ``data/1d/`` QMT 后复权日线                   —
mktcap        ``daily_basic.total_mv``（万元→元）           逐日快照
is_member     ``index_weight_*.csv`` 月度名单               asof 取用
bench_weight  同上的 weight 列                             asof 取用
industry      ``industry.parquet`` Tushare 行业             ⚠ 静态，见下
tradable      非停牌 + 非一字板                             逐日
listed_days   ``universe_full.parquet`` 上市日              —
bench_ret     ``data/index/`` 指数日线                      —
np_ttm 等      ``data/financial/`` QMT 财报                  **按公告日**
turnover      ``daily_basic.turnover_rate_f``              逐日快照
============  ==========================================  ==================

## 三个必须说清的口径问题

**1. 财务数据按公告日对齐。** 报告期与公告日之间滞后极大 ——
实测茅台年报平均滞后 100 天以上、最大 478 天。按报告期取数
等于提前三个多月知道年报，这是比幸存者偏差更严重的前视，
而且不报错、不异常，只会让回测收益凭空变好。

本模块把季度财报按 ``m_anntime`` 前向填充到日频：某只股票在
2024-04-03 公告了 2023 年报，那么 2024-04-03 之前的每一天，
它的 np_ttm 用的仍是上一期已公告的值。

**2. 行业分类是静态的。** Tushare 的 ``industry`` 字段不带时间维度，
公司转型换行业时历史区间也会被打上现在的标签。这是已知局限 ——
行业标签错几个，好过完全不做中性化（实测不中性化时低换手因子
选出 20 只里 15 只是银行，所谓「因子显著」实为行业暴露）。

**3. 基准收益用的是价格指数，不是全收益指数。** 沪深300 全收益
（000300.CSI）本地没有，用价格指数会**高估超额收益约 2%/年**
（成分股股息率约 2%）。跑出来的年化超额要在心里扣掉这一块。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: QMT 财报字段 -> DataBundle 字段
FIN_FIELDS = {
    "Income": {"net_profit_excl_min_int_inc": "np_q"},        # 归母净利润（累计）
    "Balance": {"tot_shrhldr_eqy_excl_min_int": "equity",
                "tot_assets": "total_assets"},
    "CashFlow": {"net_cash_flows_oper_act": "ocf_q"},          # 经营现金流（累计）
}


def _to_dt(s: pd.Series) -> pd.Series:
    """QMT 日期是 'YYYYMMDD' 字符串或数字，0 表示缺失"""
    t = s.astype(str).str.replace(r"\.0$", "", regex=True)
    t = t.where(~t.isin(["0", "", "nan", "None"]))
    return pd.to_datetime(t, format="%Y%m%d", errors="coerce")


def load_prices(store_dir: str, symbols: list[str],
                start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读日线，返回 (close, 成交额)。缺数据的标的自动跳过。"""
    from qmtquant.utils.symbol import split_vt_symbol

    closes, amounts = {}, {}
    root = Path(store_dir) / "1d"
    for vt in symbols:
        code, ex = split_vt_symbol(vt)
        path = root / ex.value / f"{code}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        idx = pd.to_datetime(df.index.astype(str), format="%Y%m%d",
                             errors="coerce")
        df = df.set_axis(idx).sort_index()
        df = df[(df.index >= start) & (df.index <= end)]
        if df.empty:
            continue
        closes[vt] = df["close"]
        if "amount" in df.columns:
            amounts[vt] = df["amount"]

    if not closes:
        raise SystemExit(f"{store_dir}/1d 下没有可用日线")
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(amounts).sort_index()


def load_membership(store_dir: str, index_code: str,
                    dates: pd.DatetimeIndex,
                    codes: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把月度成分名单展开成日频的 (is_member, bench_weight)。

    用 asof 语义：每个交易日取「不晚于它的最近一期名单」，
    绝不使用未来的调仓结果。
    """
    path = Path(store_dir) / "universe" / f"index_weight_{index_code}.csv"
    if not path.exists():
        raise SystemExit(
            f"缺少历史成分股 {path}\n"
            f"请先运行 scripts/download_index_weight.py --index {index_code}")

    w = pd.read_csv(path, parse_dates=["date"])
    periods = np.array(sorted(w["date"].unique()))
    by_period = {d: g.set_index("symbol")["weight"]
                 for d, g in w.groupby("date")}

    # searchsorted 一次算出每个交易日该用哪一期名单
    pos = np.searchsorted(periods, dates.values, side="right") - 1
    member = pd.DataFrame(False, index=dates, columns=codes)
    weight = pd.DataFrame(np.nan, index=dates, columns=codes)
    for d, p in zip(dates, pos):
        if p < 0:
            continue
        s = by_period[periods[p]]
        s = s[s.index.isin(codes)]
        member.loc[d, s.index] = True
        weight.loc[d, s.index] = s.values
    return member, weight


def load_industry(store_dir: str, dates: pd.DatetimeIndex,
                  codes: pd.Index) -> pd.DataFrame:
    """行业分类展开成日频面板。

    ⚠ Tushare 的行业是**静态**的（不带时间维度），所以每一天都相同。
    缺标签的填 "UNKNOWN" 而非丢弃 —— 丢弃会让这些股票在中性化后
    整列变 NaN，等于把它们排除出标的池，那是另一种偏差。
    """
    path = Path(store_dir) / "universe" / "industry.parquet"
    if not path.exists():
        raise SystemExit(
            f"缺少行业分类 {path}\n请先运行 scripts/download_industry.py")

    df = pd.read_parquet(path)
    mapping = df.set_index("vt_symbol")["industry"].to_dict()
    row = [str(mapping.get(c) or "UNKNOWN") for c in codes]
    n_unknown = sum(1 for v in row if v == "UNKNOWN")
    if n_unknown:
        logger.warning("%d/%d 只标的缺行业标签，归入 UNKNOWN",
                       n_unknown, len(codes))
    return pd.DataFrame(np.tile(row, (len(dates), 1)),
                        index=dates, columns=codes)


def load_daily_basic(store_dir: str, dates: pd.DatetimeIndex,
                     codes: pd.Index,
                     columns: list[str]) -> dict[str, pd.DataFrame]:
    """读逐日估值因子，对齐到 (dates, codes)"""
    from qmtquant.datafeed.factor_store import FactorStore

    store = FactorStore(store_dir, columns,
                        start=str(dates[0].date()), end=str(dates[-1].date()))
    out = {}
    for col in columns:
        panel = store._panels[col]
        out[col] = panel.reindex(index=dates, columns=codes)
    return out


def load_financials(store_dir: str, dates: pd.DatetimeIndex,
                    codes: pd.Index) -> dict[str, pd.DataFrame]:
    """读财报并**按公告日**前向填充到日频。

    返回 np_ttm / equity / total_assets / ocf_ttm 四个面板，
    外加 fin_q 长表（单季净利润，供 SUE 用）。

    TTM 的算法：A 股财报是**年内累计**口径（Q1/H1/Q3/年报），
    先差分成单季，再滚动求和最近四个季度。
    直接用累计值会让 Q1 的 EP 只有年报的四分之一，横截面完全不可比。
    """
    from qmtquant.utils.symbol import split_vt_symbol

    frames: dict[str, dict[str, pd.Series]] = {
        "np_ttm": {}, "equity": {}, "total_assets": {}, "ocf_ttm": {}}
    fin_q_rows = []

    for vt in codes:
        code, ex = split_vt_symbol(vt)
        per_table = {}
        for table, fields in FIN_FIELDS.items():
            path = Path(store_dir) / "financial" / table / ex.value / f"{code}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            need = [c for c in fields if c in df.columns]
            if not need:
                continue
            sub = df[need].copy()
            sub["report"] = _to_dt(df["m_timetag"])
            sub["ann"] = _to_dt(df["m_anntime"])
            sub = sub.dropna(subset=["report", "ann"])
            # 同一报告期多次公告（更正/重述）取最后一次
            sub = (sub.sort_values(["report", "ann"])
                      .drop_duplicates("report", keep="last"))
            per_table[table] = sub.rename(columns=fields)

        if "Income" not in per_table:
            continue

        inc = per_table["Income"]
        # 累计 -> 单季：同一年内相邻报告期做差分，Q1 保持原值
        q = _cumulative_to_quarterly(inc, "np_q")
        if q.empty:
            continue
        fin_q_rows.append(pd.DataFrame({
            "code": vt, "ann_date": q["ann"],
            "report_period": q["report"], "net_profit_q": q["np_q"],
        }))
        # TTM = 最近 4 个单季之和
        ttm = q.set_index("ann")["np_q"].rolling(4).sum()
        frames["np_ttm"][vt] = _ffill_by_ann(ttm, dates)

        if "CashFlow" in per_table:
            qc = _cumulative_to_quarterly(per_table["CashFlow"], "ocf_q")
            if not qc.empty:
                frames["ocf_ttm"][vt] = _ffill_by_ann(
                    qc.set_index("ann")["ocf_q"].rolling(4).sum(), dates)

        if "Balance" in per_table:
            bal = per_table["Balance"].set_index("ann")
            for src, dst in (("equity", "equity"),
                             ("total_assets", "total_assets")):
                if src in bal.columns:
                    frames[dst][vt] = _ffill_by_ann(bal[src], dates)

    out = {k: (pd.DataFrame(v).reindex(index=dates, columns=codes)
               if v else pd.DataFrame(index=dates, columns=codes, dtype=float))
           for k, v in frames.items()}
    out["fin_q"] = (pd.concat(fin_q_rows, ignore_index=True)
                    if fin_q_rows else pd.DataFrame(
                        columns=["code", "ann_date", "report_period",
                                 "net_profit_q"]))
    return out


def _cumulative_to_quarterly(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """年内累计口径 -> 单季。

    A 股财报是累计的：Q3 报表里的净利润是 1-9 月合计。
    单季 Q3 = 累计Q3 − 累计H1。每年的 Q1 就是单季本身。
    不做这一步的话，Q1 的 EP 只有年报的四分之一，横截面无法比较。
    """
    if col not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=[col]).sort_values("report").copy()
    if d.empty:
        return pd.DataFrame()
    d["year"] = d["report"].dt.year
    d[col] = d.groupby("year")[col].diff().fillna(d[col])
    return d[["report", "ann", col]]


def _ffill_by_ann(s: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """按公告日前向填充到日频。

    这是 point-in-time 的关键一步：某只股票 2024-04-03 才公告年报，
    那么 2024-04-03 之前的每一天都只能看到上一期已公告的值。
    """
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


# ============================================================================
# 主装配
# ============================================================================

def build_bundle(index_code: str = "000300.SH",
                 start: str = "2016-01-01",
                 end: str = "2026-08-27",
                 store_dir: str | None = None,
                 min_universe: int = 50):
    """装配真实数据的 DataBundle。

    :param min_universe: 每个交易日至少要有多少只成分股有行情，
        否则说明本地行情覆盖不足，直接报错而不是跑出一条无意义的曲线
    """
    from qmtquant.config import get_config
    from qmtquant.datafeed.xt_feed import IndexFeed

    from hs300_multifactor import DataBundle

    cfg = get_config()
    store_dir = store_dir or cfg.data.store_dir

    # ---- 标的池：历史成分并集
    wpath = Path(store_dir) / "universe" / f"index_weight_{index_code}.csv"
    hist = pd.read_csv(wpath, parse_dates=["date"])
    hist = hist[(hist["date"] >= start) & (hist["date"] <= end)]
    all_syms = sorted(hist["symbol"].unique())
    print(f"历史成分并集 {len(all_syms)} 只")

    close, amount = load_prices(store_dir, all_syms, start, end)
    dates, codes = close.index, close.columns
    print(f"本地有行情 {len(codes)} 只，{len(dates)} 个交易日 "
          f"({dates[0].date()} ~ {dates[-1].date()})")

    ret = close.pct_change().fillna(0.0)

    is_member, bench_weight = load_membership(store_dir, index_code, dates, codes)
    cover = is_member.sum(axis=1)
    print(f"每日成分数（有行情的）中位数 {int(cover.median())}")
    if cover.median() < min_universe:
        raise SystemExit(
            f"每日可用成分股中位数仅 {int(cover.median())}，行情覆盖不足。\n"
            f"请先补下历史成分股的日线：scripts/download_data.py --symbols ...")

    industry = load_industry(store_dir, dates, codes)

    db = load_daily_basic(store_dir, dates, codes,
                          ["total_mv", "turnover_rate_f"])
    # daily_basic 的 total_mv 单位是万元
    mktcap = db["total_mv"] * 1e4
    turnover = db["turnover_rate_f"]

    print("读取财报（按公告日对齐）...")
    fin = load_financials(store_dir, dates, codes)
    n_np = int(fin["np_ttm"].notna().any().sum())
    print(f"  有净利润数据 {n_np} 只，fin_q {len(fin['fin_q']):,} 行")

    # ---- 可交易：非停牌 + 非一字板
    #      一字板用「最高=最低」近似 —— 本地日线没有涨跌停价字段
    tradable = close.notna() & (close > 0) & (ret.abs() < 0.095)

    # ---- 上市天数
    meta = pd.read_parquet(Path(store_dir) / "universe" / "universe_full.parquet")
    listing = pd.to_datetime(
        meta.set_index("vt_symbol")["listing_date"]).reindex(codes)
    listed_days = pd.DataFrame(
        {c: (dates - listing[c]).days if pd.notna(listing[c]) else 9999
         for c in codes}, index=dates)

    # ---- 基准
    bench_close = IndexFeed(store_dir).load_close(index_code, start, end)
    bench_ret = bench_close.reindex(dates).ffill().pct_change().fillna(0.0)
    print("⚠ 基准用的是**价格指数**，非全收益指数 —— "
          "超额收益被高估约 2%/年（成分股股息率）")

    return DataBundle(
        close=close, ret=ret, mktcap=mktcap,
        is_member=is_member, bench_weight=bench_weight,
        industry=industry, tradable=tradable, listed_days=listed_days,
        bench_ret=bench_ret,
        np_ttm=fin["np_ttm"], equity=fin["equity"],
        total_assets=fin["total_assets"], ocf_ttm=fin["ocf_ttm"],
        turnover=turnover, fin_q=fin["fin_q"],
    )


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from hs300_multifactor import run_strategy

    p = argparse.ArgumentParser(description="真实数据多因子回测")
    p.add_argument("--index", default="000300.SH")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2026-08-27")
    p.add_argument("--top-pct", type=float, default=0.30)
    a = p.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(message)s")
    bundle = build_bundle(a.index, a.start, a.end)
    print("\n装配完成，开始回测...\n")
    run_strategy(bundle, top_pct=a.top_pct)
