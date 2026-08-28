"""
沪深300 指数增强 —— 多因子策略回测框架
=====================================

设计原则
--------
1. 所有数据接口都是 date(index) x code(columns) 的 DataFrame，你自己接数据源。
2. 严格避免未来函数：财务数据按公告日(ann_date)对齐；信号在 T 日收盘算，
   T+1 日收盘建仓，收益从 T+2 起算。
3. 成分股必须是 point-in-time 的真实历史名单，否则幸存者偏差能凭空造出 3~5% 年化。

用法
----
    python hs300_multifactor.py --demo      # 用合成数据自测框架
    # 实盘：填好 DataBundle，调用 run_strategy(bundle)

作者备注：这是研究框架，不是投资建议。参数是我给的先验，不是拟合出来的最优值，
请务必做样本外检验（建议 2018 年以前调参，2018 年以后一次都别回头看）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# 数据容器
# ============================================================================

@dataclass
class DataBundle:
    """
    所有面板数据统一为 date(DatetimeIndex) x code(str columns)。

    必填
    ----
    close        : 后复权收盘价
    ret          : 日收益率（由 close 算，或直接给，注意要和 close 一致）
    mktcap       : 总市值（元）
    is_member    : bool, 是否为当日沪深300成分股（point-in-time！）
    bench_weight : 指数成分权重（%或小数均可，内部会归一化），非成分为 NaN
    industry     : str, 一级行业代码（中信/申万，保持全历史一致）
    tradable     : bool, 当日可成交（非停牌、非一字涨跌停、非ST）
    listed_days  : int, 上市天数
    bench_ret    : Series, 基准日收益（用全收益指数！）

    因子原料（缺哪个就跳过哪个因子）
    ----
    np_ttm       : 归母净利润 TTM
    equity       : 归母净资产
    total_assets : 总资产
    ocf_ttm      : 经营活动现金流净额 TTM
    turnover     : 日换手率（自由流通口径更好）

    fin_q : 长表，用于 SUE，列 = [code, ann_date, report_period, net_profit_q]
            net_profit_q 是**单季**归母净利润
    """
    close: pd.DataFrame
    ret: pd.DataFrame
    mktcap: pd.DataFrame
    is_member: pd.DataFrame
    bench_weight: pd.DataFrame
    industry: pd.DataFrame
    tradable: pd.DataFrame
    listed_days: pd.DataFrame
    bench_ret: pd.Series

    np_ttm: Optional[pd.DataFrame] = None
    equity: Optional[pd.DataFrame] = None
    total_assets: Optional[pd.DataFrame] = None
    ocf_ttm: Optional[pd.DataFrame] = None
    turnover: Optional[pd.DataFrame] = None
    fin_q: Optional[pd.DataFrame] = None


# ============================================================================
# 预处理工具
# ============================================================================

def mad_winsorize(s: pd.Series, n: float = 5.0) -> pd.Series:
    """MAD 去极值。比 3-sigma 稳健，A股财务因子必须做。"""
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return s
    return s.clip(med - n * mad, med + n * mad)


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def neutralize_cs(y: pd.Series,
                  lncap: pd.Series,
                  ind: pd.Series,
                  wgt: pd.Series) -> pd.Series:
    """
    单截面市值+行业中性化：对 ln(市值) 和行业哑变量做 WLS，返回残差。
    权重用 sqrt(市值)，避免小票主导回归。
    """
    df = pd.concat([y.rename("y"), lncap.rename("cap"), ind.rename("ind")],
                   axis=1).dropna()
    if len(df) < 20:
        return pd.Series(np.nan, index=y.index)

    X = pd.get_dummies(df["ind"], prefix="i").astype(float)
    X["cap"] = df["cap"].values
    Xv = X.values
    yv = df["y"].values

    w = wgt.reindex(df.index).fillna(0.0).clip(lower=0.0).values
    w = np.sqrt(w)
    w = w / (w.mean() if w.mean() > 0 else 1.0)

    Xw = Xv * w[:, None]
    yw = yv * w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yv - Xv @ beta
    return pd.Series(resid, index=df.index).reindex(y.index)


def prep_factor(raw: pd.DataFrame,
                dates: pd.DatetimeIndex,
                mktcap: pd.DataFrame,
                industry: pd.DataFrame,
                universe: pd.DataFrame) -> pd.DataFrame:
    """去极值 -> 中性化 -> 标准化，只在 universe 内做截面处理。"""
    out = {}
    lncap_all = np.log(mktcap.replace(0, np.nan))
    for d in dates:
        if d not in raw.index:
            continue
        mask = universe.loc[d]
        codes = mask[mask].index
        y = raw.loc[d, codes].astype(float).dropna()
        if len(y) < 30:
            continue
        y = mad_winsorize(y)
        r = neutralize_cs(y,
                          lncap_all.loc[d, y.index],
                          industry.loc[d, y.index],
                          mktcap.loc[d, y.index])
        out[d] = zscore(r.dropna())
    if not out:
        return pd.DataFrame(index=dates, columns=raw.columns, dtype=float)
    return pd.DataFrame(out).T.reindex(columns=raw.columns)


# ============================================================================
# 因子计算
# ============================================================================

def build_raw_factors(b: DataBundle, dates: pd.DatetimeIndex) -> Dict[str, pd.DataFrame]:
    """在调仓日截面上计算原始因子值（未处理）。"""
    f: Dict[str, pd.DataFrame] = {}
    mc = b.mktcap

    # ---- 价值 ----
    if b.np_ttm is not None:
        f["EP"] = (b.np_ttm / mc).reindex(dates)
    if b.equity is not None:
        f["BP"] = (b.equity / mc).reindex(dates)

    # ---- 质量 ----
    if b.np_ttm is not None and b.equity is not None:
        avg_eq = (b.equity + b.equity.shift(250)) / 2.0
        f["ROE"] = (b.np_ttm / avg_eq.replace(0, np.nan)).reindex(dates)
    if b.np_ttm is not None and b.ocf_ttm is not None and b.total_assets is not None:
        acc = (b.np_ttm - b.ocf_ttm) / b.total_assets.replace(0, np.nan)
        f["ACCRUAL"] = (-acc).reindex(dates)          # 应计低 = 盈利质量高

    # ---- 量价 ----
    # 特异波动率：对市场收益做 60 日滚动回归，取残差波动，负号
    mkt = b.bench_ret.reindex(b.ret.index).fillna(0.0)
    win = 60
    var_m = mkt.rolling(win).var()
    cov = b.ret.rolling(win).cov(mkt)
    beta = cov.div(var_m, axis=0)
    alpha = b.ret.rolling(win).mean().sub(beta.mul(mkt.rolling(win).mean(), axis=0))
    resid = b.ret.sub(beta.mul(mkt, axis=0), axis=0).sub(alpha)
    f["IVOL"] = (-resid.rolling(win).std()).reindex(dates)

    # 短期反转
    f["REV20"] = (-(b.close / b.close.shift(20) - 1.0)).reindex(dates)

    # 换手率（负）
    if b.turnover is not None:
        f["TURN20"] = (-b.turnover.rolling(20).mean()).reindex(dates)

    # ---- 预期 ----
    if b.fin_q is not None:
        f["SUE"] = build_sue(b.fin_q, dates, b.close.columns)

    return f


def build_sue(fin_q: pd.DataFrame,
              dates: pd.DatetimeIndex,
              codes: pd.Index,
              n_quarters: int = 12,
              min_diffs: int = 5) -> pd.DataFrame:
    """
    Foster (1984) SUE，严格按公告日对齐。

        d_t   = E_t - E_{t-4}                （单季净利润同比差分）
        SUE_t = (d_t - mean(d_{t-1..t-k})) / std(d_{t-1..t-k})

    fin_q 列: code, ann_date, report_period, net_profit_q
    """
    fq = fin_q.copy()
    fq["ann_date"] = pd.to_datetime(fq["ann_date"])
    fq["report_period"] = pd.to_datetime(fq["report_period"])
    fq = fq.sort_values(["code", "report_period", "ann_date"])

    out = {}
    for d in dates:
        sub = fq[fq["ann_date"] <= d]
        if sub.empty:
            continue
        # 同一报告期多次公告，取最后一次已披露的
        sub = sub.drop_duplicates(["code", "report_period"], keep="last")
        sub = sub.groupby("code").tail(n_quarters)

        vals = {}
        for code, g in sub.groupby("code"):
            e = g.sort_values("report_period")["net_profit_q"].values
            if len(e) < 4 + min_diffs + 1:
                continue
            diffs = e[4:] - e[:-4]
            if len(diffs) < min_diffs + 1:
                continue
            latest, base = diffs[-1], diffs[:-1]
            sd = base.std(ddof=1)
            if not np.isfinite(sd) or sd <= 0:
                continue
            vals[code] = (latest - base.mean()) / sd
        if vals:
            out[d] = pd.Series(vals)

    if not out:
        return pd.DataFrame(index=dates, columns=codes, dtype=float)
    return pd.DataFrame(out).T.reindex(index=dates, columns=codes)


#: 默认因子分组。
#:
#: ⚠ **SUE 已从 growth 大类移除。** 真实数据实测 t=-0.64，
#: 完全无效，却因为独占 growth 大类而拿到 1/4 的合成权重 ——
#: 等于把 25% 的信号预算分配给了噪声。
#: 需要对照时用 FACTOR_GROUPS_WITH_SUE。
FACTOR_GROUPS: Dict[str, List[str]] = {
    "value":   ["EP", "BP"],
    "quality": ["ROE", "ACCRUAL"],
    "pv":      ["IVOL", "REV20", "TURN20"],
}

#: 含 SUE 的原始分组，仅供对照
FACTOR_GROUPS_WITH_SUE: Dict[str, List[str]] = {
    **FACTOR_GROUPS, "growth": ["SUE"],
}


def combine_factors(proc: Dict[str, pd.DataFrame],
                    groups: Dict[str, List[str]] = None,
                    group_weights: Dict[str, float] = None) -> pd.DataFrame:
    """大类内等权 -> 再标准化 -> 大类间加权。默认大类等权。"""
    groups = groups or FACTOR_GROUPS
    grp_scores = {}
    for g, names in groups.items():
        avail = [n for n in names if n in proc]
        if not avail:
            continue
        s = sum(proc[n].fillna(0.0) * proc[n].notna() for n in avail)
        cnt = sum(proc[n].notna().astype(float) for n in avail)
        s = s.where(cnt > 0) / cnt.replace(0, np.nan)
        grp_scores[g] = s.apply(lambda r: zscore(r.dropna()).reindex(r.index), axis=1)

    if not grp_scores:
        raise ValueError("没有任何可用因子")
    gw = group_weights or {g: 1.0 / len(grp_scores) for g in grp_scores}
    total = sum(grp_scores[g] * gw.get(g, 0.0) for g in grp_scores)
    return total


# ============================================================================
# 单因子检验
# ============================================================================

def factor_ic(factor: pd.DataFrame,
              fwd_ret: pd.DataFrame,
              method: str = "spearman") -> pd.Series:
    """Rank IC 序列。"""
    ics = {}
    for d in factor.index:
        if d not in fwd_ret.index:
            continue
        x = factor.loc[d].dropna()
        y = fwd_ret.loc[d].reindex(x.index).dropna()
        x = x.reindex(y.index)
        if len(x) < 30:
            continue
        ics[d] = x.corr(y, method=method)
    return pd.Series(ics).sort_index()


def ic_report(factors: Dict[str, pd.DataFrame], fwd_ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, f in factors.items():
        ic = factor_ic(f, fwd_ret).dropna()
        if len(ic) < 12:
            continue
        icir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
        t = ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else np.nan
        rows.append({
            "factor": name,
            "IC_mean": ic.mean(),
            "IC_std": ic.std(),
            "ICIR": icir,
            "ICIR_ann": icir * np.sqrt(12),
            "t_stat": t,
            "IC>0 占比": (ic > 0).mean(),
            "n": len(ic),
        })
    return pd.DataFrame(rows).set_index("factor").round(4)


# ============================================================================
# 组合构建（行业严格中性）
# ============================================================================

def build_weights(score: pd.DataFrame,
                  b: DataBundle,
                  universe: pd.DataFrame,
                  top_pct: float = 0.30,
                  max_active: float = 0.01,
                  max_abs: float = 0.03) -> pd.DataFrame:
    """
    行业内选股，行业权重锁定为基准行业权重。
    行业内按 sqrt(基准权重) 分配，再做单票上限截断+再分配。
    """
    rows = {}
    for d in score.index:
        if d not in universe.index:
            continue
        mask = universe.loc[d]
        codes = mask[mask].index
        if len(codes) < 50:
            continue

        sc = score.loc[d, codes].dropna()
        if len(sc) < 50:
            continue
        ind = b.industry.loc[d, sc.index]
        bw = b.bench_weight.loc[d].reindex(sc.index).fillna(0.0)
        bw = bw / bw.sum() if bw.sum() > 0 else bw

        # 基准行业权重（用全部成分股算，不只是可交易的）
        bw_full = b.bench_weight.loc[d].dropna()
        bw_full = bw_full / bw_full.sum()
        ind_full = b.industry.loc[d].reindex(bw_full.index)
        ind_target = bw_full.groupby(ind_full).sum()

        w = pd.Series(0.0, index=sc.index)
        for iname, g in sc.groupby(ind):
            tgt = ind_target.get(iname, 0.0)
            if tgt <= 0 or len(g) == 0:
                continue
            k = max(1, int(np.ceil(len(g) * top_pct)))
            picked = g.sort_values(ascending=False).head(k).index
            base = np.sqrt(bw.reindex(picked).fillna(0.0) + 1e-6)
            if base.sum() <= 0:
                base = pd.Series(1.0, index=picked)
            w.loc[picked] = base / base.sum() * tgt

        # 单票上限：min(基准权重 + max_active, max_abs)
        cap = (bw + max_active).clip(upper=max_abs)
        for _ in range(20):
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap[over]).sum()
            w[over] = cap[over]
            room = (cap - w)[(w > 0) & (~over)]
            if room.sum() <= 1e-12:
                break
            w[room.index] += room / room.sum() * excess

        s = w.sum()
        if s > 0:
            rows[d] = w / s
    return pd.DataFrame(rows).T.reindex(columns=score.columns).fillna(0.0)


# ============================================================================
# 回测引擎
# ============================================================================

def backtest(w_target: pd.DataFrame,
             ret: pd.DataFrame,
             bench_ret: pd.Series,
             cost_buy: float = 0.0007,
             cost_sell: float = 0.0012,
             exec_lag: int = 1) -> pd.DataFrame:
    """
    w_target : 信号日(index) x code 的目标权重
    exec_lag : 信号日后第几个交易日收盘建仓（默认 1）；收益从建仓日的下一天起算
    """
    tdays = ret.index
    exec_map = {}
    for sig_d in w_target.index:
        pos = tdays.searchsorted(sig_d)
        ep = pos + exec_lag
        if ep < len(tdays):
            exec_map[tdays[ep]] = sig_d

    cur = pd.Series(0.0, index=ret.columns)
    recs = []
    for d in tdays:
        r = ret.loc[d].fillna(0.0)

        # 先结算今天的收益（用昨天收盘后的持仓）。
        # cur 之和可以小于 1 —— 差额是现金，收益按 0 计
        pr = float((cur * r).sum())
        # 权重漂移：新权重 = 个股增长 / 组合整体增长。
        #
        # ⚠ 这里**不能归一化到和为 1**。原实现用 grown/grown.sum()，
        # 那会把「半仓」在次日强行拉回满仓，
        # 使择时给出的仓位缩放当天就失效（实测 exposure 参数完全不起作用）。
        if cur.abs().sum() > 0:
            cur = cur * (1.0 + r) / (1.0 + pr) if (1.0 + pr) != 0 else cur

        cost = 0.0
        buy = sell = 0.0
        if d in exec_map:
            tgt = w_target.loc[exec_map[d]].reindex(ret.columns).fillna(0.0)
            diff = tgt - cur
            buy = diff.clip(lower=0).sum()
            sell = (-diff.clip(upper=0)).sum()
            cost = buy * cost_buy + sell * cost_sell
            cur = tgt

        recs.append({
            "date": d,
            "gross": pr,
            "cost": cost,
            "net": pr - cost,
            "bench": float(bench_ret.get(d, 0.0)),
            "turnover": buy + sell,
        })

    out = pd.DataFrame(recs).set_index("date")
    if exec_map:
        out = out.loc[min(exec_map.keys()):]      # 截掉建仓前的空仓期
    out["excess"] = out["net"] - out["bench"]
    out["nav"] = (1 + out["net"]).cumprod()
    out["bench_nav"] = (1 + out["bench"]).cumprod()
    out["excess_nav"] = out["nav"] / out["bench_nav"]
    return out


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min())


def perf_stats(bt: pd.DataFrame, ann: int = 244) -> pd.Series:
    n = len(bt)
    yrs = n / ann
    stats = {
        "策略年化": bt["nav"].iloc[-1] ** (1 / yrs) - 1,
        "基准年化": bt["bench_nav"].iloc[-1] ** (1 / yrs) - 1,
        "年化超额": bt["excess_nav"].iloc[-1] ** (1 / yrs) - 1,
        "策略波动": bt["net"].std() * np.sqrt(ann),
        "跟踪误差": bt["excess"].std() * np.sqrt(ann),
        "信息比IR": bt["excess"].mean() / bt["excess"].std() * np.sqrt(ann),
        "夏普(rf=0)": bt["net"].mean() / bt["net"].std() * np.sqrt(ann),
        "策略最大回撤": max_drawdown(bt["nav"]),
        "超额最大回撤": max_drawdown(bt["excess_nav"]),
        "月度胜率": (bt["excess"].resample("ME").sum() > 0).mean(),
        "年化单边换手": bt["turnover"].sum() / 2 / yrs,
        "年化成本拖累": bt["cost"].sum() / yrs,
    }
    return pd.Series(stats).round(4)


def yearly_table(bt: pd.DataFrame) -> pd.DataFrame:
    g = bt.groupby(bt.index.year)
    df = pd.DataFrame({
        "策略": g["net"].apply(lambda s: (1 + s).prod() - 1),
        "基准": g["bench"].apply(lambda s: (1 + s).prod() - 1),
    })
    df["超额"] = (1 + df["策略"]) / (1 + df["基准"]) - 1
    return df.round(4)


# ============================================================================
# 主流程
# ============================================================================

def month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).last().values)


def run_strategy(b: DataBundle,
                 top_pct: float = 0.30,
                 min_listed: int = 250,
                 groups: Dict[str, List[str]] = None,
                 exposure: pd.Series = None,
                 verbose: bool = True):
    """
    :param groups: 因子分组。默认已剔除 SUE（实测 t=-0.64 无效）
    :param exposure: 择时层给出的**总仓位比例**（index 与调仓日对齐）。
        指数增强满仓持有、beta≈1，不躲系统性下跌 —— 实测最大回撤 -33.5%。
        传入 exposure 后目标权重整体缩放，未投部分视为现金（不计收益）。
    """
    dates = month_end_dates(b.ret.index)
    dates = dates[dates >= b.ret.index[max(260, 0)]]  # 留足因子窗口

    universe = (b.is_member.reindex(dates).fillna(False)
                & b.tradable.reindex(dates).fillna(False)
                & (b.listed_days.reindex(dates).fillna(0) >= min_listed))

    raw = build_raw_factors(b, dates)
    proc = {k: prep_factor(v, dates, b.mktcap, b.industry, universe)
            for k, v in raw.items()}

    # 下期收益（用于 IC 检验）
    fwd = pd.DataFrame(index=dates, columns=b.ret.columns, dtype=float)
    cum = (1 + b.ret.fillna(0.0)).cumprod()
    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i + 1]
        fwd.loc[d0] = cum.loc[d1] / cum.loc[d0] - 1.0
    fwd = fwd.where(universe)

    ic = ic_report(proc, fwd)
    score = combine_factors(proc, groups)
    score = score.where(universe)
    w = build_weights(score, b, universe, top_pct=top_pct)
    if exposure is not None:
        # 按择时信号缩放总仓位。行业内部的相对权重不变，
        # 只改变「投多少钱进去」—— 择时与选股职责分离
        scale = exposure.reindex(w.index).ffill().fillna(1.0).clip(0.0, 1.0)
        w = w.mul(scale, axis=0)
    bt = backtest(w, b.ret, b.bench_ret)

    if verbose:
        print("\n===== 单因子 Rank IC =====")
        print(ic.to_string())
        print("\n===== 合成因子 IC =====")
        print(ic_report({"COMBO": score}, fwd).to_string())
        print("\n===== 组合绩效 =====")
        print(perf_stats(bt).to_string())
        print("\n===== 分年度 =====")
        print(yearly_table(bt).to_string())
    return {"ic": ic, "score": score, "weights": w, "bt": bt, "factors": proc}


# ============================================================================
# 合成数据自测（只为验证框架管道，不代表真实收益）
# ============================================================================

def make_demo(seed: int = 7, n: int = 300, start="2016-01-01", end="2023-12-31"):
    """
    合成数据自测：造一个「月度时变的真实信号 s」，让 s 同时驱动下期收益和 ROE。
    目的只是验证 IC 检验 / 中性化 / 组合构建 / 回测 这条管道是通的，
    收益数字本身没有任何现实含义。
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    codes = [f"S{i:04d}" for i in range(n)]
    T, N = len(days), n
    inds = np.array([f"IND{i % 10}" for i in range(N)])

    # 月度 AR(1) 信号，横截面均值 0 —— 不会累积成永久漂移，避免污染市值
    months = pd.Index(days).to_period("M")
    um = months.unique()
    S = np.zeros((len(um), N))
    S[0] = rng.normal(0, 1, N)
    for k in range(1, len(um)):
        S[k] = 0.85 * S[k - 1] + rng.normal(0, 0.53, N)
    S = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    m_idx = {m: k for k, m in enumerate(um)}
    sig_daily = np.array([S[m_idx[m]] for m in months])          # T x N

    mkt = rng.normal(0.0002, 0.011, T)
    beta = rng.uniform(0.7, 1.3, N)
    eps = rng.normal(0, 0.018, (T, N))
    r = mkt[:, None] * beta[None, :] + eps + 0.0010 * sig_daily  # 信号 -> 收益

    ret = pd.DataFrame(r, index=days, columns=codes)
    close = 10 * (1 + ret).cumprod()
    shares = pd.Series(rng.uniform(2e7, 8e8, N), index=codes)
    mc = close.mul(shares, axis=1)

    sig = pd.DataFrame(sig_daily, index=days, columns=codes)
    eq = mc * pd.Series(rng.uniform(0.3, 1.2, N), index=codes)   # BP 与信号无关
    # ROE 是信号的含噪读数；净利润 = ROE x 净资产
    roe = 0.10 + 0.03 * sig + pd.DataFrame(rng.normal(0, 0.02, (T, N)), days, codes)
    npttm = eq * roe
    ta = eq * 2.2
    ocf = npttm * (1.0 + 0.15 * sig)                              # 应计也带一点信号
    to = pd.DataFrame(np.abs(rng.normal(0.02, 0.008, (T, N))), days, codes)

    bw = mc.div(mc.sum(axis=1), axis=0)
    return DataBundle(
        close=close, ret=ret, mktcap=mc,
        is_member=pd.DataFrame(True, index=days, columns=codes),
        bench_weight=bw,
        industry=pd.DataFrame(np.tile(inds, (T, 1)), index=days, columns=codes),
        tradable=pd.DataFrame(True, index=days, columns=codes),
        listed_days=pd.DataFrame(np.tile(np.arange(T)[:, None] + 400, (1, N)),
                                 index=days, columns=codes),
        bench_ret=(ret * bw.shift(1).fillna(bw.iloc[0])).sum(axis=1),
        np_ttm=npttm, equity=eq, total_assets=ta, ocf_ttm=ocf, turnover=to,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="用合成数据跑通框架")
    a = p.parse_args()
    if a.demo:
        print("跑合成数据（仅验证管道，收益无意义）...")
        run_strategy(make_demo())
    else:
        print("请填好 DataBundle 后调用 run_strategy(bundle)")
