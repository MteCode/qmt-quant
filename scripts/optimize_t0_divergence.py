"""量价背离日内做 T —— 单标的参数搜索 + 样本外验证。

## 量价背离的逻辑

价格和成交量的配合关系是判断趋势真伪的经典依据：

| 形态 | 含义 | 交易含义 |
|------|------|---------|
| 价跌量增 | 恐慌抛售、卖压释放 | 超跌反弹机会 → 买 |
| 价涨量缩 | 上涨乏力、追高意愿弱 | 涨势可疑 → 卖 |
| 价涨量增 | healthy 上涨 | 顺势持有 |
| 价跌量缩 | 缩量阴跌、无人接盘 | 观望 |

「背离」指前两种：价格方向与量能方向不一致。

## 本脚本实现的三类背离信号

1. **corr 背离**：滚动窗口内「价格变化」与「量变化」的相关系数。
   corr 显著为负 = 背离。
2. **OBV 背离**：OBV（能量潮）与价格的短期斜率符号相反。
3. **量能脉冲背离**：量能 z-score 突增，但价格反向走。

## 成本门槛（必须先看）

单次往返成本 = 佣金双边 0.05% + 印花税 0.10% + 滑点双边 0.10% = **0.25%**

年化 100% 的要求（20 万本金、每天 1 次、每次投 10 万）：
每笔需净赚 0.82%，即**毛 edge ≥ 1.07%**。作为参照，日均振幅约 4.8%，
意味着每次做 T 要吃掉日内振幅的 22%。这个门槛非常高。

## 用法

    python scripts/optimize_t0_divergence.py
    python scripts/optimize_t0_divergence.py --symbol 600711 --exchange SSE
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "models" / "t0_divergence"

COMMISSION = 0.00025
STAMP_TAX = 0.001
SLIPPAGE = 0.0005
ROUND_TRIP = 2 * COMMISSION + STAMP_TAX + 2 * SLIPPAGE
LOT = 100


def load_bars(symbol: str, exchange: str, start=None, end=None) -> pd.DataFrame:
    for base in (ROOT / "data" / "clean" / "1m", ROOT / "data" / "1m"):
        p = base / exchange / f"{symbol}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            break
    else:
        raise FileNotFoundError(f"找不到 {symbol}.{exchange} 的 1m 数据")
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= pd.Timestamp(end) + pd.Timedelta(days=1)]
    return df.sort_index()


def prep(df: pd.DataFrame, win: int = 20) -> pd.DataFrame:
    """计算量价背离指标。所有窗口只看过去，无前视。"""
    d = df[["open", "high", "low", "close", "volume", "amount"]].copy()
    d["day"] = d.index.normalize()
    d["tmin"] = d.index.hour * 60 + d.index.minute

    g = d.groupby("day")

    # ---- VWAP（单位归一化，见 optimize_t0_600711.py 的说明）----
    ratio = (d["amount"] / (d["close"] * d["volume"])).replace(
        [np.inf, -np.inf], np.nan)
    scale = float(ratio.median())
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    d.attrs["amount_scale"] = scale
    cum_amt = g["amount"].cumsum()
    cum_vol = g["volume"].cumsum().replace(0, np.nan)
    d["vwap"] = ((cum_amt / cum_vol) / scale).replace(
        [np.inf, -np.inf], np.nan)
    d["vwap"] = d.groupby("day")["vwap"].ffill().bfill()
    d["vwap_gap"] = d["close"] / d["vwap"] - 1

    # ---- 价格 / 量能变化率 ----
    d["pret"] = g["close"].pct_change().fillna(0)
    vol = d["volume"].replace(0, np.nan)
    d["vchg"] = g["volume"].pct_change().replace(
        [np.inf, -np.inf], np.nan).fillna(0)

    # ---- 1. 滚动相关系数背离 ----
    # 分组 rolling corr 很慢，用 numpy 手算（E[xy]-E[x]E[y]）/(σx σy)
    def _roll_corr(a, b, n):
        sa = a.rolling(n, min_periods=n // 2)
        sb = b.rolling(n, min_periods=n // 2)
        cov = (a * b).rolling(n, min_periods=n // 2).mean() \
            - sa.mean() * sb.mean()
        den = sa.std() * sb.std()
        return (cov / den.replace(0, np.nan))

    d["pv_corr"] = _roll_corr(d["pret"], d["vchg"], win).fillna(0)

    # ---- 2. OBV 背离 ----
    sign = np.sign(d["pret"])
    d["obv"] = (sign * d["volume"]).groupby(d["day"]).cumsum()
    # 短期斜率：当前值相对 win 根前的变化，各自标准化后比较符号
    obv_chg = d["obv"] - d.groupby("day")["obv"].shift(win)
    px_chg = d["close"] - d.groupby("day")["close"].shift(win)
    obv_std = d.groupby("day")["obv"].transform(
        lambda x: x.rolling(win * 3, min_periods=win).std())
    px_std = d.groupby("day")["close"].transform(
        lambda x: x.rolling(win * 3, min_periods=win).std())
    d["obv_z"] = (obv_chg / obv_std.replace(0, np.nan)).fillna(0)
    d["px_z"] = (px_chg / px_std.replace(0, np.nan)).fillna(0)
    # 背离度：符号相反且幅度都大时，值为负且绝对值大
    d["obv_div"] = d["obv_z"] * d["px_z"]

    # ---- 3. 量能脉冲 ----
    vmu = d.groupby("day")["volume"].transform(
        lambda x: x.rolling(win, min_periods=win // 2).mean())
    vsd = d.groupby("day")["volume"].transform(
        lambda x: x.rolling(win, min_periods=win // 2).std())
    d["vol_z"] = ((d["volume"] - vmu) / vsd.replace(0, np.nan)).fillna(0)
    # 窗口内累计价格变化（用于判断脉冲方向）
    d["ret_n"] = (d["close"] / d.groupby("day")["close"].shift(win)
                  - 1).fillna(0)

    # ---- 日内位置 ----
    d["run_high"] = g["high"].cummax()
    d["run_low"] = g["low"].cummin()
    rng = (d["run_high"] - d["run_low"]).replace(0, np.nan)
    d["pos"] = ((d["close"] - d["run_low"]) / rng).fillna(0.5)
    d["day_open"] = g["open"].transform("first")
    d["day_ret"] = d["close"] / d["day_open"] - 1

    med = float(d["vwap_gap"].median())
    if abs(med) > 0.05:
        raise ValueError(
            f"VWAP 归一化失败：中位数 {med:.4f}（scale={scale:.3f}）")
    return d


def simulate(d: pd.DataFrame, base_value: float, cash_value: float,
             signal: str, thr: float, vol_thr: float,
             take_profit: float, stop_loss: float, max_trades: int,
             entry_min: int, exit_min: int,
             hold_max: int, reverse: bool) -> dict:
    """量价背离做 T 模拟。

    :param signal: corr / obv / pulse
    :param thr:    背离阈值
    :param vol_thr: 量能 z-score 门槛
    :param hold_max: 单腿最长持有分钟数（超时强平，避免死等）
    :param reverse: 反转信号方向（背离做反向 vs 顺向）
    """
    px_a = d["close"].values.astype(np.float64)
    tmin_a = d["tmin"].values.astype(np.int32)
    pos_a = d["pos"].values.astype(np.float64)
    volz_a = d["vol_z"].values.astype(np.float64)
    retn_a = d["ret_n"].values.astype(np.float64)
    ts_a = d.index

    if signal == "corr":
        sig_a = d["pv_corr"].values.astype(np.float64)
    elif signal == "obv":
        sig_a = d["obv_div"].values.astype(np.float64)
    else:
        sig_a = d["ret_n"].values.astype(np.float64)

    day_codes, day_uniq = pd.factorize(d["day"], sort=True)
    days = pd.DatetimeIndex(day_uniq)
    first_px = float(px_a[0])

    base_shares = int(base_value / first_px / LOT) * LOT
    if base_shares < LOT:
        return {"error": "底仓资金不足一手"}
    cash = cash_value
    locked = 0
    # 单次做 T 的股数，受两边同时约束：
    #   - 现金：正T 要先买入，买不起就开不了仓
    #   - 底仓：正T 买入后要从**底仓**卖出等量平掉（今日买入的锁定，
    #           卖不了），底仓不够这条腿就永远平不掉
    #
    # 只按现金算的话，底仓 6 万 / 现金 14 万这种配置下 t_shares 会
    # 超过底仓股数，腿开出去平不回来，现金被抽干、净值算崩 ——
    # 表现为「底仓越小回撤越大」这种明显反直觉的结果。
    by_cash = int(cash_value / first_px / max(1, max_trades) / LOT) * LOT
    t_shares = max(LOT, min(by_cash, base_shares))

    t_pnl = 0.0
    trades, daily = [], []

    n = len(px_a)
    bounds, s = [], 0
    for i in range(1, n + 1):
        if i == n or day_codes[i] != day_codes[s]:
            bounds.append((s, i))
            s = i

    for di, (lo, hi) in enumerate(bounds):
        if hi - lo < 40:
            continue
        day = days[di]
        day_cash0 = cash
        n_done = 0
        leg = None
        dtr = 0

        for k in range(lo, hi):
            tm = int(tmin_a[k])
            px = px_a[k]
            if px <= 0:
                continue
            t = ts_a[k]

            # ---------- 尾盘强平 ----------
            if tm >= exit_min:
                if leg is not None:
                    sh = leg["shares"]
                    if leg["side"] == "buy" and base_shares >= sh:
                        pro = px * sh * (1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                        cst = leg["price"] * sh * (1 + COMMISSION + SLIPPAGE)
                        base_shares -= sh; locked += sh; cash += pro
                        t_pnl += pro - cst
                        trades.append({
                            "day": str(day.date()), "type": "正T",
                            "buy_time": str(leg["t"]), "sell_time": str(t),
                            "buy_px": round(leg["price"], 3),
                            "sell_px": round(px, 3), "shares": sh,
                            "pnl": round(pro - cst, 2),
                            "pnl_pct": round(px / leg["price"] - 1, 5),
                            "reason": "尾盘"})
                        dtr += 1
                    elif leg["side"] == "sell":
                        cst = px * sh * (1 + COMMISSION + SLIPPAGE)
                        if cash >= cst:
                            pro = leg["price"] * sh * (
                                1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                            cash -= cst; locked += sh
                            t_pnl += pro - cst
                            trades.append({
                                "day": str(day.date()), "type": "反T",
                                "buy_time": str(t),
                                "sell_time": str(leg["t"]),
                                "buy_px": round(px, 3),
                                "sell_px": round(leg["price"], 3),
                                "shares": sh, "pnl": round(pro - cst, 2),
                                "pnl_pct": round(leg["price"] / px - 1, 5),
                                "reason": "尾盘"})
                            dtr += 1
                    leg = None
                continue

            if tm < entry_min:
                continue

            sg = sig_a[k]
            vz = volz_a[k]
            rn = retn_a[k]
            pz = pos_a[k]

            # ---------- 平仓 ----------
            if leg is not None:
                ep = leg["price"]
                chg = (px / ep - 1) if leg["side"] == "buy" else (ep / px - 1)
                held = tm - leg["tmin"]
                hit = (chg >= take_profit or chg <= -stop_loss
                       or held >= hold_max)
                if hit:
                    sh = leg["shares"]
                    why = ("止盈" if chg >= take_profit else
                           "止损" if chg <= -stop_loss else "超时")
                    if leg["side"] == "buy" and base_shares >= sh:
                        pro = px * sh * (1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                        cst = ep * sh * (1 + COMMISSION + SLIPPAGE)
                        base_shares -= sh; locked += sh; cash += pro
                        t_pnl += pro - cst
                        trades.append({
                            "day": str(day.date()), "type": "正T",
                            "buy_time": str(leg["t"]), "sell_time": str(t),
                            "buy_px": round(ep, 3), "sell_px": round(px, 3),
                            "shares": sh, "pnl": round(pro - cst, 2),
                            "pnl_pct": round(chg, 5), "reason": why})
                        leg = None; n_done += 1; dtr += 1
                    elif leg["side"] == "sell":
                        cst = px * sh * (1 + COMMISSION + SLIPPAGE)
                        if cash >= cst:
                            pro = ep * sh * (
                                1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                            cash -= cst; locked += sh
                            t_pnl += pro - cst
                            trades.append({
                                "day": str(day.date()), "type": "反T",
                                "buy_time": str(t), "sell_time": str(leg["t"]),
                                "buy_px": round(px, 3),
                                "sell_px": round(ep, 3), "shares": sh,
                                "pnl": round(pro - cst, 2),
                                "pnl_pct": round(chg, 5), "reason": why})
                            leg = None; n_done += 1; dtr += 1
                continue

            # ---------- 开仓 ----------
            if n_done >= max_trades or tm > exit_min - hold_max - 5:
                continue

            buy_sig = sell_sig = False
            if signal == "corr":
                # pv_corr 显著为负 = 量价背离
                if sg < -thr and abs(vz) > vol_thr:
                    # 价跌量增 → 买；价涨量缩 → 卖
                    if rn < 0:
                        buy_sig = True
                    elif rn > 0:
                        sell_sig = True
            elif signal == "obv":
                # obv_div < 0 表示 OBV 与价格背离
                if sg < -thr:
                    if rn < 0:
                        buy_sig = True
                    elif rn > 0:
                        sell_sig = True
            else:  # pulse：量能突增 + 价格方向
                if vz > vol_thr:
                    if rn < -thr:
                        buy_sig = True      # 放量下跌 → 抄底
                    elif rn > thr:
                        sell_sig = True     # 放量上涨 → 冲高卖

            if reverse:
                buy_sig, sell_sig = sell_sig, buy_sig

            sh = t_shares
            if buy_sig:
                cst = px * sh * (1 + COMMISSION + SLIPPAGE)
                if cash >= cst:
                    cash -= cst
                    leg = {"side": "buy", "price": px, "shares": sh,
                           "t": t, "tmin": tm}
            elif sell_sig and base_shares >= sh:
                pro = px * sh * (1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                base_shares -= sh; cash += pro
                leg = {"side": "sell", "price": px, "shares": sh,
                       "t": t, "tmin": tm}

        base_shares += locked
        locked = 0
        cpx = float(px_a[hi - 1])
        daily.append({"date": str(day.date()),
                      "equity": round(cash + base_shares * cpx, 2),
                      "close": round(cpx, 3), "t_trades": dtr,
                      "day_t_pnl": round(cash - day_cash0, 2)})

    if not daily:
        return {"error": "无有效交易日"}

    dl = pd.DataFrame(daily)
    tdf = pd.DataFrame(trades)
    init = base_value + cash_value
    eq = dl["equity"].values
    total = eq[-1] / init - 1
    nd = len(eq)
    rets = np.diff(eq) / eq[:-1] if nd > 1 else np.array([0.0])
    rmax = np.maximum.accumulate(eq)
    mdd = float(((eq - rmax) / rmax).min())
    af = 244 / nd
    ann = (1 + total) ** af - 1 if total > -1 else -1
    vol_ = float(rets.std() * np.sqrt(244)) if nd > 1 else 0.0
    shp = float(ann / vol_) if vol_ > 1e-9 else 0.0

    bh_sh = int(base_value / first_px / LOT) * LOT
    lpx = float(dl["close"].iloc[-1])
    bh = (cash_value + bh_sh * lpx) / init - 1
    bh_ann = (1 + bh) ** af - 1 if bh > -1 else -1
    t_ret = t_pnl / init

    gross = ((tdf["sell_px"] - tdf["buy_px"]) / tdf["buy_px"]).mean() \
        if not tdf.empty else 0.0

    return {
        "total_return": round(total, 4),
        "annual_return": round(float(ann), 4),
        "max_drawdown": round(mdd, 4),
        "sharpe": round(shp, 3),
        "trading_days": nd,
        "n_trades": len(tdf),
        "trades_per_day": round(len(tdf) / nd, 2),
        "days_with_trade": int((dl["t_trades"] > 0).sum()),
        "coverage": round(float((dl["t_trades"] > 0).mean()), 4),
        "win_rate": round(float((tdf["pnl"] > 0).mean())
                          if not tdf.empty else 0.0, 4),
        "t_pnl": round(t_pnl, 2),
        "t_return": round(t_ret, 4),
        "t_annual": round(float((1 + t_ret) ** af - 1)
                          if t_ret > -1 else -1, 4),
        "gross_edge": round(float(gross), 6),
        "net_edge": round(float(gross - ROUND_TRIP), 6),
        "buyhold_annual": round(float(bh_ann), 4),
        "date_range": [dl["date"].iloc[0], dl["date"].iloc[-1]],
        "_daily": dl, "_trades": tdf,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="量价背离日内做 T 优化")
    p.add_argument("--symbol", default="600711")
    p.add_argument("--exchange", default="SSE")
    p.add_argument("--base", type=float, default=100000.0)
    p.add_argument("--cash", type=float, default=100000.0)
    p.add_argument("--entry-time", default="09:35")
    p.add_argument("--exit-time", default="14:50")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--output", default=str(OUT_DIR))
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    def _tm(s):
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    init = args.base + args.cash
    print("=" * 76)
    print(f"  量价背离日内做 T —— {args.symbol}.{args.exchange}")
    print(f"  底仓 {args.base:,.0f} + 现金 {args.cash:,.0f} "
          f"= 总资产 {init:,.0f} 元")
    print("=" * 76)
    print(f"  往返成本 {ROUND_TRIP:.3%}")
    need_net = 1.0 / 244 * init / args.cash
    print(f"  年化 100% 需要：每天净赚 {init/244:,.0f} 元 → "
          f"每笔净 {need_net:.2%} → 毛 edge {need_net + ROUND_TRIP:.2%}")
    print("=" * 76)

    df = load_bars(args.symbol, args.exchange)
    d = prep(df, args.window)
    days = pd.DatetimeIndex(d["day"].unique()).sort_values()
    print(f"  数据 {len(df):,} bar，{len(days)} 交易日，"
          f"{days[0].date()} ~ {days[-1].date()}")
    print(f"  VWAP 归一化系数 {d.attrs['amount_scale']:.3f}")
    print(f"  pv_corr 分位: 10%={d['pv_corr'].quantile(.1):.3f} "
          f"50%={d['pv_corr'].quantile(.5):.3f}")
    print(f"  vol_z   分位: 90%={d['vol_z'].quantile(.9):.2f} "
          f"99%={d['vol_z'].quantile(.99):.2f}")

    # ---------------- 参数网格 ----------------
    signals = ["corr", "obv", "pulse"]
    thrs = [0.1, 0.2, 0.3, 0.005, 0.01]      # corr/obv 用前三，pulse 用后二
    vol_thrs = [0.5, 1.0, 1.5, 2.0]
    tps = [0.004, 0.006, 0.010, 0.015]
    sls = [0.004, 0.008, 0.012]
    mts = [1, 2, 4]
    holds = [15, 30, 60]
    revs = [False, True]

    combos = []
    for sig in signals:
        valid_thr = ([0.1, 0.2, 0.3] if sig in ("corr", "obv")
                     else [0.003, 0.005, 0.01])
        for c in itertools.product(valid_thr, vol_thrs, tps, sls, mts,
                                   holds, revs):
            combos.append((sig,) + c)
    print(f"\n  参数组合 {len(combos)} 组")

    em, xm = _tm(args.entry_time), _tm(args.exit_time)
    res = []
    t0 = time.time()
    for i, (sig, thr, vt, tp, sl, mt, hd, rv) in enumerate(combos, 1):
        r = simulate(d, args.base, args.cash, sig, thr, vt, tp, sl, mt,
                     em, xm, hd, rv)
        if "error" in r or r["n_trades"] == 0:
            continue
        r.pop("_daily"); r.pop("_trades")
        r.update({"signal": sig, "thr": thr, "vol_thr": vt, "take_profit": tp,
                  "stop_loss": sl, "max_trades": mt, "hold_max": hd,
                  "reverse": rv})
        res.append(r)
        if i % 500 == 0:
            print(f"    {i}/{len(combos)} ({time.time()-t0:.0f}s)", flush=True)

    if not res:
        print("无有效结果")
        return 1

    rdf = pd.DataFrame(res).sort_values("t_annual", ascending=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(out / f"{args.symbol}_grid.csv", index=False)

    bh = rdf["buyhold_annual"].iloc[0]
    npos = int((rdf["t_annual"] > 0).sum())
    print(f"\n{'=' * 76}")
    print(f"  搜索完成：{len(rdf)} 组有交易，耗时 {time.time()-t0:.0f}s")
    print(f"  基准 —— 纯持有不做T 年化 {bh:.2%}")
    print(f"  做T为正的组合: {npos}/{len(rdf)} ({npos/len(rdf):.1%})")
    print(f"{'=' * 76}")
    print(f"  {'#':<3} {'信号':<6} {'阈值':<6} {'量阈':<5} {'止盈':<6} "
          f"{'止损':<6} {'次数':<4} {'持仓':<5} {'反转':<5} "
          f"{'做T年化':>9} {'毛edge':>8} {'净edge':>8} "
          f"{'交易':>6} {'覆盖':>6} {'胜率':>7}")
    print("  " + "-" * 118)
    for i, (_, r) in enumerate(rdf.head(args.top).iterrows(), 1):
        print(f"  {i:<3} {r['signal']:<6} {r['thr']:<6.3f} "
              f"{r['vol_thr']:<5.1f} {r['take_profit']:<6.3f} "
              f"{r['stop_loss']:<6.3f} {int(r['max_trades']):<4} "
              f"{int(r['hold_max']):<5} {'是' if r['reverse'] else '否':<5} "
              f"{r['t_annual']:>9.2%} {r['gross_edge']:>8.4f} "
              f"{r['net_edge']:>8.4f} {int(r['n_trades']):>6} "
              f"{r['coverage']:>6.1%} {r['win_rate']:>7.2%}")

    # ---------------- 样本外验证 ----------------
    print(f"\n{'=' * 76}")
    print("  样本外验证 —— 前半段选参数，后半段检验")
    print(f"{'=' * 76}")
    mid = days[len(days) // 2]
    d_in, d_out = d[d["day"] < mid], d[d["day"] >= mid]
    print(f"  样本内 {days[0].date()} ~ {mid.date()} ({len(days)//2} 天)")
    print(f"  样本外 {mid.date()} ~ {days[-1].date()} "
          f"({len(days)-len(days)//2} 天)")

    ins = []
    for sig, thr, vt, tp, sl, mt, hd, rv in combos:
        r = simulate(d_in, args.base, args.cash, sig, thr, vt, tp, sl, mt,
                     em, xm, hd, rv)
        if "error" in r or r["n_trades"] == 0:
            continue
        ins.append((r["t_annual"], sig, thr, vt, tp, sl, mt, hd, rv))
    ins.sort(reverse=True)

    wf = []
    print(f"\n  {'#':<3} {'样本内做T':>10} {'样本外做T':>10}  参数")
    print("  " + "-" * 70)
    for i, (tin, sig, thr, vt, tp, sl, mt, hd, rv) in enumerate(ins[:10], 1):
        ro = simulate(d_out, args.base, args.cash, sig, thr, vt, tp, sl, mt,
                      em, xm, hd, rv)
        tout = ro.get("t_annual", float("nan"))
        wf.append({"rank": i, "in_sample_t_annual": tin,
                   "out_sample_t_annual": tout, "signal": sig, "thr": thr,
                   "vol_thr": vt, "take_profit": tp, "stop_loss": sl,
                   "max_trades": mt, "hold_max": hd, "reverse": rv})
        print(f"  {i:<3} {tin:>9.2%} {tout:>10.2%}  {sig} thr={thr} "
              f"vz={vt} tp={tp} sl={sl} ×{mt} hold={hd} "
              f"{'反转' if rv else ''}")

    wdf = pd.DataFrame(wf)
    wdf.to_csv(out / f"{args.symbol}_walkforward.csv", index=False)
    surv = int((wdf["out_sample_t_annual"] > 0).sum())
    print(f"\n  样本内前 10 组中样本外仍为正: {surv}/10")

    # ---------------- 最优组合明细 ----------------
    b = rdf.iloc[0]
    br = simulate(d, args.base, args.cash, b["signal"], b["thr"],
                  b["vol_thr"], b["take_profit"], b["stop_loss"],
                  int(b["max_trades"]), em, xm, int(b["hold_max"]),
                  bool(b["reverse"]))
    bdl = br.pop("_daily"); btr = br.pop("_trades")
    bdl.to_csv(out / f"{args.symbol}_best_daily.csv", index=False)
    if not btr.empty:
        btr.to_csv(out / f"{args.symbol}_best_trades.csv", index=False)

    print(f"\n{'=' * 76}")
    print("  样本内最优组合")
    print(f"{'=' * 76}")
    print(f"  信号 {b['signal']}  阈值 {b['thr']}  量能门槛 {b['vol_thr']}")
    print(f"  止盈 {b['take_profit']:.2%}  止损 {b['stop_loss']:.2%}  "
          f"每日 {int(b['max_trades'])} 次  最长持有 {int(b['hold_max'])} 分钟")
    print()
    print(f"  年化 {br['annual_return']:.2%}（做T部分 {br['t_annual']:.2%}）")
    print(f"  回撤 {br['max_drawdown']:.2%}  夏普 {br['sharpe']:.2f}")
    print(f"  交易 {br['n_trades']} 笔，日均 {br['trades_per_day']}，"
          f"覆盖 {br['coverage']:.1%} 的交易日")
    print(f"  毛 edge {br['gross_edge']:.4%}  成本 {ROUND_TRIP:.3%}  "
          f"净 edge {br['net_edge']:+.4%}")
    print(f"  基准（纯持有）年化 {br['buyhold_annual']:.2%}")

    target_gross = need_net + ROUND_TRIP
    print(f"\n  距年化 100% 的差距：")
    print(f"    需要毛 edge {target_gross:.2%}，实际 {br['gross_edge']:.2%}，"
          f"差 {target_gross / max(br['gross_edge'], 1e-9):.1f} 倍")

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": f"{args.symbol}.{args.exchange}",
        "base_value": args.base, "cash_value": args.cash,
        "date_range": br["date_range"], "trading_days": br["trading_days"],
        "cost_model": {"commission": COMMISSION, "stamp_tax": STAMP_TAX,
                       "slippage": SLIPPAGE, "round_trip": ROUND_TRIP},
        "target_100pct": {"need_net_edge": round(need_net, 5),
                          "need_gross_edge": round(target_gross, 5),
                          "actual_gross_edge": br["gross_edge"],
                          "shortfall_multiple": round(
                              target_gross / max(br["gross_edge"], 1e-9), 2)},
        "best_params": {k: (bool(b[k]) if k == "reverse" else
                            (int(b[k]) if k in ("max_trades", "hold_max")
                             else (b[k] if k == "signal" else float(b[k]))))
                        for k in ("signal", "thr", "vol_thr", "take_profit",
                                  "stop_loss", "max_trades", "hold_max",
                                  "reverse")},
        "performance": br,
        "robustness": {"n_configs": len(rdf), "n_positive_t": npos,
                       "pct_positive_t": round(npos / len(rdf), 4),
                       "t_annual_mean": round(
                           float(rdf["t_annual"].mean()), 4),
                       "walkforward_survivors": surv},
        "walkforward": wf,
        "top_configs": rdf.head(args.top).to_dict("records"),
    }
    (out / f"{args.symbol}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\n  结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
