"""单标的日内做 T 参数优化 —— 盛屯矿业 600711.SSE。

## A 股 T+0 的真实机制

A 股是 T+1 交收：**当日买入的股票当日不可卖**。做 T 靠的是「底仓」——
昨日之前就持有的股票，这部分随时可卖。两种做法：

- **正T**：低点用现金买入 N 股（锁定到明日）→ 高点从**底仓**卖出 N 股。
  净持股不变，现金净增 (卖价 - 买价) × N。
- **反T**：高点先从底仓卖出 N 股 → 低点用现金买回 N 股（锁定）。
  净持股不变，现金净增同上。

两者数学上等价，区别只是先买还是先卖。约束不同：
正T 先要有现金，反T 先要有可卖底仓。

本模拟严格区分 `base_shares`（可卖）与 `locked_shares`（今日买入、锁定），
收盘后 locked 转入 base。任何违反 T+1 的操作都不会发生。

## 收益拆解

- **T 收益**：做 T 操作本身赚的价差（策略的真实 alpha）
- **底仓收益**：持有底仓的市值涨跌（与做 T 无关，是 beta）
- **总收益**：两者之和，对应账户总资产变化

「年化收益越高越好」优化的是**总收益**，但两部分会分开报告 ——
把 beta 当成策略业绩是自欺欺人。

## 用法

    python scripts/optimize_t0_600711.py
    python scripts/optimize_t0_600711.py --symbol 600711 --exchange SSE
    python scripts/optimize_t0_600711.py --start 2025-09-08 --end 2025-12-31
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

OUT_DIR = ROOT / "models" / "t0_single"

# 交易成本
COMMISSION = 0.00025     # 佣金万 2.5（双边）
STAMP_TAX = 0.001        # 印花税千 1（仅卖出）
SLIPPAGE = 0.0005        # 滑点万 5

LOT = 100                # A 股一手 100 股


def load_bars(symbol: str, exchange: str, start: str | None,
              end: str | None) -> pd.DataFrame:
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


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """预计算日内指标：VWAP、日内位置、动量。"""
    d = df[["open", "high", "low", "close", "volume", "amount"]].copy()
    d["day"] = d.index.normalize()
    d["tmin"] = d.index.hour * 60 + d.index.minute

    g = d.groupby("day")

    # ---- VWAP：必须先做单位归一化 ----
    # xtdata 的 amount 与 close×volume 不同量纲，实测比值在不同标的间从
    # 7 到 47 不等（本例约 10.16）。不归一化的话 vwap_gap 恒为 -0.9 左右，
    # 任何 ±0.5% 量级的阈值都永不触发 —— 参数看似生效，实则全是死代码。
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

    # 自检：归一化后 gap 应以 0 为中心且量级在百分之几，否则说明单位仍不对
    med = float(d["vwap_gap"].median())
    if abs(med) > 0.05:
        raise ValueError(
            f"VWAP 归一化失败：vwap_gap 中位数 {med:.4f} 偏离 0 太远"
            f"（scale={scale:.3f}）。阈值判断会全部失效，拒绝继续。")

    d["day_open"] = g["open"].transform("first")
    d["day_ret"] = d["close"] / d["day_open"] - 1
    d["run_high"] = g["high"].cummax()
    d["run_low"] = g["low"].cummin()
    # 距日内已实现高/低点的相对位置
    rng = (d["run_high"] - d["run_low"]).replace(0, np.nan)
    d["pos_in_range"] = ((d["close"] - d["run_low"]) / rng).fillna(0.5)

    d["ret5"] = d.groupby("day")["close"].pct_change(5).fillna(0)
    return d


def simulate(d: pd.DataFrame, base_value: float, cash_value: float,
             buy_gap: float, sell_gap: float, max_trades: int,
             stop_loss: float, entry_min: int, exit_min: int,
             use_range_filter: bool, take_profit: float = 0.0,
             momentum: bool = False) -> dict:
    """跑一遍做 T 模拟。

    :param buy_gap:  低于 VWAP 多少比例时买入（正数，如 0.006 = 0.6%）
    :param sell_gap: 高于 VWAP 多少比例时卖出
    :param max_trades: 每日最多做 T 次数（一买一卖算一次）
    :param stop_loss: 单次做 T 的止损线
    :param use_range_filter: 是否叠加日内位置过滤（低位才买、高位才卖）
    """
    # 转 numpy：iterrows 在 5.8 万根 bar × 600 组参数下慢到不可用
    px_a = d["close"].values.astype(np.float64)
    gap_a = d["vwap_gap"].values.astype(np.float64)
    pos_a = d["pos_in_range"].values.astype(np.float64)
    tmin_a = d["tmin"].values.astype(np.int32)
    day_codes, day_uniq = pd.factorize(d["day"], sort=True)
    ts_a = d.index

    days = pd.DatetimeIndex(day_uniq)
    first_px = float(px_a[0])

    base_shares = int(base_value / first_px / LOT) * LOT
    if base_shares < LOT:
        return {"error": "底仓资金不足一手"}
    cash = cash_value
    locked_shares = 0

    # 每次做 T 的股数：用现金能买的量，分 max_trades 份
    t_shares = int(cash_value / first_px / max_trades / LOT) * LOT
    if t_shares < LOT:
        t_shares = LOT

    t_pnl_total = 0.0        # 纯做 T 收益（不含底仓涨跌）
    trades = []
    daily = []

    # 每日的 bar 索引区间（day_codes 已按日排序）
    n_bars = len(px_a)
    day_bounds = []
    s = 0
    for i in range(1, n_bars + 1):
        if i == n_bars or day_codes[i] != day_codes[s]:
            day_bounds.append((s, i))
            s = i

    for di, (lo, hi) in enumerate(day_bounds):
        if hi - lo < 30:
            continue
        day = days[di]

        day_start_cash = cash
        n_done = 0
        open_leg = None      # 未平的做 T 腿: {"side","price","shares","t"}
        day_trades = 0

        for k in range(lo, hi):
            tmin = int(tmin_a[k])
            px = px_a[k]
            if px <= 0:
                continue
            t = ts_a[k]

            # ---------- 尾盘强制平掉未完成的腿 ----------
            if tmin >= exit_min:
                if open_leg is not None:
                    if open_leg["side"] == "buy":
                        # 已买入，需卖出等量底仓平掉
                        sh = open_leg["shares"]
                        if base_shares >= sh:
                            proceeds = px * sh * (
                                1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                            cost = open_leg["price"] * sh * (
                                1 + COMMISSION + SLIPPAGE)
                            base_shares -= sh
                            locked_shares += sh
                            cash += proceeds
                            pnl = proceeds - cost
                            t_pnl_total += pnl
                            trades.append({
                                "day": str(day.date()), "type": "正T",
                                "buy_time": str(open_leg["t"]),
                                "sell_time": str(t),
                                "buy_px": round(open_leg["price"], 3),
                                "sell_px": round(px, 3),
                                "shares": sh, "pnl": round(pnl, 2),
                                "pnl_pct": round(
                                    px / open_leg["price"] - 1, 5),
                                "reason": "尾盘平仓"})
                            day_trades += 1
                    else:
                        # 已卖出底仓，需买回
                        sh = open_leg["shares"]
                        cost = px * sh * (1 + COMMISSION + SLIPPAGE)
                        if cash >= cost:
                            proceeds = open_leg["price"] * sh * (
                                1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                            cash -= cost
                            locked_shares += sh
                            pnl = proceeds - cost
                            t_pnl_total += pnl
                            trades.append({
                                "day": str(day.date()), "type": "反T",
                                "buy_time": str(t),
                                "sell_time": str(open_leg["t"]),
                                "buy_px": round(px, 3),
                                "sell_px": round(open_leg["price"], 3),
                                "shares": sh, "pnl": round(pnl, 2),
                                "pnl_pct": round(
                                    open_leg["price"] / px - 1, 5),
                                "reason": "尾盘买回"})
                            day_trades += 1
                    open_leg = None
                continue

            if tmin < entry_min:
                continue

            gap = gap_a[k]
            pos = pos_a[k]

            # ---------- 有未平腿：找平仓机会 ----------
            if open_leg is not None:
                if open_leg["side"] == "buy":
                    entry_px = open_leg["price"]
                    chg = px / entry_px - 1
                    # 止损 或 止盈 或 反向信号
                    hit_stop = chg < -stop_loss
                    hit_target = (take_profit > 0 and chg >= take_profit) or (
                        gap > sell_gap and (
                            not use_range_filter or pos > 0.6))
                    if hit_stop or hit_target:
                        sh = open_leg["shares"]
                        if base_shares >= sh:
                            proceeds = px * sh * (
                                1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                            cost = entry_px * sh * (1 + COMMISSION + SLIPPAGE)
                            base_shares -= sh
                            locked_shares += sh
                            cash += proceeds
                            pnl = proceeds - cost
                            t_pnl_total += pnl
                            trades.append({
                                "day": str(day.date()), "type": "正T",
                                "buy_time": str(open_leg["t"]),
                                "sell_time": str(t),
                                "buy_px": round(entry_px, 3),
                                "sell_px": round(px, 3),
                                "shares": sh, "pnl": round(pnl, 2),
                                "pnl_pct": round(chg, 5),
                                "reason": "止损" if hit_stop else "达标"})
                            open_leg = None
                            n_done += 1
                            day_trades += 1
                else:  # side == "sell"（反T，已卖出待买回）
                    entry_px = open_leg["price"]
                    chg = entry_px / px - 1     # 卖高买低为正
                    hit_stop = chg < -stop_loss
                    hit_target = (take_profit > 0 and chg >= take_profit) or (
                        gap < -buy_gap and (
                            not use_range_filter or pos < 0.4))
                    if hit_stop or hit_target:
                        sh = open_leg["shares"]
                        cost = px * sh * (1 + COMMISSION + SLIPPAGE)
                        if cash >= cost:
                            proceeds = entry_px * sh * (
                                1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                            cash -= cost
                            locked_shares += sh
                            pnl = proceeds - cost
                            t_pnl_total += pnl
                            trades.append({
                                "day": str(day.date()), "type": "反T",
                                "buy_time": str(t),
                                "sell_time": str(open_leg["t"]),
                                "buy_px": round(px, 3),
                                "sell_px": round(entry_px, 3),
                                "shares": sh, "pnl": round(pnl, 2),
                                "pnl_pct": round(chg, 5),
                                "reason": "止损" if hit_stop else "达标"})
                            open_leg = None
                            n_done += 1
                            day_trades += 1
                continue

            # ---------- 无持腿：找开仓机会 ----------
            if n_done >= max_trades:
                continue
            # 留足时间平仓
            if tmin > exit_min - 15:
                continue

            sh = t_shares
            # momentum=False（均值回归）：跌破 VWAP 买、涨过 VWAP 卖
            # momentum=True （动量）    ：涨过 VWAP 买、跌破 VWAP 卖
            if momentum:
                buy_sig = gap > buy_gap and (
                    not use_range_filter or pos > 0.6)
                sell_sig = gap < -sell_gap and (
                    not use_range_filter or pos < 0.4)
            else:
                buy_sig = gap < -buy_gap and (
                    not use_range_filter or pos < 0.4)
                sell_sig = gap > sell_gap and (
                    not use_range_filter or pos > 0.6)

            # 正T：先买入（用现金），后从底仓卖出
            if buy_sig:
                cost = px * sh * (1 + COMMISSION + SLIPPAGE)
                if cash >= cost:
                    cash -= cost
                    open_leg = {"side": "buy", "price": px,
                                "shares": sh, "t": t}
            # 反T：先卖底仓，后买回
            elif sell_sig:
                if base_shares >= sh:
                    proceeds = px * sh * (
                        1 - COMMISSION - STAMP_TAX - SLIPPAGE)
                    base_shares -= sh
                    cash += proceeds
                    open_leg = {"side": "sell", "price": px,
                                "shares": sh, "t": t}

        # 收盘：锁定股转为可卖底仓
        base_shares += locked_shares
        locked_shares = 0

        close_px = float(px_a[hi - 1])
        equity = cash + base_shares * close_px
        daily.append({
            "date": str(day.date()),
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "shares": base_shares,
            "close": round(close_px, 3),
            "t_trades": day_trades,
            "day_t_pnl": round(cash - day_start_cash, 2),
        })

    if not daily:
        return {"error": "无有效交易日"}

    dl = pd.DataFrame(daily)
    tdf = pd.DataFrame(trades)

    init_equity = base_value + cash_value
    eq = dl["equity"].values
    final_equity = float(eq[-1])
    total_ret = final_equity / init_equity - 1

    n_days = len(eq)
    rets = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
    running_max = np.maximum.accumulate(eq)
    max_dd = float(((eq - running_max) / running_max).min())

    ann_factor = 244 / n_days
    ann_ret = (1 + total_ret) ** ann_factor - 1 if total_ret > -1 else -1
    vol = float(rets.std() * np.sqrt(244)) if len(rets) > 1 else 0.0
    sharpe = float(ann_ret / vol) if vol > 1e-9 else 0.0

    # 收益拆解：底仓 beta vs 做 T alpha
    buy_hold_shares = int(base_value / first_px / LOT) * LOT
    last_px = float(dl["close"].iloc[-1])
    base_pnl = buy_hold_shares * (last_px - first_px)
    t_ret = t_pnl_total / init_equity
    base_ret = base_pnl / init_equity
    # 纯持有底仓（不做T）的对照组
    bh_equity = cash_value + buy_hold_shares * last_px
    bh_ret = bh_equity / init_equity - 1
    bh_ann = (1 + bh_ret) ** ann_factor - 1 if bh_ret > -1 else -1

    win_rate = float((tdf["pnl"] > 0).mean()) if not tdf.empty else 0.0

    return {
        "initial_equity": init_equity,
        "final_equity": round(final_equity, 2),
        "total_return": round(total_ret, 4),
        "annual_return": round(float(ann_ret), 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 3),
        "volatility": round(vol, 4),
        "trading_days": n_days,
        "n_trades": len(tdf),
        "trades_per_day": round(len(tdf) / n_days, 2),
        "win_rate": round(win_rate, 4),
        "t_pnl": round(t_pnl_total, 2),
        "t_return": round(t_ret, 4),
        "t_annual": round(float((1 + t_ret) ** ann_factor - 1), 4)
        if t_ret > -1 else -1,
        "base_pnl": round(base_pnl, 2),
        "base_return": round(base_ret, 4),
        "buyhold_return": round(bh_ret, 4),
        "buyhold_annual": round(float(bh_ann), 4),
        "excess_vs_buyhold": round(total_ret - bh_ret, 4),
        "date_range": [dl["date"].iloc[0], dl["date"].iloc[-1]],
        "_daily": dl,
        "_trades": tdf,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="单标的日内做 T 参数优化")
    p.add_argument("--symbol", default="600711")
    p.add_argument("--exchange", default="SSE")
    p.add_argument("--base", type=float, default=100000.0, help="底仓市值")
    p.add_argument("--cash", type=float, default=100000.0, help="可用现金")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--entry-time", default="09:35")
    p.add_argument("--exit-time", default="14:50")
    p.add_argument("--output", default=str(OUT_DIR))
    p.add_argument("--top", type=int, default=15, help="展示前 N 组")
    args = p.parse_args()

    def _tomin(s):
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    print("=" * 70)
    print(f"  单标的日内做 T 优化 —— {args.symbol}.{args.exchange}")
    print(f"  底仓 {args.base:,.0f} 元 + 现金 {args.cash:,.0f} 元 "
          f"= 总资产 {args.base + args.cash:,.0f} 元")
    print("=" * 70)

    df = load_bars(args.symbol, args.exchange, args.start, args.end)
    d = prep(df)
    days = pd.DatetimeIndex(d["day"].unique())
    print(f"  数据: {len(df):,} 根 1m bar，{len(days)} 个交易日")
    print(f"  区间: {d.index.min()} → {d.index.max()}")

    daily_amp = d.groupby("day").apply(
        lambda x: (x["high"].max() - x["low"].min()) / x["close"].iloc[-1],
        include_groups=False)
    print(f"  日均振幅: {daily_amp.mean():.2%}（做 T 的收益空间来源）")

    # 信号频率自检 —— 阈值必须能被实际触发，否则参数搜索毫无意义
    gapq = d["vwap_gap"].abs()
    print(f"  VWAP 单位归一化系数: {d.attrs.get('amount_scale', 1.0):.3f}")
    print(f"  |vwap_gap| 分位: 50%={gapq.quantile(.5):.4f} "
          f"75%={gapq.quantile(.75):.4f} 90%={gapq.quantile(.9):.4f} "
          f"99%={gapq.quantile(.99):.4f}")

    # ---------------- 参数网格 ----------------
    buy_gaps = [0.005, 0.008, 0.012, 0.018, 0.025]
    sell_gaps = [0.005, 0.010, 0.018]
    max_trades_list = [1, 2, 3, 5]
    stop_losses = [0.008, 0.015, 0.025]
    take_profits = [0.0, 0.003, 0.005, 0.008, 0.012]
    range_filters = [False, True]
    directions = [False, True]      # False=均值回归, True=动量

    combos = list(itertools.product(
        buy_gaps, sell_gaps, max_trades_list, stop_losses,
        take_profits, range_filters, directions))
    print(f"\n  参数组合: {len(combos)} 组"
          f"（含止盈 {len(take_profits)} 档 × 方向 {len(directions)} 种）")

    entry_m, exit_m = _tomin(args.entry_time), _tomin(args.exit_time)
    results = []
    t0 = time.time()

    for i, (bg, sg, mt, sl, tp, rf, mom) in enumerate(combos, 1):
        r = simulate(d, args.base, args.cash, bg, sg, mt, sl,
                     entry_m, exit_m, rf, take_profit=tp, momentum=mom)
        if "error" in r:
            continue
        r.pop("_daily"); r.pop("_trades")
        r.update({"buy_gap": bg, "sell_gap": sg, "max_trades": mt,
                  "stop_loss": sl, "take_profit": tp,
                  "range_filter": rf, "momentum": mom})
        results.append(r)
        if i % 500 == 0:
            print(f"    {i}/{len(combos)} ({time.time()-t0:.0f}s)", flush=True)

    if not results:
        print("所有参数组合都无有效结果")
        return 1

    rdf = pd.DataFrame(results)
    # 只保留有实际交易的组合
    traded = rdf[rdf["n_trades"] > 0].copy()
    if traded.empty:
        print("没有任何参数组合产生交易 —— 阈值可能过严")
        return 1

    traded.sort_values("annual_return", ascending=False, inplace=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    traded.to_csv(out / f"{args.symbol}_grid.csv", index=False)

    bh_ann = traded["buyhold_annual"].iloc[0]
    print(f"\n{'=' * 70}")
    print(f"  参数搜索完成（{len(traded)} 组有交易，耗时 {time.time()-t0:.0f}s）")
    print(f"  对照组 —— 纯持有底仓不做T: 年化 {bh_ann:.2%}")
    print(f"{'=' * 70}")
    print(f"  {'#':<3} {'方向':<5} {'买阈':<6} {'卖阈':<6} {'止盈':<6} "
          f"{'止损':<6} {'次数':<4} {'年化':>9} {'做T年化':>9} "
          f"{'回撤':>8} {'夏普':>7} {'交易':>6} {'胜率':>7}")
    print(f"  {'-'*3} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*4} "
          f"{'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*6} {'-'*7}")
    for i, (_, r) in enumerate(traded.head(args.top).iterrows(), 1):
        print(f"  {i:<3} {'动量' if r['momentum'] else '回归':<5} "
              f"{r['buy_gap']:<6.3f} {r['sell_gap']:<6.3f} "
              f"{r['take_profit']:<6.3f} {r['stop_loss']:<6.3f} "
              f"{int(r['max_trades']):<4} "
              f"{r['annual_return']:>8.2%} {r['t_annual']:>9.2%} "
              f"{r['max_drawdown']:>8.2%} {r['sharpe']:>7.2f} "
              f"{int(r['n_trades']):>6} {r['win_rate']:>7.2%}")

    # ---------------- 样本外验证（必做，不可跳过）----------------
    # 在 N 组参数里挑最优，等于做了 N 次检验。样本内最优很可能只是噪声
    # 的极值。不做样本外验证就报告「最优方案」是在误导。
    print(f"\n{'=' * 70}")
    print("  样本外验证 —— 前半段选参数，后半段检验")
    print(f"{'=' * 70}")

    all_days = pd.DatetimeIndex(d["day"].unique()).sort_values()
    mid = all_days[len(all_days) // 2]
    d_in = d[d["day"] < mid]
    d_out = d[d["day"] >= mid]
    print(f"  样本内 {all_days[0].date()} ~ {mid.date()} "
          f"({len(all_days)//2} 天)")
    print(f"  样本外 {mid.date()} ~ {all_days[-1].date()} "
          f"({len(all_days)-len(all_days)//2} 天)")

    in_res = []
    for bg, sg, mt, sl, tp, rf, mom in combos:
        r = simulate(d_in, args.base, args.cash, bg, sg, mt, sl,
                     entry_m, exit_m, rf, take_profit=tp, momentum=mom)
        if "error" in r or r["n_trades"] == 0:
            continue
        in_res.append((r["t_annual"], bg, sg, mt, sl, tp, rf, mom))
    in_res.sort(reverse=True)

    wf_rows = []
    print(f"\n  {'#':<3} {'样本内做T':>10} {'样本外做T':>10}  参数")
    print(f"  {'-'*3} {'-'*10} {'-'*10}  {'-'*40}")
    for i, (tin, bg, sg, mt, sl, tp, rf, mom) in enumerate(in_res[:10], 1):
        ro = simulate(d_out, args.base, args.cash, bg, sg, mt, sl,
                      entry_m, exit_m, rf, take_profit=tp, momentum=mom)
        tout = ro.get("t_annual", float("nan"))
        wf_rows.append({"rank": i, "in_sample_t_annual": tin,
                        "out_sample_t_annual": tout,
                        "buy_gap": bg, "sell_gap": sg, "max_trades": mt,
                        "stop_loss": sl, "take_profit": tp,
                        "range_filter": rf, "momentum": mom})
        print(f"  {i:<3} {tin:>9.2%} {tout:>10.2%}  "
              f"{'动量' if mom else '回归'} 买{bg:.3f} 卖{sg:.3f} "
              f"止盈{tp:.3f} 止损{sl:.3f} ×{mt}")

    wf = pd.DataFrame(wf_rows)
    wf.to_csv(out / f"{args.symbol}_walkforward.csv", index=False)
    n_survive = int((wf["out_sample_t_annual"] > 0).sum())
    print(f"\n  样本内前 10 组中，样本外仍为正收益的: "
          f"{n_survive}/10")
    if n_survive == 0:
        print("  → 无一存活。样本内的正收益是噪声，策略不具备样本外预测能力。")

    # ---------------- 用最优参数重跑，保存明细 ----------------
    best = traded.iloc[0]
    print(f"\n{'=' * 70}")
    print("  最优方案")
    print(f"{'=' * 70}")
    br = simulate(d, args.base, args.cash, best["buy_gap"], best["sell_gap"],
                  int(best["max_trades"]), best["stop_loss"],
                  entry_m, exit_m, bool(best["range_filter"]),
                  take_profit=float(best["take_profit"]),
                  momentum=bool(best["momentum"]))
    bdl = br.pop("_daily"); btr = br.pop("_trades")
    bdl.to_csv(out / f"{args.symbol}_best_daily.csv", index=False)
    if not btr.empty:
        btr.to_csv(out / f"{args.symbol}_best_trades.csv", index=False)

    mom = bool(best["momentum"])
    print(f"  策略方向   {'动量（追涨杀跌）' if mom else '均值回归（低吸高抛）'}")
    print(f"  开仓阈值   VWAP {'上方' if mom else '下方'} "
          f"{best['buy_gap']:.2%} 买入")
    print(f"  反向阈值   VWAP {'下方' if mom else '上方'} "
          f"{best['sell_gap']:.2%}")
    print(f"  止盈       {best['take_profit']:.2%}"
          f"{'（未启用，等反向信号）' if best['take_profit'] == 0 else ''}")
    print(f"  止损       {best['stop_loss']:.2%}")
    print(f"  每日次数   {int(best['max_trades'])} 次")
    print(f"  位置过滤   {'开启' if best['range_filter'] else '关闭'}")
    print(f"  交易时段   {args.entry_time} ~ {args.exit_time}")
    print()
    print(f"  总资产     {br['initial_equity']:,.0f} → "
          f"{br['final_equity']:,.0f} 元")
    print(f"  总收益     {br['total_return']:>8.2%}    "
          f"年化 {br['annual_return']:>8.2%}")
    print(f"  最大回撤   {br['max_drawdown']:>8.2%}    "
          f"夏普 {br['sharpe']:>8.2f}")
    print(f"  交易       {br['n_trades']} 笔（日均 {br['trades_per_day']}），"
          f"胜率 {br['win_rate']:.2%}")
    print()
    print("  收益拆解（关键）:")
    print(f"    做 T 净收益   {br['t_pnl']:>12,.0f} 元  "
          f"占总资产 {br['t_return']:>7.2%}  年化 {br['t_annual']:>8.2%}")
    print(f"    底仓涨跌      {br['base_pnl']:>12,.0f} 元  "
          f"占总资产 {br['base_return']:>7.2%}")
    print()
    print(f"  对照 —— 纯持有不做T: 总收益 {br['buyhold_return']:.2%}，"
          f"年化 {br['buyhold_annual']:.2%}")
    print(f"  做 T 相对持有的超额: {br['excess_vs_buyhold']:>+.2%}")

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": f"{args.symbol}.{args.exchange}",
        "base_value": args.base,
        "cash_value": args.cash,
        "date_range": br["date_range"],
        "trading_days": br["trading_days"],
        "daily_amplitude": round(float(daily_amp.mean()), 4),
        "best_params": {
            "direction": "momentum" if best["momentum"] else "mean_reversion",
            "buy_gap": float(best["buy_gap"]),
            "sell_gap": float(best["sell_gap"]),
            "take_profit": float(best["take_profit"]),
            "stop_loss": float(best["stop_loss"]),
            "max_trades": int(best["max_trades"]),
            "range_filter": bool(best["range_filter"]),
            "entry_time": args.entry_time,
            "exit_time": args.exit_time,
        },
        "performance": br,
        "cost_model": {"commission": COMMISSION, "stamp_tax": STAMP_TAX,
                       "slippage": SLIPPAGE,
                       "round_trip": 2 * COMMISSION + STAMP_TAX
                       + 2 * SLIPPAGE},
        "robustness": {
            "n_configs": len(traded),
            "n_positive_t": int((traded["t_annual"] > 0).sum()),
            "pct_positive_t": round(
                float((traded["t_annual"] > 0).mean()), 4),
            "t_annual_mean": round(float(traded["t_annual"].mean()), 4),
            "walkforward_top10_survivors": n_survive,
            "verdict": ("样本外无一存活，样本内正收益为噪声"
                        if n_survive == 0 else
                        f"样本内前10组中{n_survive}组样本外仍为正"),
        },
        "walkforward": wf_rows,
        "top_configs": traded.head(args.top).to_dict("records"),
    }
    (out / f"{args.symbol}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print(f"\n  结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
