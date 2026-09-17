"""全市场日内 GBM 策略回测 —— 向量化实现。

## 为什么不用事件驱动引擎

全市场 3400 只标的 × 57000 根 1m bar = 1.96 亿次截面推送。事件驱动逐 bar
回放要跑几十小时。这里用向量化：一次性算出全市场所有时点的模型概率，
再按策略规则做矩阵运算，几分钟出结果。

代价是省略了逐笔撮合细节（部分成交、排队）。日内策略以分钟收盘价成交、
按固定滑点估算冲击成本，对策略优劣的相对排序足够。

## 三种模式

- **t_plus_0**：做 T。持有底仓，模型看多加仓、看空减仓，日内回转。
- **mean_reversion**：均值回归。VWAP 偏离超阈值 + 模型同向确认。
- **momentum**：打板追涨。高概率 + 量能放大 + 日内正收益。

## 风控（硬约束，策略无法绕过）

- 单票日内止损：亏损超 max_intraday_loss 立即平仓
- 组合回撤控制：从峰值回撤超阈值分档减仓/停止开仓
- 单票仓位上限、同时持仓数上限
- 尾盘强制平仓（不留隔夜，规避跳空风险）

## 输出

    models/intraday_gbm/backtest/
        {mode}_equity.csv       净值曲线
        {mode}_trades.csv       成交明细
        {mode}_daily.csv        每日收益
        summary.json            三种模式绩效对比 + 硬性指标达标情况

## 用法

    python scripts/backtest_intraday_gbm.py
    python scripts/backtest_intraday_gbm.py --mode momentum
    python scripts/backtest_intraday_gbm.py --capital 50000 --max-positions 5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qmtquant.core.costs import DEFAULT_COST  # noqa: E402
from qmtquant.engine.performance import TRADING_DAYS  # noqa: E402

MODEL_DIR = ROOT / "models" / "intraday_gbm"
OUT_DIR = MODEL_DIR / "backtest"

#: 成本走 qmtquant.core.costs 这一个事实源。此前本文件自己定义
#: COMMISSION/STAMP_TAX/SLIPPAGE，与另外 7 个文件各写各的，
#: 出现过佣金三个版本、印花税全部用 2023-08-28 减半前的旧值、
#: 过户费全体漏算，且**没有一处施加券商的 5 元最低佣金**。
#:
#: 原先的写法是 `金额 * (1 ± 费率)` —— 这个乘法式结构上就表达不了
#: 一个固定下限，所以最低佣金不是"忘了加"，是加不进去。
#: 改成 `金额 ± _fee(金额, 是否卖出)`。
COST = DEFAULT_COST
COMMISSION = COST.commission_rate
STAMP_TAX = COST.stamp_tax_rate
SLIPPAGE = COST.slippage_rate
MIN_COMMISSION = COST.commission_min


def _fee(amount: float, is_sell: bool) -> float:
    """单笔费用 + 滑点。含 5 元最低佣金 —— 单笔低于 58,548 元就触发，
    而做 T 的单笔几乎全在这条线以下。"""
    return COST.cost(amount, is_sell)

MODES = ["t_plus_0", "mean_reversion", "momentum"]


def load_model() -> dict:
    p = MODEL_DIR / "model.joblib"
    if not p.exists():
        raise FileNotFoundError(
            f"模型不存在：{p}\n先运行 python scripts/train_intraday_gbm.py")
    return joblib.load(p)


def collect_scored(clean_dir: Path, model_data: dict, max_symbols: int,
                   min_bars: int, start: str | None,
                   end: str | None) -> pd.DataFrame:
    """读取 1m 数据、算特征、模型打分，返回长表。

    返回列：datetime(index), symbol, close, prob_up, day_ret, vol_z,
             vwap_gap, day (日期)
    """
    from qmtquant.datafeed.adjust import adj_factor, real_price
    from qmtquant.features.intraday import (INTRADAY_FEATURES,
                                            compute_features_for_symbol)

    model = model_data["model"]
    features = model_data.get("features", INTRADAY_FEATURES)

    src = clean_dir / "1m"
    if not src.exists():
        src = clean_dir.parent / "1m"
    if not src.exists():
        raise FileNotFoundError(f"找不到 1m 数据：{src}")

    # 与训练一致的过滤规则：排除北交所、科创板、高股价
    ex_dirs = sorted(d for d in src.iterdir()
                     if d.is_dir() and d.name != "BSE")
    files = []
    for ex in ex_dirs:
        for f in sorted(ex.glob("*.parquet")):
            if f.stem.startswith("688"):
                continue
            files.append((f, f"{f.stem}.{ex.name}"))

    if max_symbols and len(files) > max_symbols:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(files), max_symbols, replace=False)
        files = [files[i] for i in sorted(idx)]

    print(f"  对 {len(files)} 只标的打分...")
    MAX_PRICE = 500.0
    keep = ["close", "day_ret", "vol_z_30", "vwap_gap"]
    chunks, batch = [], []
    factors: dict[str, float | None] = {}
    #: 因 ST 被排除的标的数。与实盘 forbid_st 对齐，
    #: 不为零是正常的；为零反而要查 ST 数据是不是没加载上。
    n_st = 0
    skipped = n_ok = 0

    from qmtquant.engine.backtest_engine import _load_st_checker
    is_st = _load_st_checker()
    if is_st is None:
        print("  [!] 加载不到 ST 历史，ST 标的不会被排除 —— "
              "回测会成交实盘风控拒绝的委托")
    t0 = time.time()

    for i, (path, vt) in enumerate(files, 1):
        feat = compute_features_for_symbol(path, vt)
        if feat is None or len(feat) < min_bars:
            skipped += 1
            continue
        # 与训练一致：按**真实价**判 1 手 > 5 万，不是后复权价。
        # 后复权因子 1.6~280 不等，按 close 过滤会误排便宜的老蓝筹。
        px = real_price(feat)
        if px is None:
            px = float(feat["close"].iloc[-1])
        if px > MAX_PRICE:
            skipped += 1
            continue

        # ST 标的不进候选池 —— 实盘风控 forbid_st 会拒绝买入它们
        # （见 RiskManager._do_check）。回测里放行等于让回测成交
        # 实盘拒单的委托，系统性高估机会集。1m 池里 74 只（1.8%）
        # 在回测期间是 ST；它们波动大、5% 涨跌停下更容易触及模型阈值，
        # 所以更容易被选中。
        #
        # 判据取「期间**任一时点**为 ST 就排除」，而不是只看最后一根 bar：
        # ST 状态会变，只看末值会把「前半段 ST、后半段摘帽」的标的
        # 整段放进来，回测就在它还是 ST 的那半段上成交了。
        #
        # 代价是保守：前半段 ST、后半段干净的标的，后半段本来可以交易。
        # 更精确的做法是逐 bar 过滤（在选股时按当日 ST 状态判），
        # 但那要把 (标的, 日期) 的 ST 表带进撮合循环。
        # 在 1.8% 的量级上，先取保守的一侧。
        if is_st is not None and len(feat):
            try:
                probe = pd.date_range(feat.index[0], feat.index[-1],
                                      freq="MS")
                if len(probe) == 0:
                    probe = [feat.index[0], feat.index[-1]]
                if any(is_st(vt, d) for d in probe):
                    n_st += 1
                    skipped += 1
                    continue
            except Exception:                       # noqa: BLE001
                pass

        if start:
            feat = feat[feat.index >= start]
        if end:
            feat = feat[feat.index <= end]
        if len(feat) < 60:
            skipped += 1
            continue

        # 模型打分
        x = feat[features].values.astype(np.float32)
        prob = model.predict_proba(x)[:, 1].astype(np.float32)

        sub = feat[keep].copy()
        for c in sub.columns:
            if sub[c].dtype == np.float64:
                sub[c] = sub[c].astype(np.float32)
        sub["prob_up"] = prob
        sub["symbol"] = vt
        # 复权因子：整手取整要作用在真实股数上（见 datafeed/adjust.py）。
        # 取不到就记 None，下游退回按后复权价取整并计数。
        factors[vt] = adj_factor(feat)
        batch.append(sub)
        n_ok += 1

        if len(batch) >= 200:
            chunks.append(pd.concat(batch, axis=0))
            batch.clear()
        if i % 200 == 0:
            print(f"    {i}/{len(files)} ({time.time()-t0:.0f}s)")

    if batch:
        chunks.append(pd.concat(batch, axis=0))
    if not chunks:
        raise ValueError("无可用数据")

    df = pd.concat(chunks, axis=0)
    del chunks
    df["day"] = df.index.normalize()
    df.sort_index(inplace=True)
    n_nf = sum(1 for v in factors.values() if v is None)
    print(f"  完成：{n_ok} 只标的，{len(df):,} 条打分，跳过 {skipped} 只，"
          f"耗时 {time.time()-t0:.0f}s")
    if n_st:
        print(f"  排除 ST 标的 {n_st} 只（与实盘 forbid_st 对齐）")
    if n_nf:
        print(f"  [!] {n_nf} 只反推不出复权因子，整手取整退回按后复权价 —— "
              f"这些标的可能被算成不足一手而跳过")
    return df, factors


def _time_of_day(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.hour * 60 + idx.minute, index=idx)


def backtest(df: pd.DataFrame, mode: str, capital: float,
             max_positions: int, prob_buy: float, prob_sell: float,
             max_intraday_loss: float, entry_min: int, exit_min: int,
             vol_z_threshold: float, vwap_deviation: float,
             max_drawdown_stop: float,
             factors: dict | None = None,
             t_plus_1: bool = True) -> dict:
    """按 mode 跑一遍回测。

    逐日推进：每天在时间窗内按分钟遍历，选股/开仓/止损/卖出。

    ## T+1（默认开启）

    A 股当日买入的股票当日不可卖。此前这个回测完全没有该约束：持仓字典
    每天重置、日终无条件强平，于是自由地做同日买卖回合 —— 有一条记录是
    10:33 买、10:37 卖。那种交易在 A 股根本不可能发生。

    夏普 31.97、日均 18.2 个回合就是这么来的：收益建立在一个不存在的
    交易机制上。实盘印证了这点 —— 每分钟发卖单，每分钟被券商以
    「可卖数量不足」拒掉，连止损都执行不了。

    开启后持仓跨日保留，只有入场日早于当日的仓位才可卖；净值也相应改为
    现金 + 持仓盯市，否则隔夜仓的盈亏不会反映在曲线上。

    关掉只适用于 T+0 品种（ETF、可转债）。
    """
    tod = _time_of_day(df.index)
    # 只保留交易时段
    in_window = (tod >= entry_min) & (tod <= exit_min)
    d = df[in_window.values]

    days = pd.DatetimeIndex(d["day"].unique()).sort_values()

    cash = capital
    equity_curve = []
    trades = []
    daily_rows = []

    peak_equity = capital
    trading_halted = False

    # T+1 下持仓必须跨日保留 —— 原先每天重置，等于假设所有仓位隔夜凭空消失
    positions: dict[str, dict] = {}   # symbol -> {vol, entry, entry_t, entry_day}

    def _can_sell(pos: dict, today) -> bool:
        """T+1：入场日必须早于当日。"""
        return (not t_plus_1) or pos["entry_day"] != today

    for day in days:
        dd = d[d["day"] == day]
        if dd.empty:
            continue

        day_start_equity = cash + sum(
            p["vol"] * p["last"] for p in positions.values())

        # 组合回撤控制：从峰值回撤超阈值则当日停止开新仓。
        # 必须用净值而非现金 —— T+1 下持仓过夜，现金天然偏低，
        # 拿它比会把「正常持仓」误判成回撤，直接停掉开仓
        cur_dd = ((day_start_equity - peak_equity) / peak_equity
                  if peak_equity > 0 else 0)
        allow_new = (cur_dd > -max_drawdown_stop) and not trading_halted

        # 该日的分钟时点
        times = dd.index.unique()
        times = times.sort_values()

        for t in times:
            snap = dd.loc[[t]] if t in dd.index else None
            if snap is None or snap.empty:
                continue
            snap = snap.set_index("symbol")
            t_min = t.hour * 60 + t.minute

            # ---------- 先处理持仓：止损 / 卖出信号 ----------
            for sym in list(positions.keys()):
                if sym not in snap.index:
                    continue
                row = snap.loc[sym]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                price = float(row["close"])
                pos = positions[sym]
                pos["last"] = price          # 盯市价，隔夜仓算净值要用
                pnl_pct = (price - pos["entry"]) / pos["entry"]

                # T+1：当日买入的一律不可卖，止损与信号都只能顺延到次日
                if not _can_sell(pos, day):
                    continue

                should_exit = False
                reason = ""

                # 风控：单票止损（硬约束，优先于任何信号）
                if pnl_pct < -max_intraday_loss:
                    should_exit, reason = True, "止损"
                # 尾盘平仓（T+1 下只平得掉昨仓）
                elif t_min >= exit_min:
                    should_exit, reason = True, "尾盘平仓"
                # 模型转空
                elif float(row["prob_up"]) < prob_sell:
                    should_exit, reason = True, "模型转空"
                elif mode == "mean_reversion":
                    if float(row["vwap_gap"]) > vwap_deviation:
                        should_exit, reason = True, "回归到位"

                if should_exit:
                    proceeds = price * pos["vol"] - _fee(price * pos["vol"], True)
                    cost = pos["entry"] * pos["vol"] + _fee(pos["entry"] * pos["vol"], False)
                    cash += proceeds
                    trades.append({
                        "day": str(pd.Timestamp(day).date()),
                        "symbol": sym, "side": "sell",
                        "entry_time": str(pos["entry_t"]),
                        "exit_time": str(t),
                        "entry_price": round(pos["entry"], 4),
                        "exit_price": round(price, 4),
                        "volume": int(pos["vol"]),
                        "pnl": round(proceeds - cost, 2),
                        "pnl_pct": round(pnl_pct, 5),
                        "reason": reason,
                    })
                    del positions[sym]

            # ---------- 再考虑开仓 ----------
            if (not allow_new) or t_min >= exit_min - 5:
                continue
            if len(positions) >= max_positions:
                continue

            cands = snap[~snap.index.isin(positions.keys())].copy()
            if cands.empty:
                continue

            if mode == "t_plus_0":
                # 做T：模型强看多即开仓（简化：无底仓约束，视为可回转）
                cands = cands[cands["prob_up"] > prob_buy]
            elif mode == "mean_reversion":
                cands = cands[(cands["vwap_gap"] < -vwap_deviation)
                              & (cands["prob_up"] > prob_buy)]
            elif mode == "momentum":
                cands = cands[(cands["prob_up"] > prob_buy)
                              & (cands["vol_z_30"] > vol_z_threshold)
                              & (cands["day_ret"] > 0)]

            if cands.empty:
                continue

            cands = cands.sort_values("prob_up", ascending=False)
            slots = max_positions - len(positions)
            per_size = capital / max_positions

            for sym, row in cands.head(slots).iterrows():
                price = float(row["close"])
                if price <= 0:
                    continue
                # A 股 100 股一手 —— 一手是 100 **真实**股。
                # price 是后复权价，直接按它取整，高因子的标的会被算成
                # 不足一手而静默跳过（实测 2 万/票时剔除 10.3%）。
                # 按真实股数取整，再换算回后复权口径，这样
                # price * vol 仍然是正确的金额。
                f = (factors or {}).get(sym) or 1.0
                real_vol = int(per_size / price * f / 100) * 100
                vol = real_vol / f
                if real_vol < 100:
                    continue
                cost = price * vol + _fee(price * vol, False)
                if cost > cash:
                    continue
                cash -= cost
                positions[sym] = {"vol": vol, "entry": price, "entry_t": t,
                                  "entry_day": day, "last": price}

        # 收盘：平掉**可卖**的残留持仓。T+1 下当日买入的平不掉，留隔夜
        if positions:
            last_snap = dd.loc[[times[-1]]].set_index("symbol")
            for sym, pos in list(positions.items()):
                if sym in last_snap.index:
                    r = last_snap.loc[sym]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    price = float(r["close"])
                    pos["last"] = price
                else:
                    price = pos["last"]
                if not _can_sell(pos, day):
                    continue
                proceeds = price * pos["vol"] - _fee(price * pos["vol"], True)
                cost = pos["entry"] * pos["vol"] + _fee(pos["entry"] * pos["vol"], False)
                cash += proceeds
                trades.append({
                    "day": str(pd.Timestamp(day).date()),
                    "symbol": sym, "side": "sell",
                    "entry_time": str(pos["entry_t"]),
                    "exit_time": str(times[-1]),
                    "entry_price": round(pos["entry"], 4),
                    "exit_price": round(price, 4),
                    "volume": int(pos["vol"]),
                    "pnl": round(proceeds - cost, 2),
                    "pnl_pct": round(
                        (price - pos["entry"]) / pos["entry"], 5),
                    "reason": "收盘平仓",
                })
                # 只删已平的 —— clear() 会把留隔夜的仓位一并抹掉，
                # 那等于假设隔夜仓凭空消失，盈亏全丢
                del positions[sym]

        # 净值 = 现金 + 持仓盯市。T+1 下隔夜仓是常态，只算现金会让
        # 「买入当天」凭空亏掉全部买入金额，次日卖出再凭空赚回来
        hold_value = sum(p["vol"] * p["last"] for p in positions.values())
        equity = cash + hold_value
        peak_equity = max(peak_equity, equity)
        daily_rows.append({
            "date": str(pd.Timestamp(day).date()),
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "hold_value": round(hold_value, 2),
            "n_holding": len(positions),
            "ret": round((equity - day_start_equity) / day_start_equity, 6)
            if day_start_equity > 0 else 0,
        })
        equity_curve.append(equity)

    # ---------- 绩效统计 ----------
    daily = pd.DataFrame(daily_rows)
    tdf = pd.DataFrame(trades)

    if daily.empty:
        return {"mode": mode, "error": "无交易日"}

    eq = daily["equity"].values
    rets = daily["ret"].values
    total_ret = (eq[-1] - capital) / capital
    n_days = len(eq)

    running_max = np.maximum.accumulate(eq)
    dd_series = (eq - running_max) / running_max
    max_dd = float(dd_series.min())

    ann_factor = TRADING_DAYS / n_days if n_days > 0 else 0
    ann_ret = (1 + total_ret) ** ann_factor - 1 if total_ret > -1 else -1
    vol = float(rets.std() * np.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0
    sharpe = float((ann_ret - 0.02) / vol) if vol > 1e-9 else 0

    # 月化收益（按实际交易日折算）
    monthly_ret = (1 + total_ret) ** (21 / n_days) - 1 if n_days > 0 else 0

    win_rate = float((tdf["pnl"] > 0).mean()) if not tdf.empty else 0
    avg_win = float(tdf.loc[tdf["pnl"] > 0, "pnl"].mean()) if (
        not tdf.empty and (tdf["pnl"] > 0).any()) else 0
    avg_loss = float(tdf.loc[tdf["pnl"] <= 0, "pnl"].mean()) if (
        not tdf.empty and (tdf["pnl"] <= 0).any()) else 0

    return {
        "mode": mode,
        "initial_capital": capital,
        "final_equity": round(float(eq[-1]), 2),
        "total_return": round(total_ret, 4),
        "annual_return": round(float(ann_ret), 4),
        "monthly_return": round(float(monthly_ret), 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 3),
        "volatility": round(vol, 4),
        "trading_days": n_days,
        "n_trades": len(tdf),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "date_range": [daily["date"].iloc[0], daily["date"].iloc[-1]],
        "_daily": daily,
        "_trades": tdf,
    }


def check_targets(r: dict) -> dict:
    """对照硬性指标：月收益≥30%、回撤≤10%、夏普>1。"""
    return {
        "monthly_return_30pct": {
            "target": ">= 30%",
            "actual": f"{r.get('monthly_return', 0) * 100:.2f}%",
            "pass": r.get("monthly_return", 0) >= 0.30,
        },
        "max_drawdown_10pct": {
            "target": "<= 10%",
            "actual": f"{abs(r.get('max_drawdown', 0)) * 100:.2f}%",
            "pass": abs(r.get("max_drawdown", 0)) <= 0.10,
        },
        "sharpe_above_1": {
            "target": "> 1.0",
            "actual": f"{r.get('sharpe', 0):.3f}",
            "pass": r.get("sharpe", 0) > 1.0,
        },
    }


RUNS_DIR = ROOT / "strategies" / "intraday_gbm" / "runs"

#: 进 manifest.metrics 的核心指标。回测脚本内部还算了波动率、交易日数等，
#: 但列表页排序和跨 run 对比只用得上这几个。
_CORE_METRICS = ("total_return", "annual_return", "monthly_return",
                 "max_drawdown", "sharpe", "volatility",
                 "n_trades", "win_rate", "trading_days")


def _save_run(summary: dict, out: Path, results: dict, args) -> str:
    """把本次回测存成独立 run，供管理台按历史对比。

    旧路径 models/intraday_gbm/backtest/ 是固定覆写的 —— 跑第二次就把第一次
    冲掉，既没法比较参数，也随时可能弄丢有价值的结果。这里额外落一份带
    参数快照的 run，旧路径保留为「最近一次」，不破坏现有页面。
    """
    if not results:
        return ""

    import shutil

    from webui.model_registry import (create_manifest, generate_run_id,
                                      save_manifest)

    run_id = generate_run_id("bt")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {"summary_json": "summary.json"}
    shutil.copy2(out / "summary.json", run_dir / "summary.json")
    for mode in results:
        for suffix, key in (("daily", "daily_csv"), ("trades", "trades_csv")):
            src = out / f"{mode}_{suffix}.csv"
            if src.exists():
                shutil.copy2(src, run_dir / src.name)
                artifacts[f"{mode}_{key}"] = src.name

    # 顶层指标取夏普最高的模式，让回测 run 能和模型 run 用同一组列展示、排序
    best_mode = max(results, key=lambda m: results[m].get("sharpe") or 0)
    best = results[best_mode]
    metrics = {k: best.get(k) for k in _CORE_METRICS}
    metrics["best_mode"] = best_mode
    metrics["modes"] = sorted(results)
    metrics["by_mode"] = {}
    for m, r in results.items():
        tg = r.get("targets") or {}
        row = {k: r.get(k) for k in _CORE_METRICS}
        row["targets_passed"] = sum(1 for v in tg.values() if v.get("pass"))
        row["targets_total"] = len(tg)
        metrics["by_mode"][m] = row

    manifest = create_manifest(
        run_id=run_id,
        kind="backtest",
        model_type="IntradayGBM",
        strategy_id="intraday_gbm",
        params={**summary["config"], "modes": sorted(results),
                "start": args.start or "", "end": args.end or ""},
        metrics=metrics,
        artifacts=artifacts,
        test_period=[args.start or "", args.end or ""],
        notes=f"回测 {'/'.join(sorted(results))}，标的上限 {args.max_symbols}",
    )
    save_manifest(run_dir, manifest)
    return run_id


def main() -> int:
    p = argparse.ArgumentParser(description="全市场日内 GBM 策略回测")
    p.add_argument("--mode", default="all",
                   help="t_plus_0 / mean_reversion / momentum / all")
    p.add_argument("--capital", type=float, default=50000.0,
                   help="初始资金（默认 5 万）")
    p.add_argument("--max-positions", type=int, default=5,
                   help="同时持仓标的数")
    p.add_argument("--prob-buy", type=float, default=0.60)
    p.add_argument("--prob-sell", type=float, default=0.40)
    p.add_argument("--max-intraday-loss", type=float, default=0.02,
                   help="单票日内止损线")
    p.add_argument("--max-drawdown-stop", type=float, default=0.10,
                   help="组合回撤达此值停止开新仓")
    p.add_argument("--entry-time", default="09:35")
    p.add_argument("--exit-time", default="14:50")
    p.add_argument("--vol-z", type=float, default=1.5)
    p.add_argument("--vwap-dev", type=float, default=0.005)
    p.add_argument("--max-symbols", type=int, default=800,
                   help="回测使用的标的数（默认 800，全市场太慢）")
    p.add_argument("--min-bars", type=int, default=1000)
    p.add_argument("--start", default=None, help="回测起始日 YYYY-MM-DD")
    p.add_argument("--end", default=None)
    p.add_argument("--output", default=str(OUT_DIR))
    args = p.parse_args()

    from qmtquant.config import get_config

    cfg = get_config()
    store = Path(cfg.data.store_dir)

    def _tomin(s):
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    modes = MODES if args.mode == "all" else [args.mode]

    print("=" * 66)
    print("  全市场日内 GBM 策略回测")
    print(f"  初始资金 {args.capital:,.0f}  最大持仓 {args.max_positions} 只")
    print(f"  买入阈值 {args.prob_buy}  卖出阈值 {args.prob_sell}")
    print(f"  单票止损 {args.max_intraday_loss:.1%}  "
          f"组合回撤停止 {args.max_drawdown_stop:.1%}")
    print("=" * 66)

    model_data = load_model()
    print(f"  模型: horizon={model_data.get('horizon')} "
          f"threshold={model_data.get('threshold')}")

    df, factors = collect_scored(
        store / "clean" if (store / "clean" / "1m").exists() else store,
        model_data, args.max_symbols, args.min_bars, args.start, args.end)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for mode in modes:
        print(f"\n--- 回测模式: {mode} ---")
        t0 = time.time()
        r = backtest(
            df, mode, args.capital, args.max_positions,
            args.prob_buy, args.prob_sell, args.max_intraday_loss,
            _tomin(args.entry_time), _tomin(args.exit_time),
            args.vol_z, args.vwap_dev, args.max_drawdown_stop, factors=factors)

        if "error" in r:
            print(f"  {r['error']}")
            continue

        daily = r.pop("_daily")
        tdf = r.pop("_trades")
        daily.to_csv(out / f"{mode}_daily.csv", index=False)
        if not tdf.empty:
            tdf.to_csv(out / f"{mode}_trades.csv", index=False)

        r["targets"] = check_targets(r)
        results[mode] = r

        print(f"  耗时 {time.time()-t0:.0f}s")
        print(f"  总收益 {r['total_return']:>8.2%}   "
              f"月化 {r['monthly_return']:>7.2%}   "
              f"年化 {r['annual_return']:>8.2%}")
        print(f"  最大回撤 {r['max_drawdown']:>6.2%}   "
              f"夏普 {r['sharpe']:>7.3f}   "
              f"波动 {r['volatility']:>7.2%}")
        print(f"  交易 {r['n_trades']} 笔   胜率 {r['win_rate']:.2%}   "
              f"交易日 {r['trading_days']} 天")
        for k, v in r["targets"].items():
            mark = "达标" if v["pass"] else "未达标"
            print(f"    [{mark}] {k}: {v['actual']} (目标 {v['target']})")

    summary = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "capital": args.capital,
            "max_positions": args.max_positions,
            "prob_buy": args.prob_buy,
            "prob_sell": args.prob_sell,
            "max_intraday_loss": args.max_intraday_loss,
            "max_drawdown_stop": args.max_drawdown_stop,
            "entry_time": args.entry_time,
            "exit_time": args.exit_time,
            "max_symbols": args.max_symbols,
            "commission": COMMISSION,
            "stamp_tax": STAMP_TAX,
            "slippage": SLIPPAGE,
        },
        "model": {
            "horizon": model_data.get("horizon"),
            "threshold": model_data.get("threshold"),
        },
        "results": results,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    run_id = _save_run(summary, out, results, args)

    print(f"\n{'=' * 66}")
    print(f"  回测结果已保存: {out}")
    if run_id:
        print(f"  回测 run: strategies/intraday_gbm/runs/{run_id}")
    if results:
        print(f"\n  {'模式':<16} {'总收益':>9} {'月化':>9} {'回撤':>9} "
              f"{'夏普':>8} {'交易数':>7}")
        print(f"  {'-'*16} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*7}")
        for m, r in results.items():
            print(f"  {m:<16} {r['total_return']:>8.2%} "
                  f"{r['monthly_return']:>8.2%} {r['max_drawdown']:>8.2%} "
                  f"{r['sharpe']:>8.3f} {r['n_trades']:>7}")
    print(f"\n  管理台查看: http://127.0.0.1:8800/intraday-gbm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
