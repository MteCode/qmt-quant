"""回测产物 -> 图表数据。

各实验的输出格式不一，这里统一成 plotly 图形对象供页面渲染。
读不到的文件一律返回 None，页面据此显示「尚未运行」而不是报错 ——
管理台要能在任何产物缺失的状态下打开。
"""
import json
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go

from .registry import ROOT, STRATEGY

BACKTEST = ROOT / STRATEGY / "backtest"
MODELS = ROOT / STRATEGY / "models"
SIGNALS = ROOT / STRATEGY / "signals"
STATE = ROOT / STRATEGY / "state"

#: 统一配色。红涨绿跌是 A 股习惯，与国际相反，这里按 A 股来
C_UP = "#e04b4b"
C_DOWN = "#2e9e5b"
C_LINE = "#2c7be5"
C_MUTED = "#8a94a6"
C_GRID = "rgba(140,150,170,.15)"

LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=48, r=24, t=48, b=40),
    height=340,
    font=dict(size=12),
    xaxis=dict(gridcolor=C_GRID),
    yaxis=dict(gridcolor=C_GRID),
    showlegend=False,
)


def _mtime_str(ts: float) -> str:
    """文件修改时间 -> 本地时间字符串。

    不用 ``pd.Timestamp(ts, unit="s")`` —— 它把 unix 时间戳当 UTC，
    与 ``datetime.fromtimestamp`` 的本地时间差一个时区（东八区差 8 小时）。
    两者混用会让页面上的时间偏移，更糟的是让新鲜度比对静默漏报。
    """
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _read_json(name: str):
    p = BACKTEST / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


# ------------------------------------------------------------------ 净值曲线

def equity_figure():
    """PPO 组合与 ALSTM 单独的净值对比。"""
    import pandas as pd

    series = []
    for fname, label, color in [
        ("ppo_equity.csv", "ALSTM + PPO", C_LINE),
        ("alstm_only_equity.csv", "ALSTM 单独（满仓）", C_MUTED),
    ]:
        p = BACKTEST / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except (OSError, pd.errors.ParserError):
            continue
        date_col = next((c for c in df.columns
                         if c.lower() in ("date", "datetime", "trade_date")), None)
        val_col = next((c for c in df.columns
                        if c.lower() in ("equity", "value", "total")), None)
        if date_col is None or val_col is None:
            continue
        s = pd.Series(df[val_col].values,
                      index=pd.to_datetime(df[date_col])).sort_index()
        series.append((label, s / s.iloc[0], color))

    if not series:
        return None

    fig = go.Figure()
    for label, s, color in series:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=label,
                                 line=dict(color=color, width=2)))
    fig.add_hline(y=1.0, line=dict(color=C_MUTED, width=1, dash="dot"))
    fig.update_layout(**{**LAYOUT, "showlegend": True,
                         "legend": dict(orientation="h", y=1.12, x=0),
                         "title": "净值曲线（归一化）"})
    return _fig_html(fig)


def drawdown_figure():
    import pandas as pd

    p = BACKTEST / "ppo_equity.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except (OSError, pd.errors.ParserError):
        return None
    date_col = next((c for c in df.columns
                     if c.lower() in ("date", "datetime", "trade_date")), None)
    val_col = next((c for c in df.columns
                    if c.lower() in ("equity", "value", "total")), None)
    if date_col is None or val_col is None:
        return None

    s = pd.Series(df[val_col].values,
                  index=pd.to_datetime(df[date_col])).sort_index()
    dd = (s / s.cummax() - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                             line=dict(color=C_DOWN, width=1),
                             fillcolor="rgba(46,158,91,.18)"))
    # 20% 是硬约束，画出来才看得见有没有越线
    fig.add_hline(y=-20, line=dict(color=C_UP, width=1.5, dash="dash"),
                  annotation_text="硬约束 -20%", annotation_position="bottom right")
    fig.update_layout(**{**LAYOUT, "title": "回撤",
                         "yaxis": dict(gridcolor=C_GRID, ticksuffix="%")})
    return _fig_html(fig)


# ------------------------------------------------------------------ 实验图表

def seed_figure():
    """多种子方差 —— 每个种子的 Sharpe 与回撤。"""
    d = _read_json("seed_experiment.json")
    if not d or not d.get("runs"):
        return None
    runs = sorted(d["runs"], key=lambda r: r["seed"])
    seeds = [f"种子 {r['seed']}" for r in runs]
    sharpe = [r["sharpe"] for r in runs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=seeds, y=sharpe,
        marker_color=[C_UP if v > 0 else C_DOWN for v in sharpe],
        text=[f"{v:+.3f}" for v in sharpe], textposition="outside"))
    fig.add_hline(y=0, line=dict(color=C_MUTED, width=1))
    med = sorted(sharpe)[len(sharpe) // 2]
    fig.add_hline(y=med, line=dict(color=C_LINE, width=1.5, dash="dash"),
                  annotation_text=f"中位数 {med:+.3f}")
    fig.update_layout(**{**LAYOUT,
                         "title": "各随机种子的 Sharpe —— 衡量训练随机性"})
    return _fig_html(fig)


def scaling_figure():
    """集成规模 vs 结果离散度。"""
    import numpy as np

    d = _read_json("ensemble_scaling.json")
    if not d or not d.get("runs"):
        return None

    by_k = {}
    for r in d["runs"]:
        by_k.setdefault(r["k"], []).append(r["sharpe"])
    ks = sorted(by_k)

    fig = go.Figure()
    for k in ks:
        v = by_k[k]
        fig.add_trace(go.Box(y=v, name=str(k), marker_color=C_LINE,
                             boxpoints="all", jitter=.5, pointpos=0,
                             marker=dict(size=5, opacity=.55)))
    fig.add_hline(y=0, line=dict(color=C_MUTED, width=1))
    fig.update_layout(**{**LAYOUT, "height": 360,
                         "title": "集成规模 vs Sharpe 分布（箱体不收窄 = 模型高度相关）",
                         "xaxis": dict(title="集成的模型数", gridcolor=C_GRID)})
    return _fig_html(fig)


def sweep_figure():
    """组合参数网格热力图。"""
    d = _read_json("sweep_portfolio.json")
    if not d or not d.get("runs"):
        return None
    runs = d["runs"]
    hold = sorted({r["holdings"] for r in runs})
    reb = sorted({r["rebalance"] for r in runs})
    grid = {(r["holdings"], r["rebalance"]): r for r in runs}

    z, text = [], []
    for h in hold:
        row_z, row_t = [], []
        for rb in reb:
            r = grid.get((h, rb))
            if r is None:
                row_z.append(None); row_t.append("")
            else:
                row_z.append(r["sharpe"])
                ok = "达标" if r["drawdown_ok"] else "超限"
                row_t.append(f"Sharpe {r['sharpe']:+.3f}<br>"
                             f"收益 {r['total_return']:+.1%}<br>"
                             f"回撤 {r['max_drawdown']:.1%} {ok}")
        z.append(row_z); text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{c} 日" for c in reb], y=[f"{h} 只" for h in hold],
        text=text, hoverinfo="text",
        colorscale=[[0, C_DOWN], [.5, "#f2f4f7"], [1, C_UP]], zmid=0,
        colorbar=dict(title="Sharpe", thickness=12)))
    fig.update_layout(**{**LAYOUT, "height": 320,
                         "title": "组合参数网格 —— 孤立亮格是过拟合信号",
                         "xaxis": dict(title="调仓周期"),
                         "yaxis": dict(title="持仓只数")})
    return _fig_html(fig)


# ------------------------------------------------------------------ 概况

def strategy_overview() -> dict:
    """策略当前状态：模型产物、信号、风控。"""
    import pandas as pd

    info = {"models": [], "signal": None, "risk": None, "metrics": []}

    for label, p in [
        ("ALSTM 权重", MODELS / "alstm_weights.pt"),
        ("ALSTM 分数", MODELS / "alstm_scores.parquet"),
        ("PPO 模型", MODELS / "ppo_model.zip"),
    ]:
        if p.exists():
            st = p.stat()
            info["models"].append({
                "label": label, "size_kb": st.st_size / 1024,
                "mtime": _mtime_str(st.st_mtime),
            })
        else:
            info["models"].append({"label": label, "size_kb": None,
                                   "mtime": "缺失"})

    ens_dir = MODELS / "ensemble"
    if ens_dir.exists():
        n = len(list(ens_dir.glob("scores_seed*.parquet")))
        if n:
            info["models"].append({"label": f"集成分数（{n} 个种子）",
                                   "size_kb": None, "mtime": "已保存"})

    sig = SIGNALS / "target_latest.csv"
    if sig.exists():
        try:
            df = pd.read_csv(sig)
            info["signal"] = {
                "count": len(df),
                "weight": float(df["weight"].sum()) if "weight" in df else None,
                "value": float(df["target_value"].sum())
                if "target_value" in df else None,
                "mtime": _mtime_str(sig.stat().st_mtime),
                "rows": df.head(10).to_dict("records"),
            }
        except (OSError, pd.errors.ParserError, KeyError):
            pass

    rs = STATE / "risk_state.json"
    if rs.exists():
        try:
            data = json.loads(rs.read_text(encoding="utf-8"))
            levels = {0: "正常", 1: "只平不开", 2: "强制减仓", 3: "全平停止"}
            info["risk"] = {
                "level": levels.get(data.get("level", 0), "?"),
                "drawdown": data.get("drawdown", 0),
                "peak": data.get("peak", 0),
                "observations": data.get("observations", 0),
                "peak_resets": data.get("peak_resets", 0),
                "updated_at": data.get("updated_at", ""),
            }
        except (json.JSONDecodeError, OSError):
            pass

    # 关键结论指标
    seed = _read_json("seed_experiment.json")
    if seed and seed.get("runs"):
        import numpy as np
        sh = np.array([r["sharpe"] for r in seed["runs"]])
        dd = np.array([r["max_drawdown"] for r in seed["runs"]])
        info["metrics"].append({
            "label": "单模型 Sharpe 中位数", "value": f"{np.median(sh):+.3f}",
            "sub": f"区间 {sh.min():+.3f} ~ {sh.max():+.3f}（{len(sh)} 个种子）",
            "bad": np.median(sh) <= 0.2,
        })
        ok = int((dd <= 0.20).sum())
        info["metrics"].append({
            "label": "回撤达标率", "value": f"{ok}/{len(dd)}",
            "sub": f"中位回撤 {np.median(dd):.2%}",
            "bad": ok < len(dd),
        })

    ens = _read_json("ensemble_result.json")
    if ens and ens.get("ensemble"):
        e = ens["ensemble"]
        info["metrics"].append({
            "label": "集成 Sharpe", "value": f"{e['sharpe']:+.3f}",
            "sub": f"回撤 {e['max_drawdown']:.2%}",
            "bad": e["sharpe"] <= 0.2,
        })

    sweep = _read_json("sweep_portfolio.json")
    if sweep and sweep.get("runs"):
        runs = sweep["runs"]
        pos = sum(1 for r in runs if r["sharpe"] > 0)
        info["metrics"].append({
            "label": "参数网格正收益比例", "value": f"{pos}/{len(runs)}",
            "sub": "孤立亮格通常意味着过拟合",
            "bad": pos < len(runs) * 0.3,
        })

    return info


#: 股票名称与行业。首次读取后缓存 —— 5890 行的 parquet 没必要每次请求都读
_meta_cache = None


def _stock_meta() -> dict:
    """vt_symbol -> {name, industry, area}。取不到就返回空字典。"""
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache

    import pandas as pd

    meta = {}
    ind = ROOT / "data" / "universe" / "industry.parquet"
    if ind.exists():
        try:
            df = pd.read_parquet(ind)
            for r in df.itertuples():
                vt = getattr(r, "vt_symbol", None)
                if vt:
                    meta[vt] = {
                        "name": getattr(r, "name", "") or "",
                        "industry": getattr(r, "industry", "") or "",
                        "area": getattr(r, "area", "") or "",
                    }
        except (OSError, ValueError):
            pass

    # industry.parquet 缺的标的用 universe_full 补名称
    full = ROOT / "data" / "universe" / "universe_full.parquet"
    if full.exists():
        try:
            df = pd.read_parquet(full)
            for r in df.itertuples():
                vt = getattr(r, "vt_symbol", None)
                if vt and vt not in meta:
                    meta[vt] = {"name": getattr(r, "name", "") or "",
                                "industry": "", "area": ""}
        except (OSError, ValueError):
            pass

    _meta_cache = meta
    return meta


EXECUTIONS = ROOT / STRATEGY / "executions"


def positions() -> dict | None:
    """账户实际持仓快照。由 snapshot_positions.py 生成。

    管理台不直接连 miniQMT —— 连接要 QMT 在线、会阻塞请求、
    失败时整个页面打不开。改为读快照文件。
    """
    import pandas as pd

    p = STATE / "positions.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    # 快照过期就明确标出来 —— 拿隔夜的持仓当实时数据看会误判
    try:
        age_h = (pd.Timestamp.now()
                 - pd.Timestamp(d["updated_at"])).total_seconds() / 3600
    except (KeyError, ValueError):
        age_h = None
    d["age_hours"] = age_h
    d["is_stale"] = age_h is not None and age_h > 12

    # 「与信号差异」是快照那一刻算好写死的，而页面上的信号是现读的。
    # 信号若在快照之后重新生成过，这份差异已经失效 —— 必须标出来，
    # 否则页面上下两块数据对不上，用户却看不出问题在哪
    sig = d.get("signal") or {}
    d["signal_changed"] = False
    d["signal_date"] = None
    latest = SIGNALS / "target_latest.csv"
    if latest.exists():
        cur_m = datetime.fromtimestamp(latest.stat().st_mtime)
        # 快照记的是本地时间字符串，容差 2 秒避免浮点与截断造成误报
        if sig.get("mtime"):
            try:
                d["signal_changed"] = (
                    cur_m - pd.Timestamp(sig["mtime"])).total_seconds() > 2
            except ValueError:
                pass
        elif d.get("has_target"):
            # 旧版快照没记基准，无从判断，按已变更处理
            d["signal_changed"] = True
        d["signal_date"] = cur_m.strftime("%Y-%m-%d %H:%M")

    # 快照比对的那份信号是哪天的 —— 取 signals/ 里 mtime 最接近的一份
    if sig.get("mtime"):
        try:
            d["signal_label"] = pd.Timestamp(sig["mtime"]).strftime(
                "%Y-%m-%d %H:%M")
        except ValueError:
            d["signal_label"] = sig["mtime"]
    else:
        d["signal_label"] = None
    return d


def list_exec_dates() -> list:
    """有执行记录的日期，倒序。"""
    if not EXECUTIONS.exists():
        return []
    return sorted((p.stem.replace("exec_", "")
                   for p in EXECUTIONS.glob("exec_*.csv")), reverse=True)


def executions(date: str = None) -> dict | None:
    """实盘执行记录 —— 当天下了什么单、成交与否、被拦的原因。"""
    import pandas as pd

    dates = list_exec_dates()
    if not dates:
        return None
    date = date if date in dates else dates[0]

    try:
        df = pd.read_csv(EXECUTIONS / f"exec_{date}.csv", encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError):
        return None
    if df.empty:
        return None

    meta = _stock_meta()
    rows = []
    for r in df.itertuples():
        vt = str(getattr(r, "vt_symbol", ""))
        name = str(getattr(r, "name", "") or "") or meta.get(vt, {}).get("name", "—")
        rows.append({
            "time": str(getattr(r, "time", "")),
            "vt_symbol": vt,
            "code": vt.split(".")[0] if vt else "",
            "name": name,
            "industry": meta.get(vt, {}).get("industry", ""),
            "direction": str(getattr(r, "direction", "")),
            "volume": int(getattr(r, "volume", 0) or 0),
            "price": float(getattr(r, "price", 0) or 0),
            "amount": float(getattr(r, "amount", 0) or 0),
            "result": str(getattr(r, "result", "")),
            "reason": str(getattr(r, "reason", "") or ""),
            "mode": str(getattr(r, "mode", "")),
        })

    live = [r for r in rows if r["mode"] == "实盘"]
    blocked = [r for r in rows if r["result"] == "风控拦截"]
    buys = [r for r in rows if r["direction"] == "买" and r["result"] != "风控拦截"]
    sells = [r for r in rows if r["direction"] == "卖" and r["result"] != "风控拦截"]

    # 拦截原因分布 —— 反复出现同一原因说明配置或信号有系统性问题
    reasons = {}
    for r in blocked:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    return {
        "date": date, "dates": dates, "rows": rows,
        "n_total": len(rows), "n_live": len(live),
        "n_buy": len(buys), "n_sell": len(sells),
        "n_blocked": len(blocked),
        "buy_amount": sum(r["amount"] for r in buys),
        "sell_amount": sum(r["amount"] for r in sells),
        "reasons": sorted(reasons.items(), key=lambda x: -x[1]),
        "has_live": bool(live),
    }


#: 回测成交记录。文件名 -> 展示用的策略名
TRADE_FILES = {
    "alstm_only_trades.csv": "ALSTM 单独（满仓）",
}


def backtest_picks(fname: str = None, limit_dates: int = 40) -> dict | None:
    """从回测成交记录还原每个调仓日选了哪些股。

    回测只留下成交流水，没有直接记录「这一期选了谁」。但调仓是集中发生的：
    同一天的买单就是该期新选入的标的，卖单是被换掉的。据此按日期分组即可
    还原每期的调仓动作。
    """
    import pandas as pd

    fname = fname or next(iter(TRADE_FILES))
    path = BACKTEST / fname
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None
    if df.empty or "datetime" not in df.columns:
        return None

    meta = _stock_meta()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.date

    # direction 列在不同版本里可能是中文或英文，统一判定
    def is_buy(v) -> bool:
        return str(v) in ("买入", "LONG", "Direction.LONG", "buy", "BUY")

    sessions = []
    for d, g in df.groupby("datetime", sort=False):
        buys, sells = [], []
        for r in g.itertuples():
            vt = str(r.symbol)
            item = {
                "vt_symbol": vt,
                "code": vt.split(".")[0],
                "name": meta.get(vt, {}).get("name", "—"),
                "industry": meta.get(vt, {}).get("industry", ""),
                "price": float(r.price),
                "volume": int(r.volume),
                "amount": float(r.amount),
            }
            (buys if is_buy(r.direction) else sells).append(item)
        buys.sort(key=lambda x: -x["amount"])
        sells.sort(key=lambda x: -x["amount"])
        sessions.append({
            "date": str(d),
            "buys": buys, "sells": sells,
            "n_buy": len(buys), "n_sell": len(sells),
            "buy_amount": sum(x["amount"] for x in buys),
            "sell_amount": sum(x["amount"] for x in sells),
            "commission": float(g["commission"].sum())
            if "commission" in g else 0.0,
        })

    sessions.sort(key=lambda s: s["date"], reverse=True)
    total_comm = float(df["commission"].sum()) if "commission" in df else 0.0

    return {
        "file": fname,
        "label": TRADE_FILES.get(fname, fname),
        "files": [{"f": k, "label": v} for k, v in TRADE_FILES.items()
                  if (BACKTEST / k).exists()],
        "sessions": sessions[:limit_dates],
        "n_sessions": len(sessions),
        "n_trades": len(df),
        "total_commission": total_comm,
        "period": f"{min(df['datetime'])} ~ {max(df['datetime'])}",
        "mtime": _mtime_str(path.stat().st_mtime),
    }


def picks_industry_figure(picks: dict):
    """回测期间被选中标的的行业分布 —— 看策略实际在买什么。"""
    if not picks:
        return None
    counter = {}
    for s in picks["sessions"]:
        for b in s["buys"]:
            key = b["industry"] or "未分类"
            counter[key] = counter.get(key, 0) + 1
    if not counter:
        return None

    items = sorted(counter.items(), key=lambda x: -x[1])[:12]
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]

    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                           marker_color=C_LINE, text=vals,
                           textposition="outside"))
    fig.update_layout(**{**LAYOUT, "height": max(240, 26 * len(items) + 90),
                         "title": "买入标的的行业分布",
                         "xaxis": dict(title="买入次数", gridcolor=C_GRID),
                         "margin": dict(l=100, r=34, t=48, b=40)})
    return _fig_html(fig)


def list_signal_dates() -> list:
    """有哪些历史信号文件，按日期倒序。"""
    if not SIGNALS.exists():
        return []
    dates = []
    for p in SIGNALS.glob("target_*.csv"):
        stem = p.stem.replace("target_", "")
        if stem != "latest":
            dates.append(stem)
    return sorted(dates, reverse=True)


def selection(date: str = None) -> dict | None:
    """某日选股明细，含名称、行业、与上一期的对比。"""
    import pandas as pd

    dates = list_signal_dates()
    if date is None:
        path = SIGNALS / "target_latest.csv"
        if not path.exists() and dates:
            path = SIGNALS / f"target_{dates[0]}.csv"
            date = dates[0]
    else:
        path = SIGNALS / f"target_{date}.csv"
    if not path.exists():
        return None

    try:
        # 信号文件带 BOM，用 utf-8-sig 读
        df = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError):
        return None
    if df.empty or "vt_symbol" not in df.columns:
        return None

    meta = _stock_meta()

    # 与上一期比对，标出新进 / 保留
    prev_set = set()
    cur_date = date or (dates[0] if dates else None)
    if cur_date and cur_date in dates:
        i = dates.index(cur_date)
        if i + 1 < len(dates):
            try:
                prev = pd.read_csv(SIGNALS / f"target_{dates[i + 1]}.csv",
                                   encoding="utf-8-sig")
                prev_set = set(prev["vt_symbol"].astype(str))
            except (OSError, pd.errors.ParserError, KeyError):
                pass

    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        vt = str(r.vt_symbol)
        m = meta.get(vt, {})
        rows.append({
            "rank": i,
            "vt_symbol": vt,
            "code": vt.split(".")[0],
            "exchange": "沪" if vt.endswith("SSE") else "深",
            "name": m.get("name", "—"),
            "industry": m.get("industry", ""),
            "score": float(getattr(r, "score", 0) or 0),
            "weight": float(getattr(r, "weight", 0) or 0),
            "target_value": float(getattr(r, "target_value", 0) or 0),
            "is_new": bool(prev_set) and vt not in prev_set,
        })

    # 行业分布
    by_ind = {}
    for r in rows:
        key = r["industry"] or "未分类"
        by_ind[key] = by_ind.get(key, 0) + 1
    industries = sorted(by_ind.items(), key=lambda x: -x[1])

    return {
        "date": cur_date or "latest",
        "dates": dates,
        "rows": rows,
        "count": len(rows),
        "total_weight": sum(r["weight"] for r in rows),
        "total_value": sum(r["target_value"] for r in rows),
        "new_count": sum(1 for r in rows if r["is_new"]),
        "has_prev": bool(prev_set),
        "industries": industries,
        "mtime": _mtime_str(path.stat().st_mtime),
    }


def industry_figure(sel: dict):
    """选股的行业分布。集中在少数行业意味着组合承担了行业风险。"""
    if not sel or not sel["industries"]:
        return None
    items = sel["industries"][:12]
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]

    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                           marker_color=C_LINE,
                           text=vals, textposition="outside"))
    fig.update_layout(**{**LAYOUT, "height": max(240, 26 * len(items) + 90),
                         "title": "行业分布",
                         "xaxis": dict(title="只数", gridcolor=C_GRID),
                         "margin": dict(l=90, r=30, t=48, b=40)})
    return _fig_html(fig)


def subperiod_table():
    """分段检验对比 —— 同一参数在不同时间段是否稳定。"""
    h1 = _read_json("sweep_portfolio_h1.json")
    h2 = _read_json("sweep_portfolio_h2.json")
    if not h1 or not h2:
        return None

    def index(d):
        return {(r["holdings"], r["rebalance"]): r for r in d["runs"]}

    i1, i2 = index(h1), index(h2)
    keys = sorted(set(i1) & set(i2))
    if not keys:
        return None

    rows = []
    for k in keys:
        rows.append({
            "config": f"{k[0]} 只 / {k[1]} 日",
            "h1": i1[k]["sharpe"],
            "h2": i2[k]["sharpe"],
            "flip": (i1[k]["sharpe"] > 0) != (i2[k]["sharpe"] > 0),
        })
    rows.sort(key=lambda r: -(r["h1"] + r["h2"]))
    return {
        "period1": " ~ ".join(h1.get("period", ["", ""])),
        "period2": " ~ ".join(h2.get("period", ["", ""])),
        "rows": rows,
        "flips": sum(1 for r in rows if r["flip"]),
    }
