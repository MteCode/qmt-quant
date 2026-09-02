"""回测产物 -> 图表数据。

各实验的输出格式不一，这里统一成 plotly 图形对象供页面渲染。
读不到的文件一律返回 None，页面据此显示「尚未运行」而不是报错 ——
管理台要能在任何产物缺失的状态下打开。
"""
import json
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
                "mtime": pd.Timestamp(st.st_mtime, unit="s").strftime(
                    "%Y-%m-%d %H:%M"),
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
                "mtime": pd.Timestamp(sig.stat().st_mtime, unit="s").strftime(
                    "%Y-%m-%d %H:%M"),
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
