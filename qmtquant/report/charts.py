"""回测图表构建。

每个函数返回一个 plotly Figure，由 `html_report` 组装成单页报告。
拆开是为了能单独测试每张图的数据变换逻辑，而不必渲染整个报告。

配色约定：盈利红、亏损绿（A 股习惯，与美股相反）。
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

#: A 股配色：涨红跌绿
COLOR_UP = "#d62728"
COLOR_DOWN = "#2ca02c"
COLOR_LINE = "#1f77b4"
COLOR_BENCH = "#7f7f7f"
COLOR_GRID = "#e8e8e8"

_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Microsoft YaHei, SimHei, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=40),
    hovermode="x unified",
)


def equity_curve(equity: pd.Series, benchmark: pd.Series | None = None,
                 title: str = "净值曲线") -> go.Figure:
    """净值曲线，可叠加基准。

    两条线都归一化到 1 起步 —— 否则策略资金 100 万、指数 4000 点，
    画在一起完全看不出相对表现。
    """
    fig = go.Figure()
    if equity.empty:
        return _finish(fig, title)

    norm = equity / equity.iloc[0]
    fig.add_trace(go.Scatter(
        x=norm.index, y=norm.values, name="策略",
        line=dict(color=COLOR_LINE, width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>净值 %{y:.4f}<extra></extra>",
    ))

    if benchmark is not None and not benchmark.empty:
        # 对齐到策略的交易日，避免基准有而策略没有的日期造成断裂
        bench = benchmark.reindex(equity.index).ffill().dropna()
        if not bench.empty:
            bench = bench / bench.iloc[0]
            fig.add_trace(go.Scatter(
                x=bench.index, y=bench.values, name="基准",
                line=dict(color=COLOR_BENCH, width=1.5, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>基准 %{y:.4f}<extra></extra>",
            ))

    fig.add_hline(y=1.0, line=dict(color="#bbb", width=1))
    fig.update_yaxes(title_text="归一化净值")
    return _finish(fig, title)


def drawdown_curve(equity: pd.Series, title: str = "回撤曲线") -> go.Figure:
    """水下图，并标出最大回撤区间。

    标注区间用的是「峰值日 → 谷底日」，不是「谷底日」单点 ——
    回撤持续多久往往比回撤多深更影响持有体验。
    """
    fig = go.Figure()
    if equity.empty:
        return _finish(fig, title)

    running_max = equity.cummax()
    dd = (equity / running_max - 1) * 100

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name="回撤", fill="tozeroy",
        line=dict(color=COLOR_DOWN, width=1),
        fillcolor="rgba(44,160,44,0.25)",
        hovertemplate="%{x|%Y-%m-%d}<br>回撤 %{y:.2f}%<extra></extra>",
    ))

    trough = dd.idxmin()
    # 谷底之前净值最高的那天就是峰值起点
    peak = equity.loc[:trough].idxmax()
    fig.add_vrect(x0=peak, x1=trough, fillcolor="rgba(214,39,40,0.10)",
                  line_width=0,
                  annotation_text=f"最大回撤 {dd.min():.2f}%",
                  annotation_position="bottom left")

    fig.update_yaxes(title_text="回撤 (%)", ticksuffix="")
    return _finish(fig, title)


def monthly_heatmap(equity: pd.Series, title: str = "月度收益 (%)") -> go.Figure:
    """月度收益热力图：行=年，列=月。

    用来快速看出「是否某些月份稳定亏钱」这类日历效应，
    以及收益是否集中在少数几个月（集中度高说明策略脆弱）。
    """
    fig = go.Figure()
    if equity.empty:
        return _finish(fig, title)

    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        return _finish(fig, title)

    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))

    # 色标以 0 为中心，正红负绿
    limit = float(np.nanmax(np.abs(pivot.values))) if pivot.notna().any().any() else 1.0
    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=[f"{m}月" for m in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorscale=[[0, COLOR_DOWN], [0.5, "#ffffff"], [1, COLOR_UP]],
        zmid=0, zmin=-limit, zmax=limit,
        text=np.where(pd.isna(pivot.values), "",
                      np.vectorize(lambda v: f"{v:.1f}")(
                          np.nan_to_num(pivot.values))),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="%{y}年%{x}<br>收益 %{z:.2f}%<extra></extra>",
        colorbar=dict(title="%"),
    ))
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, title, hovermode="closest")


def price_with_trades(bars: pd.DataFrame, trades: pd.DataFrame,
                      vt_symbol: str) -> go.Figure:
    """价格走势 + 买卖点标注（单标的）。

    :param bars: DatetimeIndex + close 列
    :param trades: 含 datetime / direction / price / volume 列
    """
    fig = go.Figure()
    if bars.empty:
        return _finish(fig, f"{vt_symbol} 交易点")

    fig.add_trace(go.Scatter(
        x=bars.index, y=bars["close"], name="收盘价",
        line=dict(color="#888", width=1),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ))

    if not trades.empty:
        for direction, color, symbol_shape, name in [
            ("买入", COLOR_UP, "triangle-up", "买入"),
            ("卖出", COLOR_DOWN, "triangle-down", "卖出"),
        ]:
            sub = trades[trades["direction"] == direction]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sub["datetime"]), y=sub["price"],
                mode="markers", name=name,
                marker=dict(color=color, size=10, symbol=symbol_shape,
                            line=dict(width=1, color="white")),
                customdata=sub["volume"],
                hovertemplate=(f"{name}<br>%{{x|%Y-%m-%d}}<br>"
                               "价 %{y:.2f}<br>量 %{customdata:,.0f}<extra></extra>"),
            ))

    fig.update_yaxes(title_text="价格")
    return _finish(fig, f"{vt_symbol} 交易点")


def position_count(equity_index: pd.Index, trades: pd.DataFrame,
                   title: str = "持仓只数") -> go.Figure:
    """持仓标的数量随时间变化（多标的组合用）。

    单看净值曲线看不出策略是否真的在分散持仓 ——
    有可能名义上选 10 只，实际大部分时间只持有 2、3 只。
    """
    fig = go.Figure()
    if trades.empty:
        return _finish(fig, title)

    t = trades.copy()
    t["datetime"] = pd.to_datetime(t["datetime"])
    t["delta"] = np.where(t["direction"] == "买入", 1, -1)

    # 按标的累计持仓量，>0 即视为持有
    holding = {}
    series = []
    for row in t.sort_values("datetime").itertuples():
        vol = holding.get(row.symbol, 0) + row.delta * row.volume
        holding[row.symbol] = max(vol, 0)
        series.append((row.datetime, sum(1 for v in holding.values() if v > 0)))

    s = pd.Series(dict(series)).sort_index()
    s = s.reindex(equity_index, method="ffill").fillna(0)

    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, name="持仓只数",
        line=dict(color=COLOR_LINE, width=1.5, shape="hv"), fill="tozeroy",
        fillcolor="rgba(31,119,180,0.15)",
        hovertemplate="%{x|%Y-%m-%d}<br>持仓 %{y:.0f} 只<extra></extra>",
    ))
    fig.update_yaxes(title_text="只数", rangemode="tozero")
    return _finish(fig, title)


def symbol_pnl(trades: pd.DataFrame, top_n: int = 15,
               title: str = "分标的盈亏") -> go.Figure:
    """各标的的已实现盈亏排行（FIFO 配对）。

    用来看收益是否来自少数几只 —— 若前 2 名贡献了全部利润，
    说明策略实际是在赌个股而非系统性获利。
    """
    fig = go.Figure()
    if trades.empty:
        return _finish(fig, title)

    pnl = _realized_pnl_by_symbol(trades)
    if not pnl:
        return _finish(fig, title)

    s = pd.Series(pnl).sort_values()
    # 取两端各 top_n/2，中间的贡献小无需展示
    if len(s) > top_n:
        s = pd.concat([s.head(top_n // 2), s.tail(top_n - top_n // 2)])

    fig.add_trace(go.Bar(
        x=s.values, y=s.index, orientation="h",
        marker_color=[COLOR_UP if v > 0 else COLOR_DOWN for v in s.values],
        hovertemplate="%{y}<br>盈亏 %{x:,.0f} 元<extra></extra>",
    ))
    fig.update_xaxes(title_text="已实现盈亏 (元)")
    return _finish(fig, title, hovermode="closest",
                   height=max(320, 22 * len(s) + 100))


def _realized_pnl_by_symbol(trades: pd.DataFrame) -> dict[str, float]:
    """FIFO 配对计算各标的已实现盈亏（含手续费）"""
    result: dict[str, float] = {}
    queues: dict[str, list] = {}

    t = trades.copy()
    t["datetime"] = pd.to_datetime(t["datetime"])
    for row in t.sort_values("datetime").itertuples():
        sym = row.symbol
        fee_per = (row.commission / row.volume) if row.volume else 0
        q = queues.setdefault(sym, [])

        if row.direction == "买入":
            q.append([row.volume, row.price, fee_per])
            continue

        remaining = row.volume
        while remaining > 0 and q:
            lot = q[0]
            matched = min(lot[0], remaining)
            gain = ((row.price - lot[1]) * matched
                    - lot[2] * matched - fee_per * matched)
            result[sym] = result.get(sym, 0.0) + gain
            lot[0] -= matched
            remaining -= matched
            if lot[0] <= 0:
                q.pop(0)
    return result


def _finish(fig: go.Figure, title: str, hovermode: str = "x unified",
            height: int = 380) -> go.Figure:
    layout = dict(_LAYOUT)
    layout["hovermode"] = hovermode
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=15)),
                      height=height, **{k: v for k, v in layout.items()
                                        if k != "hovermode"},
                      hovermode=hovermode)
    fig.update_xaxes(gridcolor=COLOR_GRID)
    fig.update_yaxes(gridcolor=COLOR_GRID)
    return fig
