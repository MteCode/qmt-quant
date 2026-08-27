"""回测 HTML 报告生成。

产出**单文件自包含 HTML**：plotly.js 内联（约 4MB），
无需联网、无需本地服务，双击即可打开，也能直接发给别人。

为什么值得做：此前回测只输出 CSV，得用 Excel 逐列翻。
净值曲线的形状、回撤的持续时间、收益是否集中在少数标的 ——
这些看图 10 秒就明白，翻 CSV 半小时也未必看得出来。
"""
import html
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from . import charts

_CSS = """
* { box-sizing: border-box; }
body {
  font-family: "Microsoft YaHei", "SimHei", -apple-system, sans-serif;
  margin: 0; padding: 24px; background: #f5f6f8; color: #24292f;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: #656d76; font-size: 13px; margin-bottom: 20px; }
.card {
  background: #fff; border: 1px solid #d8dee4; border-radius: 8px;
  padding: 16px 20px; margin-bottom: 16px;
}
.card h2 { font-size: 15px; margin: 0 0 12px; color: #24292f; }
.kpis { display: flex; flex-wrap: wrap; gap: 12px; }
.kpi {
  flex: 1 1 150px; background: #fafbfc; border: 1px solid #eaeef2;
  border-radius: 6px; padding: 10px 14px;
}
.kpi .label { font-size: 12px; color: #656d76; }
.kpi .value { font-size: 19px; font-weight: 600; margin-top: 2px; }
.pos { color: #d62728; }
.neg { color: #2ca02c; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 6px 10px; border-bottom: 1px solid #eaeef2; text-align: right; }
th { background: #fafbfc; font-weight: 600; text-align: right; color: #656d76; }
th:first-child, td:first-child { text-align: left; }
tbody tr:hover { background: #fafbfc; }
.warn {
  background: #fff8e6; border: 1px solid #f0d58c; border-radius: 6px;
  padding: 12px 16px; font-size: 13px; line-height: 1.7; margin-bottom: 16px;
}
.warn .title { font-weight: 600; color: #9a6700; margin-bottom: 6px; }
.muted { color: #656d76; font-size: 12px; margin-top: 8px; line-height: 1.6; }
"""


def _fmt_pct(v) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _cls(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def _kpi(label: str, value: str, cls: str = "") -> str:
    return (f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
            f'<div class="value {cls}">{value}</div></div>')


def _fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False,
                               "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


class BacktestReport:
    """把回测结果渲染成单页 HTML"""

    def __init__(self, stats, equity_df: pd.DataFrame,
                 trades_df: pd.DataFrame | None = None,
                 bias_report=None, benchmark: pd.Series | None = None,
                 bars: pd.DataFrame | None = None,
                 undersized_orders: dict | None = None,
                 rejected_orders: list | None = None,
                 title: str = "回测报告", subtitle: str = "") -> None:
        """
        :param stats: PerformanceStats
        :param equity_df: 含 equity 列，DatetimeIndex
        :param trades_df: 成交明细（BacktestEngine.get_trades_df 的输出）
        :param bias_report: 标的池 BiasReport，会显著影响结论可信度，必须一并呈现
        :param benchmark: 基准净值序列
        :param bars: 单标的回测时传入价格序列，用于绘制买卖点
        :param undersized_orders: 因不足一手而未下出的委托统计，
            必须呈现 —— 它意味着部分标的被静默排除在候选之外
        :param rejected_orders: 拒单原因列表
        """
        self.stats = stats
        self.equity_df = equity_df
        self.trades_df = (trades_df if trades_df is not None
                          else pd.DataFrame())
        self.bias_report = bias_report
        self.benchmark = benchmark
        self.bars = bars
        self.undersized_orders = undersized_orders or {}
        self.rejected_orders = rejected_orders or []
        self.title = title
        self.subtitle = subtitle

    # ------------------------------------------------------------ 各区块

    def _section_kpi(self) -> str:
        s = self.stats
        items = [
            _kpi("总收益率", _fmt_pct(s.total_return), _cls(s.total_return)),
            _kpi("年化收益率", _fmt_pct(s.annual_return), _cls(s.annual_return)),
            _kpi("最大回撤", _fmt_pct(s.max_drawdown), _cls(s.max_drawdown)),
            _kpi("Sharpe", f"{s.sharpe_ratio:.3f}", _cls(s.sharpe_ratio)),
            _kpi("Calmar", f"{s.calmar_ratio:.3f}", _cls(s.calmar_ratio)),
            _kpi("年化波动率", _fmt_pct(s.volatility)),
            _kpi("成交笔数", f"{s.total_trades:,}"),
            _kpi("胜率", _fmt_pct(s.win_rate)),
        ]
        return f'<div class="card"><div class="kpis">{"".join(items)}</div></div>'

    def _section_detail(self) -> str:
        s = self.stats
        rows = [
            ("回测区间", f"{s.start_date} ~ {s.end_date}"),
            ("交易日数", f"{s.trading_days:,}"),
            ("初始资金", f"{s.initial_capital:,.2f}"),
            ("期末资金", f"{s.final_capital:,.2f}"),
            ("最长回撤天数", f"{s.max_drawdown_duration:,}"),
            ("盈亏比", f"{s.profit_factor:.3f}"),
            ("累计手续费", f"{s.total_commission:,.2f}"),
            ("换手率", f"{s.turnover_rate:.2f} 倍"),
        ]
        body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
        fee_pct = (s.total_commission / s.initial_capital
                   if s.initial_capital else 0)
        note = ""
        if fee_pct > 0.05:
            note = (f'<div class="muted">⚠ 手续费占初始资金 {fee_pct:.1%}，'
                    f'交易成本已显著侵蚀收益，考虑降低换手</div>')
        return (f'<div class="card"><h2>绩效明细</h2><table><tbody>{body}'
                f'</tbody></table>{note}</div>')

    def _section_bias(self) -> str:
        r = self.bias_report
        if r is None:
            return ""
        if r.is_clean:
            lines = "".join(f"<div>· {html.escape(n)}</div>" for n in r.notes)
            return (f'<div class="card"><h2>标的池</h2>'
                    f'<div>标的数量 {r.size}，无幸存者偏差与成分股前视</div>'
                    f'<div class="muted">{lines}</div></div>')

        notes = "".join(f"<div>· {html.escape(n)}</div>" for n in r.notes)
        return (
            '<div class="warn">'
            '<div class="title">⚠ 标的池存在偏差，以下结论被系统性高估</div>'
            f'<div>幸存者偏差：{"存在" if r.survivorship else "已消除"}　|　'
            f'成分股前视：{"存在" if r.membership_lookahead else "已消除"}　|　'
            f'上市日过滤：{"已启用" if r.listing_filtered else "未启用"}</div>'
            f'{notes}'
            '<div>回测收益不可直接外推到实盘。</div></div>'
        )

    def _section_diagnostics(self) -> str:
        """回测执行层面的异常，与策略优劣无关但会让结论失真"""
        blocks = []

        under = self.undersized_orders or {}
        if under:
            top = sorted(under.items(), key=lambda kv: -kv[1])[:8]
            rows = "".join(f"<div>· {html.escape(s)}：{n} 次</div>"
                           for s, n in top)
            more = (f"<div>… 共 {len(under)} 只标的受影响</div>"
                    if len(under) > 8 else "")
            blocks.append(
                '<div class="warn">'
                '<div class="title">⚠ 有标的因不足一手而完全买不进</div>'
                f'<div>这些标的被静默排除在实际持仓之外，等于标的池比你以为的小。'
                f'常见于后复权数据：价格被抬高后一手的名义成本可达真实值数倍。</div>'
                f'{rows}{more}'
                '<div>处理方式：提高本金、减少持仓只数，或按真实价取整。</div></div>')

        if self.rejected_orders:
            counts: dict[str, int] = {}
            for msg in self.rejected_orders:
                counts[msg] = counts.get(msg, 0) + 1
            rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>"
                           for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
            blocks.append('<div class="card"><h2>委托拒单统计</h2>'
                          f'<table><tbody>{rows}</tbody></table>'
                          '<div class="muted">拒单占比过高说明限价缓冲或'
                          '交易时机设置不合理，回测成交率与实盘会有差距</div></div>')

        return "".join(blocks)

    def _section_charts(self) -> str:
        equity = self.equity_df["equity"] if "equity" in self.equity_df else pd.Series(
            dtype=float)
        blocks = [
            _fig_html(charts.equity_curve(equity, self.benchmark)),
            _fig_html(charts.drawdown_curve(equity)),
            _fig_html(charts.monthly_heatmap(equity)),
        ]

        if self.bars is not None and not self.bars.empty:
            sym = str(self.bars.attrs.get("vt_symbol", ""))
            blocks.append(_fig_html(
                charts.price_with_trades(self.bars, self.trades_df, sym)))

        if not self.trades_df.empty:
            n_symbols = self.trades_df["symbol"].nunique()
            if n_symbols > 1:
                blocks.append(_fig_html(
                    charts.position_count(equity.index, self.trades_df)))
                blocks.append(_fig_html(charts.symbol_pnl(self.trades_df)))

        return "".join(f'<div class="card">{b}</div>' for b in blocks)

    def _section_trades(self, limit: int = 50) -> str:
        if self.trades_df.empty:
            return ('<div class="card"><h2>成交明细</h2>'
                    '<div class="muted">本次回测没有产生任何成交。'
                    '常见原因：数据不足以计算指标、信号条件从未满足、'
                    '或委托全部被涨跌停/限价条件拒绝。</div></div>')

        df = self.trades_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime", ascending=False).head(limit)

        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
        rows = []
        for r in df.itertuples(index=False):
            cells = []
            for col, v in zip(df.columns, r):
                if isinstance(v, float):
                    v = f"{v:,.2f}"
                elif isinstance(v, pd.Timestamp):
                    v = v.strftime("%Y-%m-%d %H:%M")
                cells.append(f"<td>{html.escape(str(v))}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")

        more = ""
        if len(self.trades_df) > limit:
            more = (f'<div class="muted">仅显示最近 {limit} 笔，'
                    f'共 {len(self.trades_df):,} 笔，完整明细见 CSV</div>')
        return (f'<div class="card"><h2>成交明细</h2><table>'
                f'<thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table>{more}</div>')

    # ------------------------------------------------------------ 输出

    def to_html(self) -> str:
        generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sub = html.escape(self.subtitle) if self.subtitle else ""
        return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(self.title)}</title>
<style>{_CSS}</style>
<script>{get_plotlyjs()}</script>
</head><body><div class="wrap">
<h1>{html.escape(self.title)}</h1>
<div class="sub">{sub}{' · ' if sub else ''}生成于 {generated}</div>
{self._section_bias()}
{self._section_diagnostics()}
{self._section_kpi()}
{self._section_charts()}
{self._section_detail()}
{self._section_trades()}
<div class="muted">
本报告由 qmtquant 生成。回测结果不代表未来收益；实盘存在滑点、
冲击成本与成交不确定性，回测无法完全刻画。
</div>
</div></body></html>"""

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html(), encoding="utf-8")
        return path


def build_report(engine, stats, path: str | Path, title: str = "回测报告",
                 subtitle: str = "", benchmark: pd.Series | None = None,
                 bars: pd.DataFrame | None = None) -> Path:
    """从 BacktestEngine 直接生成报告的便捷入口"""
    bias = engine.universe.describe_bias() if engine.universe else None
    return BacktestReport(
        stats=stats,
        equity_df=engine.get_equity_df(),
        trades_df=engine.get_trades_df(),
        bias_report=bias,
        benchmark=benchmark,
        bars=bars,
        undersized_orders=getattr(engine, "undersized_orders", None),
        rejected_orders=[o.message for o in getattr(engine, "orders", [])
                         if o.message],
        title=title,
        subtitle=subtitle,
    ).save(path)


def bars_to_frame(bars: list, vt_symbols: list[str]) -> pd.DataFrame | None:
    """把 BarData 列表转成绘图用的价格表。

    只在**单标的**回测时返回数据：多标的画在一张价格图上没有可读性
    （股价量级差几十倍），多标的场景改用持仓只数与分标的盈亏图。
    """
    if not bars or len(set(vt_symbols)) != 1:
        return None

    target = vt_symbols[0]
    rows = [(b.datetime, b.close_price) for b in bars if b.vt_symbol == target]
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["datetime", "close"]).set_index("datetime")
    df.attrs["vt_symbol"] = target
    return df
