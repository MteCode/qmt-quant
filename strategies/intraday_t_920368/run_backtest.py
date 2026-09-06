from __future__ import annotations
import argparse, json, sys
from dataclasses import dataclass
from pathlib import Path
import joblib, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qmtquant.core.constants import Direction, Exchange
from qmtquant.core.objects import TradeData
from qmtquant.engine.performance import calculate_stats
from qmtquant.report.html_report import BacktestReport
try:
    from .train_model import DATA, FEATURES, OUT, make_features
except ImportError:
    from train_model import DATA, FEATURES, OUT, make_features

RESULT = Path(__file__).resolve().parent / "backtest"

@dataclass
class Sim:
    cash: float
    base_shares: int
    t_shares: int
    t_sellable: int
    peak: float
    t_cost: float
    t_pnl: float = 0.0
    halted: bool = False

def run(df, model, initial=200000.0, base_value=100000.0, t_value=100000.0):
    days = pd.Index(df.index.normalize()).unique()
    test_days = days[-242:] if len(days) > 242 else days
    test = df[df.index.normalize().isin(test_days)].copy()
    first = float(test.iloc[0].close)
    base, t = int(base_value / first / 100) * 100, int(t_value / first / 100) * 100
    sim = Sim(initial - (base + t) * first, base, t, t, initial, first)
    commission_rate, stamp_rate, slippage = 0.0003, 0.001, 0.0005
    features = make_features(df).reindex(test.index).fillna(0)
    test = test.assign(prob_up=model.predict_proba(features[FEATURES])[:, 1])
    trades, equity_rows, pending_sell = [], [], None
    for dt, row in test.iterrows():
        px, tod = float(row.close), dt.hour * 60 + dt.minute
        if tod == 570: sim.t_sellable = sim.t_shares
        equity = sim.cash + (sim.base_shares + sim.t_shares) * px
        dd = equity / max(sim.peak, 1e-9) - 1; sim.peak = max(sim.peak, equity)
        if dd <= -0.12 and not sim.halted:
            for label, vol in (("底仓风险退出", sim.base_shares), ("T仓风险退出", sim.t_shares)):
                if vol:
                    sell_px = px * (1 - slippage); fee = sell_px * vol * (commission_rate + stamp_rate)
                    sim.cash += sell_px * vol - fee
                    trades.append((dt, "卖出", sell_px, vol, fee, label, 0.0, sim.t_cost))
            sim.base_shares = sim.t_shares = sim.t_sellable = 0; sim.halted = True
        day_slice = test.loc[test.index.normalize() == dt.normalize()].loc[:dt]
        # BSE分钟数据的volume以手计，amount以元计，因此VWAP需除以100股/手。
        vol_sum = day_slice.volume.sum(); vwap = float(day_slice.amount.sum() / (vol_sum * 100)) if vol_sum > 0 else px
        # 只有高于VWAP且模型转弱才卖；只有低于卖价且模型转强才买，拒绝机械亏损T。
        if (not sim.halted and tod == 615 and sim.t_sellable > 0
                and px >= vwap * 1.0005 and row.prob_up < 0.54):
            vol = sim.t_sellable; sell_px = px * (1 - slippage); fee = sell_px * vol * (commission_rate + stamp_rate)
            sim.cash += sell_px * vol - fee; sim.t_shares -= vol; sim.t_sellable = 0
            pending_sell = (sell_px, vol, fee)
            trades.append((dt, "卖出", sell_px, vol, fee, "VWAP上方+模型转弱，T先卖", 0.0, sim.t_cost))
        if not sim.halted and tod == 870 and pending_sell is not None:
            sell_px, vol, sell_fee = pending_sell; buy_px = px * (1 + slippage)
            fee = buy_px * vol * commission_rate; sim.cash -= buy_px * vol + fee; sim.t_shares += vol
            pnl = (sell_px - buy_px) * vol - sell_fee - fee; sim.t_pnl += pnl
            sim.t_cost = max(0.0, sim.t_cost - pnl / max(base + t, 1))
            reason = ("低于卖出价0.15%+模型转强，T完成" if buy_px <= sell_px * (1 - 0.0015) and row.prob_up > 0.46
                      else "14:30强制回补，记录T失败，不留隔夜裸卖仓")
            trades.append((dt, "买入", buy_px, vol, fee, reason, pnl, sim.t_cost)); pending_sell = None
        equity_rows.append((dt, sim.cash + (sim.base_shares + sim.t_shares) * px, sim.base_shares, sim.t_shares, row.prob_up, dd, sim.t_pnl, sim.t_cost))
    eq = pd.DataFrame(equity_rows, columns=["datetime", "equity", "base_shares", "t_shares", "prob_up", "drawdown", "t_pnl", "t_cost"]).set_index("datetime")
    td = pd.DataFrame(trades, columns=["datetime", "direction", "price", "volume", "commission", "reason", "t_pnl", "effective_cost"])
    if td.empty: td = pd.DataFrame(columns=["datetime", "direction", "price", "volume", "commission", "reason", "t_pnl", "effective_cost"])
    td["symbol"], td["strategy"], td["amount"] = "920368.BSE", "IntradayT920368Strategy", td.price * td.volume
    objs = [TradeData(symbol="920368", exchange=Exchange.BSE, direction=Direction.LONG if r.direction == "买入" else Direction.SHORT, price=r.price, volume=r.volume, commission=r.commission, datetime=r.datetime, reference="IntradayT920368Strategy") for r in td.itertuples()]
    stats = calculate_stats(eq.equity, objs, initial); stats.max_drawdown_limit = 0.15
    return test, eq, td, stats, first

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(DATA)); args = ap.parse_args()
    df = pd.read_parquet(args.data).sort_index(); model = joblib.load(OUT / "gbm_t_model.joblib")["model"]
    bars, eq, trades, stats, first = run(df, model); RESULT.mkdir(parents=True, exist_ok=True)
    eq.to_csv(RESULT / "equity.csv"); trades.to_csv(RESULT / "trades.csv", index=False)
    summary = stats.to_dict() | {"symbol":"920368.BSE", "strategy":"IntradayT920368Strategy", "base_value":100000, "trade_value":100000, "risk_rule":"12%组合预警紧急退出，15%最大回撤硬上限；正常情况下底仓不动", "signal_rule":"10:15高于VWAP0.05%且模型转弱卖出；14:30低于卖价0.15%且模型转强买回", "oos_days":int(bars.index.normalize().nunique()), "drawdown_ok":stats.drawdown_ok, "t_realized_pnl":float(eq.t_pnl.iloc[-1]), "t_cost_reduction_per_share":float(eq.t_pnl.iloc[-1] / max(int(100000 / first / 100) * 200, 1)), "t_rounds":int((trades.direction == "买入").sum())}
    (RESULT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    class E: pass
    e=E(); e.universe=None; e.undersized_orders={}; e.orders=[]; e.get_equity_df=lambda: eq[["equity","drawdown"]].assign(returns=eq.equity.pct_change().fillna(0)); e.get_trades_df=lambda: trades
    plot_bars=bars[["close"]].copy(); plot_bars.attrs["vt_symbol"]="920368.BSE"
    BacktestReport(stats,e.get_equity_df(),trades,bars=plot_bars,title="920368日内做T策略回测",subtitle="最近一年1分钟数据；每笔T均要求卖高买低价差确认").save(RESULT / "report.html")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__": main()
