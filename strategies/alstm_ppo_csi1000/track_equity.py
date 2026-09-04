"""实盘净值跟踪 —— 每日记一笔，与回测预期对照。

## 为什么必须有

回测说年化 9%、回撤 14%，实盘到底走成什么样，现在没有任何记录。
账户里只有当前持仓和资金，没有历史净值序列，因此答不出三个问题：

1. 实盘的收益与回撤是多少
2. 与回测预期偏离多少 —— 偏离持续扩大意味着回测有系统性高估
3. 偏离来自哪里 —— 选股不同、仓位不同、还是执行损耗

券商能查到当前总资产，但**查不到历史序列**（对账单要另外下载且格式各异）。
所以必须自己每天记一笔。漏记的日子补不回来 —— 这是唯一不能事后补做的事。

## 记什么

每日总资产、持仓市值、可用资金、当日盈亏、累计收益、从峰值算的回撤。
另记当日信号仓位与实际仓位，两者的差是执行缺口。

## 与回测的对照

回测净值从 `backtest/ppo_equity.csv` 读，按日期对齐后算：

- 累计收益差 —— 实盘 vs 回测同期
- 回撤差 —— 实盘回撤是否超出回测预期
- 仓位差 —— 实际仓位与策略目标的偏离

用法::

    python strategies/alstm_ppo_csi1000/track_equity.py          # 记录今天
    python strategies/alstm_ppo_csi1000/track_equity.py --report # 只看曲线与对照
"""
import argparse
import csv
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paths  # noqa: E402

EQUITY_CSV = paths.STATE_DIR / "live_equity.csv"

FIELDS = ["date", "total_asset", "market_value", "cash",
          "n_holdings", "daily_pnl", "daily_return",
          "cum_return", "peak", "drawdown",
          "signal_weight", "actual_weight", "source"]


def load_history() -> list:
    if not EQUITY_CSV.exists():
        return []
    try:
        with open(EQUITY_CSV, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def save_history(rows: list) -> None:
    paths.ensure_dirs()
    with open(EQUITY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def query_account(cfg) -> dict:
    """从 miniQMT 取当前资产与持仓。"""
    from qmtquant.event.engine import EventEngine
    from qmtquant.gateway.xt_gateway import XtGateway

    ee = EventEngine()
    ee.start()
    gw = XtGateway(ee)
    if not gw.connect({"qmt_path": cfg.gateway.qmt_path,
                       "account_id": cfg.gateway.account_id}):
        ee.stop()
        return {"error": "连接 miniQMT 失败"}
    time.sleep(1)

    from xtquant.xttype import StockAccount
    acct = StockAccount(cfg.gateway.account_id)
    asset = gw.trader.query_stock_asset(acct)
    if asset is None:
        gw.close(); ee.stop()
        return {"error": "查询账户资产失败"}

    positions = gw.trader.query_stock_positions(acct) or []
    mv = sum(float(p.market_value) for p in positions if p.volume > 0)
    n = sum(1 for p in positions if p.volume > 0)

    gw.close()
    ee.stop()
    return {"total_asset": float(asset.total_asset),
            "cash": float(asset.cash), "market_value": mv, "n_holdings": n}


def signal_weight() -> float | None:
    """当前信号的目标仓位。"""
    if not paths.LATEST_SIGNAL.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(paths.LATEST_SIGNAL, encoding="utf-8-sig")
        return float(df["weight"].sum()) if "weight" in df else None
    except Exception:
        return None


def record(cfg, capital: float, when: str) -> dict:
    acc = query_account(cfg)
    if "error" in acc:
        return acc

    rows = load_history()
    rows = [r for r in rows if r["date"] != when]   # 同日重复记录以最新为准

    total = acc["total_asset"]
    prev = float(rows[-1]["total_asset"]) if rows else capital
    peak = max([float(r["peak"]) for r in rows] + [total]) if rows else total
    peak = max(peak, total)

    sw = signal_weight()
    aw = acc["market_value"] / total if total else 0.0

    row = {
        "date": when,
        "total_asset": round(total, 2),
        "market_value": round(acc["market_value"], 2),
        "cash": round(acc["cash"], 2),
        "n_holdings": acc["n_holdings"],
        "daily_pnl": round(total - prev, 2),
        "daily_return": round(total / prev - 1, 6) if prev else 0.0,
        "cum_return": round(total / capital - 1, 6) if capital else 0.0,
        "peak": round(peak, 2),
        "drawdown": round(1 - total / peak, 6) if peak else 0.0,
        "signal_weight": round(sw, 4) if sw is not None else "",
        "actual_weight": round(aw, 4),
        "source": "miniqmt",
    }
    rows.append(row)
    rows.sort(key=lambda r: r["date"])
    save_history(rows)
    return {"row": row, "n_days": len(rows)}


def compare_backtest(rows: list, capital: float) -> dict | None:
    """与回测净值对照。"""
    import pandas as pd

    bt = paths.BACKTEST_DIR / "ppo_equity.csv"
    if not bt.exists() or not rows:
        return None
    try:
        b = pd.read_csv(bt)
    except (OSError, ValueError):
        return None
    dcol = next((c for c in b.columns
                 if c.lower() in ("date", "datetime")), None)
    vcol = next((c for c in b.columns
                 if c.lower() in ("equity", "value", "total")), None)
    if dcol is None or vcol is None:
        return None

    b[dcol] = pd.to_datetime(b[dcol])
    b = b.set_index(dcol)[vcol].sort_index()
    bt_ret = b / b.iloc[0] - 1

    live_start = pd.Timestamp(rows[0]["date"])
    seg = bt_ret[bt_ret.index >= live_start]
    if seg.empty:
        # 实盘起始日晚于回测区间 —— 用回测整段的日均收益做基准
        n = len(rows)
        daily = (1 + bt_ret.iloc[-1]) ** (1 / len(bt_ret)) - 1
        expect = (1 + daily) ** n - 1
        return {"mode": "外推", "expect_return": expect,
                "live_return": float(rows[-1]["cum_return"]),
                "bt_daily": daily, "days": n}

    return {"mode": "对齐",
            "expect_return": float(seg.iloc[-1] - seg.iloc[0]),
            "live_return": float(rows[-1]["cum_return"]),
            "days": len(rows)}


def show(capital: float) -> int:
    rows = load_history()
    if not rows:
        print("尚无实盘净值记录。收盘后运行本脚本记录第一笔。")
        return 1

    print("=" * 66)
    print(f"实盘净值  {rows[0]['date']} ~ {rows[-1]['date']}  共 {len(rows)} 个交易日")
    print("=" * 66)

    last = rows[-1]
    dd = [float(r["drawdown"]) for r in rows]
    rets = [float(r["daily_return"]) for r in rows[1:]]
    print(f"  总资产    {float(last['total_asset']):>12,.0f}"
          f"   （本金 {capital:,.0f}）")
    print(f"  累计收益  {float(last['cum_return']):>+12.2%}")
    print(f"  当前回撤  {float(last['drawdown']):>12.2%}"
          f"   最大 {max(dd):.2%}")
    print(f"  持仓      {last['n_holdings']} 只"
          f"   实际仓位 {float(last['actual_weight']):.1%}"
          + (f"   信号仓位 {float(last['signal_weight']):.1%}"
             if last["signal_weight"] else ""))
    if rets:
        import statistics
        win = sum(1 for r in rets if r > 0) / len(rets)
        print(f"  日胜率    {win:>12.1%}   "
              f"日均 {statistics.fmean(rets):+.3%}")

    cmp = compare_backtest(rows, capital)
    if cmp:
        gap = cmp["live_return"] - cmp["expect_return"]
        print(f"\n与回测对照（{cmp['mode']}）:")
        print(f"  回测同期预期  {cmp['expect_return']:>+10.2%}")
        print(f"  实盘实际      {cmp['live_return']:>+10.2%}")
        print(f"  偏离          {gap:>+10.2%}")
        if gap < -0.02:
            print("  实盘落后回测超 2 个百分点 —— 检查执行损耗：")
            print("    运行 reconcile.py 看成交率与滑点")

    print(f"\n最近 10 日:")
    print(f"  {'日期':<12s}{'总资产':>12s}{'日收益':>9s}"
          f"{'累计':>9s}{'回撤':>8s}{'仓位':>7s}")
    for r in rows[-10:]:
        print(f"  {r['date']:<12s}{float(r['total_asset']):>12,.0f}"
              f"{float(r['daily_return']):>+9.2%}"
              f"{float(r['cum_return']):>+9.2%}"
              f"{float(r['drawdown']):>8.2%}"
              f"{float(r['actual_weight']):>7.0%}")
    print(f"\n明细: {EQUITY_CSV}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="实盘净值跟踪")
    p.add_argument("--report", action="store_true", help="只看曲线与对照")
    p.add_argument("--date", default=None, help="记录日期，默认今天")
    args = p.parse_args()

    from qmtquant.config import get_config
    cfg = get_config()
    capital = float(paths.load_params().get("capital") or 500_000)

    if args.report:
        return show(capital)

    when = args.date or date.today().isoformat()
    print(f"记录实盘净值 {when} ...")
    r = record(cfg, capital, when)
    if "error" in r:
        print(f"  {r['error']}")
        return 1

    row = r["row"]
    print(f"  总资产 {row['total_asset']:,.0f}   "
          f"持仓 {row['n_holdings']} 只   "
          f"仓位 {row['actual_weight']:.1%}")
    print(f"  当日 {row['daily_return']:+.2%}   "
          f"累计 {row['cum_return']:+.2%}   "
          f"回撤 {row['drawdown']:.2%}")
    print(f"  已记录 {r['n_days']} 个交易日")
    print(f"\n查看曲线: python {Path(__file__).name} --report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
