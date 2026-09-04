"""成交回报对账 —— 下单意图与实际成交的差异。

## 为什么必须有

下单脚本只知道「委托发出去了」，不知道后来怎么样。实盘里下面几件事
都会让实际结果偏离意图，而且**不会报错**：

- 部分成交：委托 1000 股只成交 300 股
- 废单：涨跌停、超出可用资金、账户权限不足
- 滑点：市价单实际成交价与信号价的差
- 撤单：长时间未成交被系统撤销

不对账就等于蒙眼交易：回测显示年化 9%，实盘可能因为持续的负滑点
和废单变成 3%，而你只会看到「账户没涨」，找不到原因。

## 对账三问

1. **成交率** —— 委托了多少，实际成交多少
2. **滑点** —— 实际成交价与决策时价格差多少（分买卖方向，买贵卖便宜都是负滑点）
3. **缺口** —— 哪些标的完全没买进，为什么

## 数据来源

`executions/` 下的下单记录（意图）与 miniQMT 的当日委托/成交查询（结果）。
两者按委托编号关联；关联不上的单独列出 —— 那通常意味着下单脚本
记录的编号与券商返回的对不上，本身就是要修的问题。

用法::

    python strategies/alstm_ppo_csi1000/reconcile.py
    python strategies/alstm_ppo_csi1000/reconcile.py --date 2026-09-04
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import paths  # noqa: E402

RECON_DIR = paths.STRATEGY_DIR / "reconcile"


def load_intent(date: str) -> list:
    """读当日的下单意图。"""
    d = paths.STRATEGY_DIR / "executions"
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob(f"*{date}*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        orders = data.get("orders") or data.get("sent") or []
        for o in orders:
            out.append({**o, "_file": p.name})
    return out


def query_broker(cfg, date: str) -> dict:
    """从 miniQMT 拉当日委托与成交。"""
    from qmtquant.event.engine import EventEngine
    from qmtquant.gateway.xt_gateway import XtGateway

    ee = EventEngine()
    ee.start()
    gw = XtGateway(ee)
    if not gw.connect({"qmt_path": cfg.gateway.qmt_path,
                       "account_id": cfg.gateway.account_id}):
        ee.stop()
        return {"error": "连接 miniQMT 失败 —— 请确认已启动并登录"}
    time.sleep(1)

    from xtquant.xttype import StockAccount
    acct = StockAccount(cfg.gateway.account_id)

    orders, trades = [], []
    try:
        for o in gw.trader.query_stock_orders(acct) or []:
            orders.append({
                "order_id": str(o.order_id),
                "stock_code": o.stock_code,
                "direction": "买入" if o.order_type == 23 else "卖出",
                "order_volume": int(o.order_volume),
                "traded_volume": int(o.traded_volume),
                "price": float(o.price),
                "status": int(o.order_status),
                "remark": getattr(o, "order_remark", ""),
            })
        for t in gw.trader.query_stock_trades(acct) or []:
            trades.append({
                "order_id": str(t.order_id),
                "trade_id": str(t.traded_id),
                "stock_code": t.stock_code,
                "direction": "买入" if t.order_type == 23 else "卖出",
                "traded_price": float(t.traded_price),
                "traded_volume": int(t.traded_volume),
            })
    except Exception as e:
        gw.close(); ee.stop()
        return {"error": f"查询失败: {e}"}

    gw.close()
    ee.stop()
    return {"orders": orders, "trades": trades}


#: 委托状态码 -> 含义。与 xt_gateway._on_order 的映射保持一致
STATUS_NAME = {
    48: "已报", 49: "未成交", 50: "部分成交", 51: "部分成交",
    52: "已撤", 53: "已撤", 54: "废单", 55: "已成", 56: "已成",
}


def reconcile(intent: list, broker: dict, prices: dict) -> dict:
    """把意图与结果对上，算成交率与滑点。"""
    orders = broker.get("orders", [])
    trades = broker.get("trades", [])

    # 按委托编号汇总成交
    by_order = {}
    for t in trades:
        e = by_order.setdefault(t["order_id"], {"vol": 0, "amt": 0.0})
        e["vol"] += t["traded_volume"]
        e["amt"] += t["traded_price"] * t["traded_volume"]

    rows = []
    for o in orders:
        got = by_order.get(o["order_id"], {"vol": 0, "amt": 0.0})
        avg = got["amt"] / got["vol"] if got["vol"] else None
        ref = prices.get(o["stock_code"])
        # 滑点统一定义为「对我不利的方向为负」：买贵了、卖便宜了都是负
        slip = None
        if avg is not None and ref:
            raw = (avg - ref) / ref
            slip = -raw if o["direction"] == "买入" else raw
        rows.append({
            "order_id": o["order_id"],
            "stock_code": o["stock_code"],
            "direction": o["direction"],
            "ordered": o["order_volume"],
            "traded": got["vol"],
            "fill_rate": got["vol"] / o["order_volume"]
            if o["order_volume"] else 0.0,
            "avg_price": round(avg, 4) if avg is not None else None,
            "ref_price": ref,
            "slippage": round(slip, 6) if slip is not None else None,
            "status": STATUS_NAME.get(o["status"], str(o["status"])),
        })

    # 意图里有、券商侧找不到的委托
    known = {r["stock_code"] for r in rows}
    missing = [i for i in intent
               if i.get("xt_code") and i["xt_code"] not in known]

    n_ord = sum(r["ordered"] for r in rows)
    n_trd = sum(r["traded"] for r in rows)
    slips = [r["slippage"] for r in rows if r["slippage"] is not None]
    import statistics
    return {
        "rows": rows,
        "missing": missing,
        "n_orders": len(rows),
        "ordered_volume": n_ord,
        "traded_volume": n_trd,
        "fill_rate": n_trd / n_ord if n_ord else 0.0,
        "n_filled": sum(1 for r in rows if r["fill_rate"] >= 0.999),
        "n_partial": sum(1 for r in rows if 0 < r["fill_rate"] < 0.999),
        "n_zero": sum(1 for r in rows if r["fill_rate"] == 0),
        "slippage_mean": round(statistics.fmean(slips), 6) if slips else None,
        "slippage_median": round(statistics.median(slips), 6) if slips else None,
        "slippage_worst": round(min(slips), 6) if slips else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="成交回报对账")
    p.add_argument("--date", default=None, help="对账日期，默认今天")
    args = p.parse_args()
    date = args.date or datetime.now().strftime("%Y-%m-%d")

    from qmtquant.config import get_config
    cfg = get_config()

    print("=" * 62)
    print(f"成交回报对账  {date}")
    print(f"  账号 {cfg.gateway.account_id}")
    print("=" * 62)

    intent = load_intent(date)
    print(f"\n下单意图记录: {len(intent)} 笔")

    print("\n查询券商委托与成交...")
    broker = query_broker(cfg, date)
    if "error" in broker:
        print(f"  {broker['error']}")
        return 1
    print(f"  委托 {len(broker['orders'])} 笔，成交回报 {len(broker['trades'])} 笔")

    # 决策时价格：优先用意图里记录的，没有则留空（滑点算不出来时不硬凑）
    prices = {}
    for i in intent:
        if i.get("xt_code") and i.get("price"):
            prices[i["xt_code"]] = float(i["price"])

    r = reconcile(intent, broker, prices)

    print("\n" + "-" * 62)
    print("成交情况")
    print("-" * 62)
    print(f"  委托 {r['n_orders']} 笔 / {r['ordered_volume']:,} 股")
    print(f"  成交 {r['traded_volume']:,} 股   成交率 {r['fill_rate']:.1%}")
    print(f"  全部成交 {r['n_filled']}   部分成交 {r['n_partial']}   "
          f"完全未成交 {r['n_zero']}")

    if r["slippage_mean"] is not None:
        print("\n滑点（负数表示对我不利：买贵了或卖便宜了）")
        print(f"  均值 {r['slippage_mean']:+.4%}   "
              f"中位数 {r['slippage_median']:+.4%}   "
              f"最差 {r['slippage_worst']:+.4%}")
    else:
        print("\n滑点: 无参考价，算不出来")
        print("  下单记录里需保存决策时的价格才能计算")

    bad = [x for x in r["rows"] if x["fill_rate"] < 0.999]
    if bad:
        print("\n未完全成交的委托:")
        print(f"  {'标的':<12s}{'方向':<6s}{'委托':>7s}{'成交':>7s}"
              f"{'成交率':>8s}  状态")
        for x in bad[:20]:
            print(f"  {x['stock_code']:<12s}{x['direction']:<6s}"
                  f"{x['ordered']:>7d}{x['traded']:>7d}"
                  f"{x['fill_rate']:>8.0%}  {x['status']}")

    if r["missing"]:
        print(f"\n[注意] {len(r['missing'])} 笔意图在券商侧找不到对应委托")
        print("  多半是下单脚本记录的编号与券商返回的对不上，需要检查")

    RECON_DIR.mkdir(parents=True, exist_ok=True)
    out = RECON_DIR / f"{date}.json"
    out.write_text(json.dumps({
        "date": date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **r,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
