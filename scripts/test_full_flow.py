"""完整交易链路联调（模拟盘专用）。

分两组，覆盖两条不同的回报路径：

- A 组「挂单 → 撤单」：限价挂在跌停附近，不会成交，验证撤单链路与状态机
- B 组「挂单 → 成交」：限价略高于卖一，立即成交，验证成交回报与持仓更新

⚠ B 组会真实建仓。A 股 T+1，当日买入次日才能卖，所以脚本结束后
持仓会保留 —— 这是 A 股规则，不是脚本的缺陷。

全程走生产路径 LiveEngine → RiskManager → MiniQmtGateway，不绕过任何一环。

用法：
    python scripts/test_full_flow.py --dry-run          # 只打印计划
    python scripts/test_full_flow.py --yes              # 两组都跑
    python scripts/test_full_flow.py --yes --only cancel  # 只跑撤单组
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import ACTIVE_STATUSES, Direction, OrderType  # noqa: E402
from qmtquant.engine.live_engine import LiveEngine  # noqa: E402
from qmtquant.event.engine import (  # noqa: E402
    EVENT_ORDER,
    EVENT_TRADE,
    EventEngine,
)
from qmtquant.gateway.miniqmt_gateway import MiniQmtGateway  # noqa: E402
from qmtquant.risk.risk_manager import RiskManager  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize, to_xt_symbol  # noqa: E402


def get_quote(vt_symbol: str) -> dict:
    """取实时买卖盘口。用盘口而非收盘价，才能算准「一定成交」的价格。"""
    from xtquant import xtdata
    xtdata.enable_hello = False

    tick = xtdata.get_full_tick([to_xt_symbol(vt_symbol)])
    d = tick.get(to_xt_symbol(vt_symbol))
    if not d:
        raise RuntimeError(f"取不到 {vt_symbol} 盘口，确认已订阅且在交易时段")

    pre_close = float(d.get("lastClose") or 0)
    return {
        "last": float(d.get("lastPrice") or 0),
        "bid1": float((d.get("bidPrice") or [0])[0]),
        "ask1": float((d.get("askPrice") or [0])[0]),
        "pre_close": pre_close,
        "limit_up": round(pre_close * 1.1, 2),
        "limit_down": round(pre_close * 1.1 - pre_close * 0.2, 2),
    }


def wait_events(seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        time.sleep(0.2)


def show_orders(orders: list, tag: str) -> None:
    print(f"  收到委托回报 {len(orders)} 条：")
    for o in orders:
        msg = f" 备注={o.message}" if o.message else ""
        print(f"    [{tag}] {o.orderid} {o.vt_symbol} {o.direction.value} "
              f"价={o.price:.2f} 量={o.volume:.0f} 已成={o.traded:.0f} "
              f"状态={o.status.value}{msg}")


def show_trades(trades: list) -> None:
    if not trades:
        print("  无成交回报")
        return
    print(f"  收到成交回报 {len(trades)} 条：")
    for t in trades:
        print(f"    [成交] {t.tradeid} {t.vt_symbol} {t.direction.value} "
              f"价={t.price:.2f} 量={t.volume:.0f} "
              f"金额={t.price * t.volume:,.2f}")


def run_cancel_case(engine, quote, vt_symbol, volume, wait, orders, trades):
    """A 组：挂跌停附近的买单，不成交，然后撤掉"""
    print("\n" + "=" * 62)
    print("A 组：报单 → 撤单（不求成交）")
    print("=" * 62)

    price = round(quote["pre_close"] * 0.91, 2)
    print(f"  委托：买入 {volume} 股 @ {price:.2f}  "
          f"金额 {price * volume:,.2f} 元（跌停 {quote['limit_down']:.2f}，"
          f"远低于市价 {quote['last']:.2f}，不应成交）")

    orders.clear(); trades.clear()
    vt_orderid = engine.send_order("FLOW_CANCEL", vt_symbol, Direction.LONG,
                                   price, volume, OrderType.LIMIT)
    if not vt_orderid:
        print("  [FAIL] 委托未发出，详见日志")
        return False

    print(f"  已发出 {vt_orderid}，等待回报 {wait:.0f}s ...")
    wait_events(wait)
    show_orders(orders, "委托")
    show_trades(trades)

    order = engine.orders.get(vt_orderid)
    if order is None:
        print("  [FAIL] 本地订单簿没有该委托")
        return False
    if order.traded > 0:
        print(f"  [WARN] 意外成交了 {order.traded} 股")

    if order.status not in ACTIVE_STATUSES:
        print(f"  委托已是终态 {order.status.value}，无需撤单")
        return True

    print("  发送撤单 ...")
    orders.clear()
    engine.cancel_order(vt_orderid)
    wait_events(wait)
    show_orders(orders, "撤单后")

    final = engine.orders.get(vt_orderid)
    ok = final is not None and not final.is_active()
    print(f"  最终状态：{final.status.value if final else '未知'}  "
          f"{'[OK] 撤单链路通过' if ok else '[FAIL] 委托仍处于活动状态'}")
    return ok


def run_fill_case(engine, quote, vt_symbol, volume, wait, orders, trades):
    """B 组：挂略高于卖一的买单，立即成交"""
    print("\n" + "=" * 62)
    print("B 组：报单 → 成交")
    print("=" * 62)

    # 用卖一 + 2 分钱确保吃到，但不超过涨停
    ask = quote["ask1"] or quote["last"]
    price = min(round(ask + 0.02, 2), quote["limit_up"])
    print(f"  委托：买入 {volume} 股 @ {price:.2f}  "
          f"金额 {price * volume:,.2f} 元（卖一 {ask:.2f}，应立即成交）")
    print("  注意：A 股 T+1，成交后当日不可卖出，持仓会保留到次日")

    orders.clear(); trades.clear()
    vt_orderid = engine.send_order("FLOW_FILL", vt_symbol, Direction.LONG,
                                   price, volume, OrderType.LIMIT)
    if not vt_orderid:
        print("  [FAIL] 委托未发出，详见日志")
        return False

    print(f"  已发出 {vt_orderid}，等待回报 {wait:.0f}s ...")
    wait_events(wait)
    show_orders(orders, "委托")
    show_trades(trades)

    order = engine.orders.get(vt_orderid)
    filled = order is not None and order.traded > 0
    print(f"  {'[OK] 成交链路通过' if filled else '[FAIL] 未成交'}"
          f"（已成 {order.traded if order else 0:.0f} / {volume}）")

    # 未成交则撤掉，不留隔夜挂单
    if order is not None and order.is_active():
        print("  仍有未成交部分，发送撤单 ...")
        engine.cancel_order(vt_orderid)
        wait_events(wait)
    return filled


def main() -> int:
    parser = argparse.ArgumentParser(description="完整交易链路联调（模拟盘）")
    parser.add_argument("--symbol", default="000001.SZ")
    parser.add_argument("--volume", type=int, default=100, help="每笔股数")
    parser.add_argument("--wait", type=float, default=5.0, help="等待回报秒数")
    parser.add_argument("--only", choices=["cancel", "fill"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    if cfg.gateway.name != "miniqmt":
        print(f"当前网关是 {cfg.gateway.name}，本脚本用于 miniQMT")
        return 1
    if not cfg.gateway.account_id:
        print("config.yaml 未配置 gateway.account_id")
        return 1

    vt_symbol = normalize(args.symbol)
    quote = get_quote(vt_symbol)

    print("=" * 62)
    print("完整交易链路联调（模拟盘）")
    print("=" * 62)
    print(f"账户    : {cfg.gateway.account_id}")
    print(f"标的    : {vt_symbol}")
    print(f"盘口    : 最新 {quote['last']:.2f}  买一 {quote['bid1']:.2f}  "
          f"卖一 {quote['ask1']:.2f}  昨收 {quote['pre_close']:.2f}")
    print(f"每笔    : {args.volume} 股，约 "
          f"{quote['last'] * args.volume:,.0f} 元")
    print(f"风控上限: 单笔 {cfg.risk.max_order_value:,.0f} 元")
    print("=" * 62)

    if args.dry_run:
        print("\n[dry-run] 未发送任何委托")
        return 0
    if not args.yes:
        print("\n未加 --yes，不发送委托")
        return 0

    # ---- 装配生产路径
    event_engine = EventEngine()
    orders, trades = [], []
    event_engine.register(EVENT_ORDER, lambda e: orders.append(e.data))
    event_engine.register(EVENT_TRADE, lambda e: trades.append(e.data))
    event_engine.start()

    gateway = MiniQmtGateway(event_engine)
    risk = RiskManager(cfg.risk, event_engine)
    engine = LiveEngine(event_engine, gateway, risk)

    if not gateway.connect({
        "qmt_path": cfg.gateway.qmt_path,
        "account_id": cfg.gateway.account_id,
        "account_type": cfg.gateway.account_type,
    }):
        print("网关连接失败")
        event_engine.stop()
        return 1

    print("\n--- 开始前对账 ---")
    engine.reconcile()

    results = {}
    try:
        if args.only != "fill":
            results["撤单链路"] = run_cancel_case(
                engine, quote, vt_symbol, args.volume, args.wait, orders, trades)
        if args.only != "cancel":
            results["成交链路"] = run_fill_case(
                engine, quote, vt_symbol, args.volume, args.wait, orders, trades)
    finally:
        print("\n--- 结束后对账 ---")
        engine.reconcile()
        engine.close()
        event_engine.stop()

    print("\n" + "=" * 62)
    for name, ok in results.items():
        print(f"  {name}: {'通过' if ok else '失败'}")
    print("=" * 62)
    print("交易明细见 logs/trade.log")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
