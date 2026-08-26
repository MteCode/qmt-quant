"""下单链路联调脚本（模拟盘专用）。

走的是完整生产路径：LiveEngine → RiskManager → Gateway，不绕过任何一环。
目的是在真实券商接口上验证：报单 → 回报 → 状态机 → 撤单 这条链路。

默认策略是「挂一笔远离市价的限价买单，不求成交，然后撤掉」——
既验证了链路，又不会真的建仓。

用法：
    # 查看将要发送什么，不实际下单
    python scripts/test_order_flow.py --symbol 000001.SZ --dry-run

    # 真实发单（模拟盘）
    python scripts/test_order_flow.py --symbol 000001.SZ --yes

    # 指定价格与数量
    python scripts/test_order_flow.py --symbol 000001.SZ --price 10.60 --volume 100 --yes
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.core.constants import ACTIVE_STATUSES, Direction, OrderType  # noqa: E402
from qmtquant.engine.live_engine import LiveEngine  # noqa: E402
from qmtquant.event.engine import EVENT_ORDER, EVENT_TRADE, Event, EventEngine  # noqa: E402
from qmtquant.gateway.miniqmt_gateway import MiniQmtGateway  # noqa: E402
from qmtquant.risk.risk_manager import RiskManager  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402
from qmtquant.utils.symbol import normalize, to_xt_symbol  # noqa: E402

#: 默认挂在跌停附近，确保不会成交
DEFAULT_DISCOUNT = 0.91


def get_reference_price(vt_symbol: str) -> float:
    """取最近收盘价，用于推算安全的挂单价"""
    from xtquant import xtdata
    xtdata.enable_hello = False
    xt_symbol = to_xt_symbol(vt_symbol)
    data = xtdata.get_market_data_ex(
        field_list=[], stock_list=[xt_symbol], period="1d",
        start_time="", end_time="", dividend_type="none", fill_data=False,
    ).get(xt_symbol)
    if data is None or data.empty:
        raise RuntimeError(f"取不到 {vt_symbol} 的行情")
    return float(data.iloc[-1]["close"])


def main() -> int:
    parser = argparse.ArgumentParser(description="下单链路联调（模拟盘）")
    parser.add_argument("--symbol", default="000001.SZ")
    parser.add_argument("--volume", type=int, default=100)
    parser.add_argument("--price", type=float, default=None,
                        help="挂单价，默认取最近收盘价 * 0.91（接近跌停，不会成交）")
    parser.add_argument("--no-cancel", action="store_true", help="不撤单，留在委托队列里")
    parser.add_argument("--wait", type=float, default=6.0, help="等待回报的秒数")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不发单")
    parser.add_argument("--yes", "-y", action="store_true", help="确认发单")
    args = parser.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)

    if cfg.gateway.name != "miniqmt":
        print(f"当前网关是 {cfg.gateway.name}，本脚本用于 miniQMT。"
              f"请在 config.yaml 中设置 gateway.name: miniqmt")
        return 1
    if not cfg.gateway.account_id:
        print("config.yaml 中未配置 gateway.account_id")
        return 1

    vt_symbol = normalize(args.symbol)
    ref = get_reference_price(vt_symbol)
    price = round(args.price if args.price else ref * DEFAULT_DISCOUNT, 2)
    amount = price * args.volume

    print("=" * 62)
    print("下单链路联调")
    print("=" * 62)
    print(f"账户      : {cfg.gateway.account_id}")
    print(f"标的      : {vt_symbol}")
    print(f"参考价    : {ref:.2f}（最近收盘）")
    print(f"委托      : 买入 {args.volume} 股 @ {price:.2f}  金额 {amount:,.2f} 元")
    print(f"预期      : 挂单价远低于市价，不应成交；随后"
          + ("保留委托" if args.no_cancel else "撤单"))
    print(f"风控上限  : 单笔 {cfg.risk.max_order_value:,.0f} 元")
    print("=" * 62)

    if args.dry_run:
        print("\n[dry-run] 未发送任何委托")
        return 0
    if not args.yes:
        print("\n未加 --yes，不发送委托。确认无误后重跑并加上 --yes")
        return 0

    # ---- 装配生产路径
    event_engine = EventEngine()
    order_events, trade_events = [], []
    event_engine.register(EVENT_ORDER, lambda e: order_events.append(e.data))
    event_engine.register(EVENT_TRADE, lambda e: trade_events.append(e.data))
    event_engine.start()

    gateway = MiniQmtGateway(event_engine)
    risk = RiskManager(cfg.risk, event_engine)
    engine = LiveEngine(event_engine, gateway, risk)

    setting = {
        "qmt_path": cfg.gateway.qmt_path,
        "account_id": cfg.gateway.account_id,
        "account_type": cfg.gateway.account_type,
    }
    if not gateway.connect(setting):
        print("网关连接失败")
        event_engine.stop()
        return 1

    engine.reconcile()

    # ---- 发单
    print("\n--- 发送委托 ---")
    vt_orderid = engine.send_order("ORDER_FLOW_TEST", vt_symbol,
                                   Direction.LONG, price, args.volume,
                                   OrderType.LIMIT)
    if not vt_orderid:
        print("委托未发出（被风控拦截或网关拒绝），详见日志")
        engine.close()
        event_engine.stop()
        return 1
    print(f"已发出，vt_orderid = {vt_orderid}")

    # ---- 等回报
    print(f"\n--- 等待回报（{args.wait:.0f}s）---")
    deadline = time.time() + args.wait
    while time.time() < deadline:
        time.sleep(0.3)

    print(f"收到委托回报 {len(order_events)} 条，成交回报 {len(trade_events)} 条")
    for o in order_events:
        print(f"  [委托] {o.vt_orderid} {o.vt_symbol} {o.direction.value} "
              f"价={o.price:.2f} 量={o.volume} 已成={o.traded} "
              f"状态={o.status.value}"
              + (f" 备注={o.message}" if o.message else ""))
    for t in trade_events:
        print(f"  [成交] {t.vt_tradeid} {t.vt_symbol} {t.direction.value} "
              f"价={t.price:.2f} 量={t.volume}")

    # ---- 撤单
    order = engine.orders.get(vt_orderid)
    if not args.no_cancel and order and order.status in ACTIVE_STATUSES:
        print("\n--- 撤单 ---")
        engine.cancel_order(vt_orderid)
        deadline = time.time() + args.wait
        while time.time() < deadline:
            time.sleep(0.3)
        final = engine.orders.get(vt_orderid)
        print(f"撤单后状态: {final.status.value if final else '未知'}")
    elif order:
        print(f"\n委托当前状态 {order.status.value}，无需撤单")

    # ---- 收尾对账
    print("\n--- 收尾对账 ---")
    engine.reconcile()

    engine.close()
    event_engine.stop()
    print("\n完成。交易明细见 logs/trade.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
