"""虚拟盘执行 —— 读取信号文件，通过 miniQMT 下单。

流程：
1. 读取 signals/target_latest.csv（目标持仓）
2. 连接 miniQMT 查询当前持仓
3. 计算差异：需要买入/卖出的股票和数量
4. 先卖后买，按市价下单

前置条件：
- miniQMT 客户端已启动并登录
- 已运行 generate_signal.py 生成当日信号

用法::

    python scripts/run_paper_trade.py
    python scripts/run_paper_trade.py --dry-run   # 只看差异，不下单
"""
import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_target() -> pd.DataFrame:
    path = Path("signals/target_latest.csv")
    if not path.exists():
        print(f"信号文件不存在: {path}")
        print("请先运行: python scripts/generate_signal.py")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"加载目标持仓: {len(df)} 只, 总金额 ¥{df['target_value'].sum():,.0f}")
    return df


def get_current_positions(gateway) -> dict:
    """查询当前持仓，返回 {vt_symbol: {'volume': int, 'available': int, 'value': float}}"""
    gateway.query_position()
    time.sleep(1)  # 等回调

    positions = {}
    # 简化：直接从 trader 查
    from xtquant.xttype import StockAccount
    account = StockAccount(gateway.account_id)
    pos_list = gateway.trader.query_stock_positions(account)
    if pos_list:
        for pos in pos_list:
            if pos.volume <= 0:
                continue
            # xtquant 用 SH/SZ，转成项目 SSE/SZSE
            code = pos.stock_code
            parts = code.split(".")
            if parts[1] == "SH":
                vt = f"{parts[0]}.SSE"
            else:
                vt = f"{parts[0]}.SZSE"
            positions[vt] = {
                "volume": pos.volume,
                "available": pos.can_use_volume,
                "avg_price": pos.avg_price,
                "market_value": pos.market_value,
            }
    return positions


def get_latest_prices(symbols: list) -> dict:
    """获取最新价格"""
    from xtquant import xtdata

    prices = {}
    for vt in symbols:
        parts = vt.split(".")
        code = parts[0]
        ex = "SH" if parts[1] == "SSE" else "SZ"
        xt_code = f"{code}.{ex}"

        tick = xtdata.get_full_tick([xt_code])
        if tick and xt_code in tick:
            t = tick[xt_code]
            prices[vt] = t.get("lastPrice", 0) or t.get("lastClose", 0)

    return prices


def calc_orders(target_df: pd.DataFrame, current_pos: dict,
                prices: dict, capital: float) -> tuple[list, list]:
    """计算买卖差异。返回 (sell_orders, buy_orders)"""
    sell_orders = []
    buy_orders = []

    target_holdings = set(target_df["vt_symbol"].tolist())
    current_holdings = set(current_pos.keys())

    # 需要卖出的：当前持有但不在目标中
    for vt in current_holdings - target_holdings:
        pos = current_pos[vt]
        if pos["available"] > 0:
            sell_orders.append({
                "vt_symbol": vt,
                "direction": "卖出",
                "volume": pos["available"],
                "reason": "清仓（不在目标持仓中）",
            })

    # 需要调整的：目标中有的
    for _, row in target_df.iterrows():
        vt = row["vt_symbol"]
        target_value = row["target_value"]
        price = prices.get(vt, 0)
        if price <= 0:
            continue

        target_shares = math.floor(target_value / price / 100) * 100
        if target_shares <= 0:
            continue

        current = current_pos.get(vt, {})
        current_shares = current.get("volume", 0)

        diff = target_shares - current_shares
        if diff > 100:
            # 买入（向下取整到 100 股）
            buy_vol = (diff // 100) * 100
            buy_orders.append({
                "vt_symbol": vt,
                "direction": "买入",
                "volume": buy_vol,
                "price": price,
                "amount": buy_vol * price,
                "reason": f"加仓 {current_shares} -> {current_shares + buy_vol}",
            })
        elif diff < -100:
            # 卖出
            available = current.get("available", 0)
            sell_vol = min(abs(diff) // 100 * 100, available)
            if sell_vol >= 100:
                sell_orders.append({
                    "vt_symbol": vt,
                    "direction": "卖出",
                    "volume": sell_vol,
                    "reason": f"减仓 {current_shares} -> {current_shares - sell_vol}",
                })

    return sell_orders, buy_orders


def execute_orders(gateway, sell_orders, buy_orders, dry_run=False):
    """先卖后买"""
    from xtquant.xttype import StockAccount
    account = StockAccount(gateway.account_id)

    total_sell = 0
    total_buy = 0

    # 卖出
    if sell_orders:
        print(f"\n--- 卖出 ({len(sell_orders)} 笔) ---")
        for order in sell_orders:
            vt = order["vt_symbol"]
            vol = order["volume"]
            parts = vt.split(".")
            xt_code = f"{parts[0]}.{'SH' if parts[1] == 'SSE' else 'SZ'}"

            print(f"  {xt_code}  {vol:>6d} 股  {order['reason']}")
            total_sell += vol

            if not dry_run:
                gateway.trader.order_stock(
                    account, xt_code, 24,  # STOCK_SELL
                    int(vol), 0,  # 0 = 市价
                    strategy_name="ALSTM_PPO",
                    order_remark="paper_sell",
                )
                time.sleep(0.2)

    # 买入
    if buy_orders:
        print(f"\n--- 买入 ({len(buy_orders)} 笔) ---")
        for order in buy_orders:
            vt = order["vt_symbol"]
            vol = order["volume"]
            price = order.get("price", 0)
            amount = order.get("amount", 0)
            parts = vt.split(".")
            xt_code = f"{parts[0]}.{'SH' if parts[1] == 'SSE' else 'SZ'}"

            print(f"  {xt_code}  {vol:>6d} 股  ¥{amount:>10,.0f}  {order['reason']}")
            total_buy += 1

            if not dry_run:
                gateway.trader.order_stock(
                    account, xt_code, 23,  # STOCK_BUY
                    int(vol), 0,  # 0 = 市价
                    strategy_name="ALSTM_PPO",
                    order_remark="paper_buy",
                )
                time.sleep(0.2)

    return total_sell, total_buy


def main():
    p = argparse.ArgumentParser(description="虚拟盘执行")
    p.add_argument("--dry-run", action="store_true",
                    help="只计算差异，不实际下单")
    p.add_argument("--capital", type=float, default=500_000)
    args = p.parse_args()

    from qmtquant.config import get_config
    cfg = get_config()

    print("=" * 50)
    print("ALSTM + PPO 虚拟盘执行")
    print("=" * 50)
    if args.dry_run:
        print("** DRY RUN 模式 — 不实际下单 **\n")

    # 1. 加载信号
    target_df = load_target()
    if target_df.empty:
        return 1

    # 2. 连接 miniQMT
    print(f"\n连接 miniQMT...")
    print(f"  路径: {cfg.gateway.qmt_path}")
    print(f"  账号: {cfg.gateway.account_id}")

    from qmtquant.event.engine import EventEngine
    from qmtquant.gateway.xt_gateway import XtGateway

    event_engine = EventEngine()
    event_engine.start()

    gateway = XtGateway(event_engine)
    connected = gateway.connect({
        "qmt_path": cfg.gateway.qmt_path,
        "account_id": cfg.gateway.account_id,
    })

    if not connected:
        print("连接失败！请确认 miniQMT 已启动并登录")
        event_engine.stop()
        return 1

    print("  连接成功 ✓")
    time.sleep(1)

    # 3. 查询当前持仓
    print("\n查询当前持仓...")
    current_pos = get_current_positions(gateway)
    if current_pos:
        print(f"  当前持有 {len(current_pos)} 只:")
        for vt, info in sorted(current_pos.items()):
            print(f"    {vt:>12s}  {info['volume']:>6d} 股  "
                  f"可卖 {info['available']:>6d}  "
                  f"市值 ¥{info['market_value']:>10,.0f}")
    else:
        print("  当前空仓")

    # 4. 获取最新价格
    print("\n获取最新价格...")
    all_symbols = list(set(
        target_df["vt_symbol"].tolist() +
        list(current_pos.keys())
    ))
    prices = get_latest_prices(all_symbols)
    print(f"  获取到 {len(prices)} 只价格")

    # 5. 计算差异
    print("\n计算调仓差异...")
    sell_orders, buy_orders = calc_orders(
        target_df, current_pos, prices, args.capital)

    if not sell_orders and not buy_orders:
        print("  无需调仓")
        gateway.close()
        event_engine.stop()
        return 0

    # 6. 执行
    mode = "DRY RUN" if args.dry_run else "执行"
    print(f"\n{'='*50}")
    print(f"调仓 {mode}")
    print(f"{'='*50}")

    total_sell, total_buy = execute_orders(
        gateway, sell_orders, buy_orders, dry_run=args.dry_run)

    print(f"\n卖出 {len(sell_orders)} 笔, 买入 {len(buy_orders)} 笔")
    if args.dry_run:
        print("\n(DRY RUN 完成，未实际下单)")

    # 7. 清理
    time.sleep(2)
    gateway.close()
    event_engine.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
