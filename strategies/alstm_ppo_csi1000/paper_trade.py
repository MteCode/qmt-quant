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

    python strategies/alstm_ppo_csi1000/paper_trade.py
    python strategies/alstm_ppo_csi1000/paper_trade.py --dry-run   # 只看差异，不下单
"""
import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import risk_state  # noqa: E402


def load_target() -> pd.DataFrame:
    path = paths.LATEST_SIGNAL
    if not path.exists():
        print(f"信号文件不存在: {path}")
        print(f"请先运行: python {paths.STRATEGY_DIR / 'generate_signal.py'}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"加载目标持仓: {len(df)} 只, 总金额 {df['target_value'].sum():,.0f} 元")
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
            # 价格取不到时退回成本价 —— 风控要用它算委托金额，给 0 会让
            # 单笔金额上限形同虚设
            px = prices.get(vt, 0) or pos.get("avg_price", 0)
            sell_orders.append({
                "vt_symbol": vt,
                "direction": "卖出",
                "volume": pos["available"],
                "price": px,
                "amount": pos["available"] * px,
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
                    "price": price,
                    "amount": sell_vol * price,
                    "reason": f"减仓 {current_shares} -> {current_shares - sell_vol}",
                })

    return sell_orders, buy_orders


def _split_code(vt: str):
    """vt_symbol -> (6位代码, xtquant 代码, Exchange)"""
    from qmtquant.core.constants import Exchange
    code, ex = vt.split(".")
    if ex == "SSE":
        return code, f"{code}.SH", Exchange.SSE
    return code, f"{code}.SZ", Exchange.SZSE


#: 执行记录的列。顺序即 CSV 列序，改动会影响已有文件的可读性
EXEC_COLUMNS = ["time", "vt_symbol", "name", "direction", "volume",
                "price", "amount", "result", "reason", "order_id", "mode"]


def record_executions(rows: list, dry_run: bool) -> Path | None:
    """把本次执行结果追加到当日记录。

    预览与实盘写同一个文件，用 mode 列区分 —— 分开存会让「今天到底做了什么」
    需要看两个地方。被风控拦截的同样记录：事后要能回答「为什么这只没买」。
    """
    if not rows:
        return None
    import csv

    paths.ensure_dirs()
    path = paths.execution_file(datetime.now().strftime("%Y-%m-%d"))
    exists = path.exists()
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=EXEC_COLUMNS)
            if not exists:
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in EXEC_COLUMNS})
    except OSError as e:
        print(f"  [WARN] 执行记录写入失败: {e}")
        return None
    return path


def _load_names() -> dict:
    """vt_symbol -> 股票名。取不到就算了，不影响下单。"""
    try:
        import pandas as pd

        from qmtquant.config import get_config
        p = Path(get_config().data.store_dir) / "universe" / "industry.parquet"
        if not p.exists():
            return {}
        df = pd.read_parquet(p)
        return dict(zip(df["vt_symbol"], df["name"]))
    except Exception:
        return {}


def execute_orders(gateway, sell_orders, buy_orders, risk_mgr=None, dry_run=False):
    """先卖后买，每笔经过风控校验。

    风控在 dry-run 下**同样执行** —— 预览的意义就是看到真实执行时会发生什么。
    只跳过最后的 `order_stock` 调用。若 dry-run 不跑风控，你会看到 10 笔委托，
    实际执行时被风控砍掉 4 笔而毫不知情。
    """
    from qmtquant.core.constants import Direction
    from qmtquant.core.objects import OrderRequest
    from xtquant.xttype import StockAccount
    account = StockAccount(gateway.account_id)
    names = _load_names()
    records = []

    passed_sell = passed_buy = rejected = 0

    def handle(order, direction, xt_order_type, remark, tag):
        nonlocal rejected
        vt = order["vt_symbol"]
        vol = int(order["volume"])
        price = order.get("price", 0)
        amount = order.get("amount", 0)
        code, xt_code, exchange = _split_code(vt)

        rec = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "vt_symbol": vt, "name": names.get(vt, ""),
            "direction": tag, "volume": vol,
            "price": round(price, 3), "amount": round(amount, 2),
            "mode": "预览" if dry_run else "实盘",
        }

        if risk_mgr is not None:
            ok, reason = risk_mgr.check(OrderRequest(
                symbol=code, exchange=exchange, direction=direction,
                price=price, volume=vol,
            ))
            if not ok:
                print(f"  [拦截] {xt_code} {vol:>6d} 股 -- {reason.value}")
                rejected += 1
                rec.update(result="风控拦截", reason=reason.value)
                records.append(rec)
                return False

        print(f"  {xt_code}  {vol:>6d} 股  {amount:>10,.0f} 元  {order['reason']}")
        order_id = ""
        if not dry_run:
            order_id = gateway.trader.order_stock(
                account, xt_code, xt_order_type,
                vol, 5, 0,  # price_type=5(最优五档即时成交), price=0
                strategy_name="ALSTM_PPO", order_remark=remark,
            )
            time.sleep(0.2)
        rec.update(result="已预览" if dry_run else "已委托",
                   reason=order.get("reason", ""), order_id=order_id or "")
        records.append(rec)
        return True

    if sell_orders:
        print(f"\n--- 卖出 ({len(sell_orders)} 笔) ---")
        for o in sell_orders:
            if handle(o, Direction.SHORT, 24, "paper_sell", "卖"):
                passed_sell += 1

    if buy_orders:
        print(f"\n--- 买入 ({len(buy_orders)} 笔) ---")
        for o in buy_orders:
            if handle(o, Direction.LONG, 23, "paper_buy", "买"):
                passed_buy += 1

    if rejected:
        print(f"\n  风控拦截 {rejected} 笔")

    path = record_executions(records, dry_run)
    if path:
        print(f"  执行记录: {path}")
    return passed_sell, passed_buy


def main():
    p = argparse.ArgumentParser(description="虚拟盘执行")
    p.add_argument("--dry-run", action="store_true",
                    help="只计算差异，不实际下单")
    p.add_argument("--capital", type=float,
                    default=paths.load_params()["capital"])
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

    print("  连接成功")
    time.sleep(1)

    # 3. 查询当前持仓（必须在初始化风控之前 —— 风控校验卖单要用它）
    print("\n查询当前持仓...")
    current_pos = get_current_positions(gateway)
    if current_pos:
        print(f"  当前持有 {len(current_pos)} 只:")
        for vt, info in sorted(current_pos.items()):
            print(f"    {vt:>12s}  {info['volume']:>6d} 股  "
                  f"可卖 {info['available']:>6d}  "
                  f"市值 {info['market_value']:>10,.0f} 元")
    else:
        print("  当前空仓")

    # 3.5 初始化风控
    print("\n初始化风控...")
    from qmtquant.core.constants import Exchange
    from qmtquant.core.objects import AccountData, PositionData
    from qmtquant.risk.risk_manager import RiskManager
    from xtquant.xttype import StockAccount as _SA

    risk_mgr = RiskManager(cfg.risk, event_engine)

    # 回撤状态与盘中监控共用同一份，否则本进程的控制器峰值从 0 开始、
    # 观测点数不足 min_observations，档位判定永远不会触发 ——
    # 会出现「监控说只平不开、下单脚本照样开新仓」的分裂
    prev_state = risk_state.load_state(risk_mgr.drawdown)

    _asset = gateway.trader.query_stock_asset(_SA(cfg.gateway.account_id))
    if not _asset:
        print("  查询账户资产失败，风控无法校验买单，中止")
        gateway.close()
        event_engine.stop()
        return 1

    market_value = sum(p["market_value"] for p in current_pos.values())
    # market_value 必须显式传入：总仓位上限用它计算，缺省 0 会让
    # 「总仓位不超过 95%」的检查以为当前空仓，从而放行超额买入
    risk_mgr.update_account(AccountData(
        accountid=cfg.gateway.account_id,
        balance=_asset.total_asset,
        available=_asset.cash,
        market_value=market_value,
        frozen=_asset.frozen_cash,
    ))
    # 日亏线的基准是**当日开盘**的总资产，不是脚本启动时的。
    # RiskManager 新建时会把它设成当前资产，于是 14:00 启动的脚本
    # 会认为「今天还没亏」，哪怕早盘已经跌了 2.5%
    risk_mgr._day_start_balance = risk_state.resolve_day_start(
        prev_state, _asset.total_asset)
    risk_mgr._check_daily_loss()

    # 持仓必须灌进风控，否则 positions 为空，每一笔卖单都会以
    # 「可卖数量不足」被拒 —— 回撤时连减仓自救都做不到
    for vt, info in current_pos.items():
        code, _, exchange = _split_code(vt)
        risk_mgr.update_position(PositionData(
            symbol=code, exchange=exchange,
            volume=info["volume"],
            frozen=info["volume"] - info["available"],
            price=info["avg_price"],
        ))

    st = risk_mgr.stats()
    day_pnl = ((_asset.total_asset - risk_mgr._day_start_balance)
               / risk_mgr._day_start_balance if risk_mgr._day_start_balance else 0)
    print(f"  账户总资产: {_asset.total_asset:,.0f} 元  "
          f"可用 {_asset.cash:,.0f} 元  持仓市值 {market_value:,.0f} 元")
    print(f"  当日盈亏  : {day_pnl:+.2%}"
          f"（日亏线 -{cfg.risk.daily_loss_limit_ratio:.0%}）")
    print(f"  回撤      : {st['drawdown']:.2%}  档位 {st['drawdown_level']}"
          f"  观测 {risk_mgr.drawdown.state.observations} 天")
    if risk_mgr.close_only:
        print("  [WARN] 已触发只平不开")
    if not risk_mgr.drawdown.allow_open():
        print("  [WARN] 回撤档位禁止开新仓，买单将被全部拦截")

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

    sent_sell, sent_buy = execute_orders(
        gateway, sell_orders, buy_orders,
        risk_mgr=risk_mgr, dry_run=args.dry_run)

    print(f"\n卖出 {sent_sell}/{len(sell_orders)} 笔, "
          f"买入 {sent_buy}/{len(buy_orders)} 笔")
    if args.dry_run:
        print("(DRY RUN — 未实际下单，但风控校验已按真实执行路径跑过)")

    # 7. 记下今日起始资产，供当日后续运行判断日亏线。
    #    dry-run 不写，避免预览污染实盘状态
    if not args.dry_run:
        from datetime import date
        risk_state.save_state(
            risk_mgr.drawdown,
            day_start_balance=risk_mgr._day_start_balance,
            day_start_date=date.today().isoformat())

    # 8. 清理
    time.sleep(2)
    gateway.close()
    event_engine.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
