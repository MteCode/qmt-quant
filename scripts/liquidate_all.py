"""清盘脚本 —— 市价卖出全部持仓。

用法：
    # 预览（不实际下单）
    python scripts/liquidate_all.py --dry-run

    # 实际执行（开盘后运行）
    python scripts/liquidate_all.py

    # 强制执行（今日已下过单时仍继续）
    python scripts/liquidate_all.py --force
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXEC_DIR = ROOT / "strategies" / "alstm_ppo_csi1000" / "executions"
EXEC_COLUMNS = ["time", "run_id", "remark", "vt_symbol", "name", "direction",
                "volume", "price", "amount", "result", "reason", "order_id",
                "mode"]


def get_execution_path(date_str: str) -> Path:
    return EXEC_DIR / f"{date_str}.csv"


def append_record(row: dict, run_id: str) -> None:
    EXEC_DIR.mkdir(parents=True, exist_ok=True)
    path = get_execution_path(datetime.now().strftime("%Y-%m-%d"))
    exists = path.exists()
    try:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=EXEC_COLUMNS)
            if not exists:
                w.writeheader()
            row = {**row, "run_id": run_id}
            w.writerow({k: row.get(k, "") for k in EXEC_COLUMNS})
            f.flush()
    except OSError as e:
        print(f"  [WARN] 执行记录写入失败: {e}")


def today_already_sent() -> list[dict]:
    path = get_execution_path(datetime.now().strftime("%Y-%m-%d"))
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8-sig") as f:
            return [r for r in csv.DictReader(f)
                    if r.get("mode") == "实盘"
                    and r.get("result") in ("已委托", "已提交")]
    except OSError:
        return []


def load_names() -> dict:
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


def main():
    parser = argparse.ArgumentParser(description="市价清盘 —— 卖出全部持仓")
    parser.add_argument("--dry-run", action="store_true",
                        help="只查看持仓，不实际下单")
    parser.add_argument("--force", action="store_true",
                        help="今日已下过单时仍继续执行")
    args = parser.parse_args()

    from qmtquant.config import get_config
    cfg = get_config()

    print("=" * 50)
    print("全持仓清盘 —— 市价卖出")
    print("=" * 50)
    if args.dry_run:
        print("** DRY RUN 模式 — 不实际下单 **\n")

    # 幂等闸门
    if not args.dry_run:
        sent = today_already_sent()
        if sent and not args.force:
            print(f"[拒绝执行] 今日已发出 {len(sent)} 笔委托，"
                  f"重复运行会造成重复卖出。")
            print("  确认要追加请加 --force；只想看持仓请加 --dry-run")
            return 1
        if sent and args.force:
            print(f"[警告] 今日已发出 {len(sent)} 笔委托，--force 继续\n")

    # 连接 miniQMT
    print(f"连接 miniQMT...")
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

    # 查询持仓
    print("\n查询当前持仓...")
    from xtquant.xttype import StockAccount
    account = StockAccount(gateway.account_id)
    gateway.query_position()
    time.sleep(1)

    pos_list = gateway.trader.query_stock_positions(account)
    positions = []
    if pos_list:
        for pos in pos_list:
            if pos.volume <= 0:
                continue
            code = pos.stock_code
            parts = code.split(".")
            vt = f"{parts[0]}.SSE" if parts[1] == "SH" else f"{parts[0]}.SZSE"
            positions.append({
                "xt_code": code,
                "vt_symbol": vt,
                "volume": pos.volume,
                "available": pos.can_use_volume,
                "avg_price": pos.avg_price,
                "market_value": pos.market_value,
            })

    if not positions:
        print("  当前空仓，无需清盘")
        event_engine.stop()
        return 0

    names = load_names()
    total_value = sum(p["market_value"] for p in positions)
    total_available = sum(p["available"] for p in positions)

    print(f"  持有 {len(positions)} 只，总市值 {total_value:,.0f} 元")
    print()
    print(f"  {'标的':>14s}  {'名称':>8s}  {'总量':>6s}  {'可卖':>6s}  {'市值':>10s}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*10}")
    for p in sorted(positions, key=lambda x: -x["market_value"]):
        name = names.get(p["vt_symbol"], "")
        print(f"  {p['vt_symbol']:>14s}  {name:>8s}  "
              f"{p['volume']:>6d}  {p['available']:>6d}  "
              f"{p['market_value']:>10,.0f}")

    # 筛选可卖出的
    sellable = [p for p in positions if p["available"] > 0]
    frozen = [p for p in positions if p["available"] <= 0]

    if frozen:
        print(f"\n  [注意] {len(frozen)} 只股票可卖数量为 0（T+1 冻结），无法卖出:")
        for p in frozen:
            name = names.get(p["vt_symbol"], "")
            print(f"    {p['vt_symbol']} {name}  "
                  f"持有 {p['volume']} 股，可卖 0")

    if not sellable:
        print("\n  没有可卖出的持仓（全部被 T+1 冻结）")
        event_engine.stop()
        return 0

    sell_value = sum(p["available"] * p["avg_price"] for p in sellable)
    print(f"\n  可卖出 {len(sellable)} 只，"
          f"预计卖出金额 ~{sell_value:,.0f} 元")

    # 执行卖出
    run_id = f"LIQ_{datetime.now().strftime('%H%M%S')}"
    mode = "预览" if args.dry_run else "实盘"
    passed = 0

    print(f"\n--- 清盘卖出 ({len(sellable)} 笔) ---")
    for seq, p in enumerate(sellable, 1):
        vt = p["vt_symbol"]
        vol = p["available"]
        xt_code = p["xt_code"]
        price = p["avg_price"]
        name = names.get(vt, "")
        remark = f"{run_id}_{seq:03d}"

        rec = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "remark": remark,
            "vt_symbol": vt, "name": name,
            "direction": "清盘卖", "volume": vol,
            "price": round(price, 3),
            "amount": round(vol * price, 2),
            "reason": "全持仓清盘",
            "mode": mode,
        }

        print(f"  {xt_code}  {name:>6s}  {vol:>6d} 股  "
              f"~{vol * price:>10,.0f} 元")

        if args.dry_run:
            rec.update(result="已预览", order_id="")
            append_record(rec, run_id)
            passed += 1
            continue

        # 先记录再发单（宁可记了没发，不可发了没记）
        rec.update(result="已提交", order_id="")
        append_record(rec, run_id)

        # xt_order_type=24 卖出, price_type=5 最优五档即时成交, price=0
        order_id = gateway.trader.order_stock(
            account, xt_code, 24,
            vol, 5, 0,
            "LIQUIDATE", remark,
        )
        time.sleep(0.2)

        append_record({
            **rec,
            "time": datetime.now().strftime("%H:%M:%S"),
            "result": "已委托",
            "order_id": order_id or "",
        }, run_id)
        passed += 1

    print(f"\n{'预览' if args.dry_run else '提交'}完成: "
          f"{passed}/{len(sellable)} 笔")

    exec_path = get_execution_path(datetime.now().strftime("%Y-%m-%d"))
    print(f"执行记录: {exec_path}")

    event_engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
