"""持仓快照 —— 从 miniQMT 拉取账户实际持仓并落盘。

## 为什么要单独做这件事

管理台此前展示的是 `signals/target_latest.csv`，那是**信号**（策略想持有什么），
不是账户里**实际持有**什么。两者经常不一致：整手约束买不满、可用资金不足、
风控拦截、市价单废单，都会让实际偏离目标。

管理台不直接连 miniQMT —— 连接要 QMT 在线、会阻塞请求、且失败时整个页面
打不开。改为由本脚本定期快照到文件，页面只读文件。

## 快照内容

账户资金 + 逐只持仓（数量、可卖、成本价、现价、市值、浮动盈亏），
并与当前信号比对，标出应买入 / 应卖出 / 已匹配。

用法::

    python strategies/alstm_ppo_csi1000/snapshot_positions.py
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

SNAPSHOT_FILE = paths.STATE_DIR / "positions.json"


def load_names() -> dict:
    try:
        import pandas as pd

        from qmtquant.config import get_config
        p = Path(get_config().data.store_dir) / "universe" / "industry.parquet"
        if not p.exists():
            return {}
        df = pd.read_parquet(p)
        return {r.vt_symbol: {"name": r.name, "industry": r.industry}
                for r in df.itertuples()}
    except Exception:
        return {}


def load_target() -> dict:
    """当前信号的目标持仓 vt_symbol -> target_value。"""
    if not paths.LATEST_SIGNAL.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(paths.LATEST_SIGNAL, encoding="utf-8-sig")
        return dict(zip(df["vt_symbol"].astype(str), df["target_value"]))
    except Exception:
        return {}


def main() -> int:
    p = argparse.ArgumentParser(description="持仓快照")
    p.add_argument("--timeout", type=int, default=30)
    args = p.parse_args()

    from qmtquant.config import get_config
    cfg = get_config()

    print("=" * 54)
    print("持仓快照")
    print(f"  账号 {cfg.gateway.account_id}")
    print("=" * 54)

    from qmtquant.event.engine import EventEngine
    from qmtquant.gateway.xt_gateway import XtGateway

    ee = EventEngine()
    ee.start()
    gateway = XtGateway(ee)
    if not gateway.connect({"qmt_path": cfg.gateway.qmt_path,
                            "account_id": cfg.gateway.account_id}):
        print("连接失败 —— 请确认 miniQMT 已启动并登录")
        ee.stop()
        return 1
    print("  连接成功")
    time.sleep(1)

    from xtquant.xttype import StockAccount
    acct = StockAccount(cfg.gateway.account_id)

    asset = gateway.trader.query_stock_asset(acct)
    if asset is None:
        print("查询账户资产失败")
        gateway.close(); ee.stop()
        return 1

    pos_list = gateway.trader.query_stock_positions(acct) or []
    names = load_names()
    target = load_target()

    holdings = []
    for pos in pos_list:
        if pos.volume <= 0:
            continue
        code, ex = pos.stock_code.split(".")
        vt = f"{code}.{'SSE' if ex == 'SH' else 'SZSE'}"
        meta = names.get(vt, {})
        cost = pos.avg_price * pos.volume
        holdings.append({
            "vt_symbol": vt, "code": code,
            "name": meta.get("name", "—"),
            "industry": meta.get("industry", ""),
            "volume": int(pos.volume),
            "available": int(pos.can_use_volume),
            "avg_price": round(float(pos.avg_price), 3),
            "market_value": round(float(pos.market_value), 2),
            "pnl": round(float(pos.market_value) - cost, 2),
            "pnl_pct": round((float(pos.market_value) / cost - 1), 4)
            if cost else 0.0,
            "in_target": vt in target,
        })
    holdings.sort(key=lambda h: -h["market_value"])

    held = {h["vt_symbol"] for h in holdings}
    to_buy = [
        {"vt_symbol": vt,
         "code": vt.split(".")[0],
         "name": names.get(vt, {}).get("name", "—"),
         "industry": names.get(vt, {}).get("industry", ""),
         "target_value": float(v)}
        for vt, v in target.items() if vt not in held
    ]
    to_buy.sort(key=lambda x: -x["target_value"])

    mv = sum(h["market_value"] for h in holdings)
    snap = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "account_id": cfg.gateway.account_id,
        "total_asset": round(float(asset.total_asset), 2),
        "cash": round(float(asset.cash), 2),
        "frozen": round(float(asset.frozen_cash), 2),
        "market_value": round(mv, 2),
        "position_ratio": round(mv / float(asset.total_asset), 4)
        if asset.total_asset else 0.0,
        "total_pnl": round(sum(h["pnl"] for h in holdings), 2),
        "holdings": holdings,
        "n_holdings": len(holdings),
        # 持有但不在目标中 —— 下次调仓应卖出
        "to_sell": [h for h in holdings if not h["in_target"]],
        # 在目标中但尚未持有 —— 下次调仓应买入
        "to_buy": to_buy,
        "has_target": bool(target),
    }

    paths.ensure_dirs()
    SNAPSHOT_FILE.write_text(json.dumps(snap, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    print(f"\n  总资产 {snap['total_asset']:,.0f} 元"
          f"   可用 {snap['cash']:,.0f}"
          f"   持仓市值 {mv:,.0f}（{snap['position_ratio']:.1%}）")
    print(f"  浮动盈亏 {snap['total_pnl']:+,.0f} 元")
    print(f"\n  持仓 {len(holdings)} 只:")
    for h in holdings[:15]:
        flag = "" if h["in_target"] else "  <- 不在目标中"
        print(f"    {h['name']:<8s} {h['code']}  {h['volume']:>6d} 股  "
              f"市值 {h['market_value']:>10,.0f}  "
              f"盈亏 {h['pnl_pct']:>+7.2%}{flag}")
    if len(holdings) > 15:
        print(f"    …… 余 {len(holdings) - 15} 只")

    if snap["has_target"]:
        print(f"\n  与信号比对: 应卖出 {len(snap['to_sell'])} 只，"
              f"应买入 {len(to_buy)} 只")

    print(f"\n已保存: {SNAPSHOT_FILE}")

    time.sleep(1)
    gateway.close()
    ee.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
