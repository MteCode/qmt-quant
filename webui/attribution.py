"""按策略归因 —— 哪个策略在跑、下了哪些单、赚了多少。

## 为什么单独一层

委托和成交在库里本来就带 `strategy` 列，但此前所有页面都是把它们
平铺展示的：看得到「今天下了 20 笔单」，看不到「其中动量策略 12 笔、
均值回归 8 笔，前者赚 3000 后者亏 800」。

不按策略拆，就无法回答量化系统最基本的问题：**哪个策略在赚钱**。

## 盈亏怎么算

- **已实现盈亏**：按 FIFO 配对同一策略、同一标的的买卖成交。
  A 股 T+1 且不能裸卖空，所以卖出一定对应更早的买入，配对是确定的。
  手续费从成交记录里直接取，不重新估算。
- **浮动盈亏**：剩余持仓 × (现价 - 加权成本)。现价取持仓快照里的
  市值/数量；快照缺失时按成本价算（浮盈为 0），并标注数据不全 ——
  宁可显示「算不出」，也不能拿成本价冒充现价让人以为不赚不亏。

## 一个必须说清的限制

盈亏归因只覆盖**经引擎下的单**。paper_trade.py 直接调 xtquant 下的单
不进 order_log，这里看不到。两条路并存期间，账户层面的总盈亏
以券商快照为准，本页只解释引擎这部分。
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dir_values():
    """买/卖在库里的字面值。取自枚举，不硬编码。"""
    from qmtquant.core.constants import Direction
    return Direction.LONG.value, Direction.SHORT.value


BUY, SELL = _dir_values()


def _store():
    from qmtquant.config import DATA_DIR
    from qmtquant.store.database import StateStore
    return StateStore(DATA_DIR / "state.db")


def _fifo_pnl(trades: list[dict]) -> tuple[float, float, dict]:
    """FIFO 配对算已实现盈亏。

    :return: (已实现盈亏, 手续费合计, {vt_symbol: {"volume","cost"}} 剩余持仓)
    """
    lots: dict[str, deque] = defaultdict(deque)   # symbol -> [(price, vol)]
    realized = 0.0
    fee = 0.0

    for t in sorted(trades, key=lambda x: x.get("datetime") or ""):
        sym = t.get("vt_symbol", "")
        px = float(t.get("price") or 0)
        vol = float(t.get("volume") or 0)
        fee += float(t.get("commission") or 0)
        if vol <= 0:
            continue

        # 用枚举值比对，不硬编码中文字面量 —— 库里存的是
        # Direction.LONG.value，写死一个字符串在枚举文案改动后会静默失配，
        # 表现为所有盈亏都算成 0（买单进不了配对队列）。
        if (t.get("direction") or "") == BUY:        # 买入
            lots[sym].append([px, vol])
            continue

        # 卖出：与最早的买入配对
        left = vol
        while left > 1e-9 and lots[sym]:
            cost_px, cost_vol = lots[sym][0]
            take = min(left, cost_vol)
            realized += (px - cost_px) * take
            cost_vol -= take
            left -= take
            if cost_vol <= 1e-9:
                lots[sym].popleft()
            else:
                lots[sym][0][1] = cost_vol
        # left > 0 说明卖出多于买入 —— 引擎外建的底仓，不计入已实现盈亏

    remain = {}
    for sym, q in lots.items():
        vol = sum(v for _, v in q)
        if vol <= 1e-9:
            continue
        cost = sum(p * v for p, v in q) / vol
        remain[sym] = {"volume": vol, "cost": cost}
    return realized, fee, remain


def _market_prices() -> dict[str, float]:
    """从持仓快照取现价。取不到就返回空 —— 调用方据此标注数据不全。"""
    import json
    out = {}
    for d in (ROOT / "strategies").glob("*/state/positions.json"):
        try:
            snap = json.loads(d.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for h in snap.get("holdings", []):
            vol = float(h.get("volume") or 0)
            mv = float(h.get("market_value") or 0)
            if vol > 0 and mv > 0:
                out[h.get("vt_symbol", "")] = mv / vol
    return out


def configured_strategies() -> list[dict]:
    """config.yaml 里登记的策略 —— 只有登记的才会被引擎加载。"""
    try:
        from qmtquant.config import get_config
        cfg = get_config()
    except Exception:                               # noqa: BLE001
        return []
    out = []
    for item in cfg.strategies or []:
        syms = item.get("vt_symbols", [])
        out.append({
            "name": item.get("name", ""),
            "class": item.get("class", ""),
            "n_symbols": (len(syms) if isinstance(syms, list) else 0),
            "dynamic_symbols": syms == "from_signal" or (
                isinstance(syms, list) and "from_signal" in syms),
            "setting": item.get("setting", {}),
        })
    return out


def by_strategy(day: str | None = None) -> dict:
    """按策略聚合委托、成交与盈亏。

    :param day: 交易日 YYYY-MM-DD；None 表示全部历史
    """
    from qmtquant.core.constants import Status

    try:
        store = _store()
        orders = store.load_orders(day)
        trades = store.load_trades(day)
    except Exception as e:                          # noqa: BLE001
        return {"error": str(e), "rows": [], "configured": [],
                "orders": [], "trades": []}

    prices = _market_prices()
    DONE = Status.ALLTRADED.value
    DEAD = {Status.CANCELLED.value, Status.REJECTED.value}

    o_by: dict[str, list] = defaultdict(list)
    t_by: dict[str, list] = defaultdict(list)
    for o in orders:
        o["untraded"] = (o.get("volume") or 0) - (o.get("traded") or 0)
        o_by[o.get("strategy") or "(未标注)"].append(o)
    for t in trades:
        t_by[t.get("strategy") or "(未标注)"].append(t)

    cfg = {c["name"]: c for c in configured_strategies()}
    names = sorted(set(o_by) | set(t_by) | set(cfg))

    rows = []
    for name in names:
        os_ = o_by.get(name, [])
        ts_ = t_by.get(name, [])
        realized, fee, remain = _fifo_pnl(ts_)

        unrealized = 0.0
        priced = True
        for sym, r in remain.items():
            px = prices.get(sym)
            if px is None:
                priced = False
                continue
            unrealized += (px - r["cost"]) * r["volume"]

        turnover = sum(float(t.get("price") or 0) * float(t.get("volume") or 0)
                       for t in ts_)
        active = [o for o in os_
                  if (o.get("status") or "") not in DEAD | {DONE}]

        rows.append({
            "name": name,
            "configured": name in cfg,
            "cls": cfg.get(name, {}).get("class", ""),
            "n_orders": len(os_),
            "n_active": len(active),
            "untraded": sum(o["untraded"] for o in active),
            "n_filled": sum(1 for o in os_
                            if (o.get("status") or "") == DONE),
            "n_dead": sum(1 for o in os_
                          if (o.get("status") or "") in DEAD),
            "n_trades": len(ts_),
            "turnover": turnover,
            "fee": fee,
            "realized": realized,
            "unrealized": unrealized,
            "total_pnl": realized + unrealized,
            "priced": priced,
            "n_holdings": len(remain),
            "holdings": [
                {"vt_symbol": s, **r,
                 "price": prices.get(s),
                 "pnl": ((prices[s] - r["cost"]) * r["volume"]
                         if s in prices else None)}
                for s, r in sorted(remain.items())
            ],
            "orders": os_,
            "trades": ts_,
        })

    rows.sort(key=lambda r: (-r["n_orders"], r["name"]))
    return {"rows": rows, "configured": list(cfg.values()),
            "orders": orders, "trades": trades, "error": None}


def order_dates(limit: int = 60) -> list[str]:
    """有委托记录的交易日，倒序。"""
    try:
        with _store()._conn() as conn:
            return [r[0] for r in conn.execute(
                "SELECT DISTINCT trade_date FROM order_log "
                "ORDER BY trade_date DESC LIMIT ?", (limit,)).fetchall()]
    except Exception:                               # noqa: BLE001
        return []
