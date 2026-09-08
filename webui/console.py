"""交易台数据层 —— 把策略的回测、实盘、服务状态聚成一个视图。

## 为什么再加一层

`strategies.py` 管回测结果、`attribution.py` 管实盘归因、`services.py`
管进程状态 —— 三者互不知道对方。页面要同时显示「这个策略回测多少、
现在跑没跑、实盘赚了多少」，就得在某处把它们对起来。

对起来的难点是**同一个策略在三处的标识不一样**：
注册表用 `id`（intraday_gbm），实盘委托用引擎里的策略名
（IntradayGBM，来自 config.yaml 的 name），两者不必相同。
这里用 `live_name` 显式声明映射，而不是靠猜 —— 猜错的表现是
「策略明明在跑，页面上显示没跑」。
"""
from __future__ import annotations

from . import attribution, services
from . import strategies as strat


def _config_names() -> dict[str, str]:
    """config.yaml 里登记的策略名 -> 类路径。

    用于把注册表里的策略与引擎里的运行实例对上。
    """
    out = {}
    try:
        from qmtquant.config import get_config
        for item in get_config().strategies or []:
            name = item.get("name", "")
            cls = item.get("class", "")
            if name:
                out[name] = cls
    except Exception:                               # noqa: BLE001
        pass
    return out


def _match_live_name(s, cfg_names: dict[str, str],
                     live_rows: dict[str, dict]) -> str | None:
    """给注册表里的策略找到它在实盘中的名字。

    三级匹配，从最可靠到最勉强：
    1. 策略显式声明了 live_name
    2. config.yaml 里有条目的 class 指向同一个策略类
    3. 实盘归因里有同名记录

    匹配不上就返回 None —— 显示「未接入实盘」，
    比错误地把别的策略的盈亏算到它头上要好。
    """
    explicit = getattr(s, "live_name", "")
    if explicit:
        return explicit

    if s.code:
        # code 形如 strategies/intraday_gbm/strategy.py，
        # 类路径形如 strategies.intraday_gbm.strategy.IntradayGBMStrategy
        stem = s.code.replace("/", ".").replace(".py", "")
        for name, cls in cfg_names.items():
            if cls and (cls.startswith(stem) or stem in cls):
                return name

    for name in live_rows:
        if name.lower().replace("_", "") == s.id.lower().replace("_", ""):
            return name
    return None


def overview() -> dict:
    """交易台总览：每个策略一行，含回测、实盘、运行状态。"""
    live = attribution.by_strategy(None)
    live_rows = {r["name"]: r for r in live.get("rows", [])}
    cfg_names = _config_names()

    svc = {x["id"]: x for x in services.list_status()}
    engine_up = any(svc.get(k, {}).get("healthy")
                    for k in ("live_sim", "live_qmt"))
    engine_mode = ("miniQMT" if svc.get("live_qmt", {}).get("running")
                   else ("模拟撮合" if svc.get("live_sim", {}).get("running")
                         else None))

    rows = []
    for s in strat.STRATEGIES:
        r = s.result()
        m = r.get("metrics") or {}
        lname = _match_live_name(s, cfg_names, live_rows)
        lv = live_rows.get(lname) if lname else None

        rows.append({
            "id": s.id,
            "name": s.name,
            "category": s.category,
            "status": s.status,
            "summary": s.summary,
            "caveat": s.caveat,
            "code": s.code,
            # ---- 回测 ----
            "has_backtest": bool(r.get("has_result")),
            "annual_return": m.get("annual_return"),
            "total_return": m.get("total_return"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe": m.get("sharpe"),
            "period": r.get("period"),
            "has_equity": bool(r.get("equity")),
            "n_trades_bt": r.get("n_trades"),
            # ---- 实盘 ----
            "in_config": lname in cfg_names if lname else False,
            "live_name": lname,
            "live_orders": lv["n_orders"] if lv else 0,
            "live_active": lv["n_active"] if lv else 0,
            "live_trades": lv["n_trades"] if lv else 0,
            "live_realized": lv["realized"] if lv else 0.0,
            "live_unrealized": lv["unrealized"] if lv else 0.0,
            "live_pnl": lv["total_pnl"] if lv else 0.0,
            "live_holdings": lv["n_holdings"] if lv else 0,
            # 「正在跑」= 引擎起着 且 这个策略在 config 里
            "running": bool(engine_up and lname and lname in cfg_names),
            "can_run": bool(lname and lname in cfg_names),
        })

    # 有回测结果的排前面，再按年化降序
    rows.sort(key=lambda x: (not x["has_backtest"],
                             -(x["annual_return"] or -99)))
    return {
        "rows": rows,
        "engine_up": engine_up,
        "engine_mode": engine_mode,
        "services": svc,
        "n_running": sum(1 for r in rows if r["running"]),
        "total_live_pnl": sum(r["live_pnl"] for r in rows),
    }


def detail(sid: str) -> dict | None:
    """单个策略的完整数据：回测 + 实盘。"""
    s = strat.BY_ID.get(sid)
    if s is None:
        return None
    ov = overview()
    row = next((r for r in ov["rows"] if r["id"] == sid), None)
    r = s.result()

    live_detail = None
    if row and row["live_name"]:
        data = attribution.by_strategy(None)
        live_detail = next((x for x in data["rows"]
                            if x["name"] == row["live_name"]), None)

    return {"s": s, "r": r, "row": row, "live": live_detail,
            "engine_up": ov["engine_up"], "engine_mode": ov["engine_mode"],
            "services": ov["services"]}
