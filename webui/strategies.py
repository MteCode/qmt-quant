"""策略注册表 —— 把「策略是什么 / 怎么回测 / 结果在哪 / 是否在跑」统一起来。

## 为什么需要这层

在此之前，管理台的页面是按「有什么产物」堆的：有 ensemble_result.json
就做一个页面读它，有 equity.csv 就再做一个。结果是同一个策略的信息散在
三四个页面里，而「这个策略到底是干什么的」根本没地方看。

这里反过来，以**策略**为中心组织：每个策略声明自己的说明、回测入口、
结果位置、实盘信号来源。页面只管渲染，不再各自 hardcode 文件路径。

## 结果格式统一

不同策略的产物格式差异很大（JSON / CSV / 嵌套结构）。每个策略提供一个
loader，把自己的产物翻译成统一的 dict，页面据此渲染同一套图表：

    {"has_result": bool, "metrics": {...}, "equity": {...},
     "period": str, "source": str, "note": str}

缺产物时返回 has_result=False，页面显示「未回测」并给出触发按钮 ——
不编造数字，也不因为缺文件就白屏。

## 实盘状态

`live_status()` 只读快照文件，不连 miniQMT。连接要 QMT 在线、会阻塞请求、
失败时整个页面打不开。没有快照就如实显示「未运行」。
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from qmtquant.core.costs import DEFAULT_COST  # noqa: E402

from . import discovery  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- 工具

def _read_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    try:
        with p.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def _norm_dd(v) -> float | None:
    """回撤统一为负数。

    各产物的符号约定不一致：ensemble_result.json 存正数 0.1706，
    t0 summary 存负数 -0.2674。不在这里统一，页面就得逐处判断，
    迟早有一处漏掉，把 -27% 显示成 +27%。
    """
    if v is None:
        return None
    try:
        return -abs(float(v))
    except (TypeError, ValueError):
        return None



def _cost_drift(recorded: dict | None) -> dict | None:
    """把产物里记的成本模型和当前的比一比。

    研究结果是一次性快照：跑完就固定在那儿，而成本模型会改。
    2026-09 那次修正把印花税从 0.001 改成 0.0005（2023-08-28 起就该是
    这个数）、佣金从万2.5 改成万0.854、并补上了最低佣金与过户费。
    在这之前跑出来的结果，数字全部偏悲观。

    页面把它们和新结果并排显示而不加区分，等于拿两套成本的结论互相比较。
    所以这里返回一个 drift 描述，让页面能标出「这批数字用的是旧成本」。
    返回 None 表示一致或无从判断。
    """
    if not recorded:
        return None
    cur = {
        "commission": DEFAULT_COST.commission_rate,
        "stamp_tax": DEFAULT_COST.stamp_tax_rate,
        "slippage": DEFAULT_COST.slippage_rate,
    }
    diff = {}
    for k, now in cur.items():
        was = recorded.get(k)
        if was is None:
            continue
        try:
            was = float(was)
        except (TypeError, ValueError):
            continue
        if abs(was - now) > 1e-9:
            diff[k] = {"recorded": was, "current": now}
    if not diff:
        return None
    return {
        "fields": diff,
        "note": ("这批结果是用旧成本模型跑的，数字不能与新结果直接比较。"
                 "重跑后才是现行口径。"),
    }


def _num(v):
    """CSV 里的数值可能是空串、'nan'、'inf'。float() 会抛，静默跳过更糟。"""
    if v is None:
        return None
    t = str(v).strip()
    if not t or t.lower() in ("nan", "none", "null", "inf", "-inf", "na"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _col(rows, key):
    return [_num(r.get(key)) for r in rows]


def _count_pos(rows, key):
    return sum(1 for x in _col(rows, key) if x is not None and x > 0)


def _metrics_from_equity(dates: list, values: list[float],
                         capital: float | None = None) -> dict:
    """从净值序列算统一指标。年化按 244 交易日折算。"""
    if len(values) < 2:
        return {}
    init = capital or values[0]
    total = values[-1] / init - 1
    n = len(values)
    peak, mdd = values[0], 0.0
    rets = []
    for i, v in enumerate(values):
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
        if i:
            prev = values[i - 1]
            if prev:
                rets.append(v / prev - 1)
    af = 244 / n if n else 0
    ann = (1 + total) ** af - 1 if total > -1 else -1.0
    if len(rets) > 1:
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
        vol = math.sqrt(var) * math.sqrt(244)
    else:
        vol = 0.0
    return {
        "total_return": total,
        "annual_return": ann,
        "max_drawdown": mdd,
        "sharpe": (ann / vol) if vol > 1e-9 else 0.0,
        "volatility": vol,
        "n_periods": n,
    }


# --------------------------------------------------------------- 数据结构

@dataclass
class Strategy:
    id: str
    name: str
    category: str                 # 日内 / 日频 / 组合 / 研究
    summary: str                  # 一句话说明，列表页显示
    how: list[str] = field(default_factory=list)     # 展开后的工作原理
    inputs: list[str] = field(default_factory=list)  # 依赖的数据
    risk: list[str] = field(default_factory=list)    # 风控约束
    code: str = ""                # 主要代码位置
    #: strategies/ 下的目录名。多数与 id 相同，但 alstm_ensemble 的目录叫
    #: alstm_ppo_csi1000 —— 直接拿 id 拼路径会找不到信号文件，
    #: 表现为「明明跑通了却显示未运行」。
    dir: str = ""
    #: 该策略/实验的产物目录（相对仓库根）。unregistered_outputs() 靠它
    #: 判断磁盘上哪些结果目录还没接进管理台 —— 从 loader 反推做不到，
    #: loader 是个闭包，路径藏在函数体里，扫不出来。
    output_dir: str = ""
    backtest_task: str = ""       # registry.py 中的 task_id
    live_task: str = ""           # 实盘/信号生成的 task_id
    #: services.py 中的常驻服务 id。有它才能从策略页直接启停实盘 ——
    #: 否则用户得自己在「服务」页找对应条目，两个页面割裂
    live_service: str = ""
    status: str = "research"      # research / backtest_only / live_ready
    caveat: str = ""              # 已知问题或结论，红字显示

    #: 返回统一格式的回测结果
    loader: object = None

    def result(self) -> dict:
        if self.loader is None:
            return {"has_result": False}
        try:
            r = self.loader()
        except Exception as e:                      # noqa: BLE001
            return {"has_result": False, "error": str(e)}
        return r or {"has_result": False}


# --------------------------------------------------------------- 各策略 loader

def _load_intraday_gbm() -> dict:
    """试点：产物已标准化为 manifest，读取交给通用 reader。

    原先这里是 51 行专读 models/intraday_gbm/ 的代码。产物统一之后，
    每个策略不再需要自己的 loader —— 其余 11 个按同样方式迁移后，
    这一族函数可以整体删除。
    """
    from . import model_registry
    return model_registry.result_from_runs("intraday_gbm")


def _load_t0_single() -> dict:
    d = ROOT / "models" / "t0_single"
    s = _read_json(d / "600711_summary.json")
    if not s:
        return {"has_result": False}
    pf = s.get("performance", {})
    rows = _read_csv(d / "600711_best_daily.csv")
    eq = {"dates": [r["date"] for r in rows],
          "values": [float(r["equity"]) for r in rows],
          "close": [float(r["close"]) for r in rows]} if rows else None
    return {
        "has_result": True,
        "source": str((d / "600711_summary.json").relative_to(ROOT)),
        "symbol": s.get("symbol"),
        "period": " ~ ".join(s.get("date_range", [])),
        "metrics": {
            "total_return": pf.get("total_return"),
            "annual_return": pf.get("annual_return"),
            "max_drawdown": _norm_dd(pf.get("max_drawdown")),
            "sharpe": pf.get("sharpe"),
        },
        "alpha_beta": {
            "t_annual": pf.get("t_annual"),
            "base_return": pf.get("base_return"),
            "buyhold_annual": pf.get("buyhold_annual"),
        },
        "robustness": s.get("robustness", {}),
        "walkforward": s.get("walkforward", []),
        "cost": s.get("cost_model", {}),
        "cost_drift": _cost_drift(s.get("cost_model")),
        "conclusion": s.get("conclusion", {}),
        "equity": eq,
        "n_trades": pf.get("n_trades"),
        "win_rate": pf.get("win_rate"),
        "generated_at": s.get("generated_at"),
    }


def _load_t0_divergence() -> dict:
    d = ROOT / "models" / "t0_divergence"
    s = _read_json(d / "600711_summary.json")
    if not s:
        return {"has_result": False}
    pf = s.get("performance", {})
    rows = _read_csv(d / "600711_best_daily.csv")
    eq = {"dates": [r["date"] for r in rows],
          "values": [float(r["equity"]) for r in rows],
          "close": [float(r["close"]) for r in rows]} if rows else None
    return {
        "has_result": True,
        "source": str((d / "600711_summary.json").relative_to(ROOT)),
        "symbol": s.get("symbol"),
        "period": " ~ ".join(s.get("date_range", [])),
        "metrics": {
            "total_return": pf.get("total_return"),
            "annual_return": pf.get("annual_return"),
            "max_drawdown": _norm_dd(pf.get("max_drawdown")),
            "sharpe": pf.get("sharpe"),
        },
        "alpha_beta": {
            "t_annual": pf.get("t_annual"),
            "buyhold_annual": pf.get("buyhold_annual"),
        },
        "target": s.get("target_100pct", {}),
        "robustness": s.get("robustness", {}),
        "walkforward": s.get("walkforward", []),
        "best_params": s.get("best_params", {}),
        "cost": s.get("cost_model", {}),
        "cost_drift": _cost_drift(s.get("cost_model")),
        "equity": eq,
        "n_trades": pf.get("n_trades"),
        "coverage": pf.get("coverage"),
        "generated_at": s.get("generated_at"),
    }


def _load_alstm_ensemble() -> dict:
    d = ROOT / "strategies" / "alstm_ppo_csi1000" / "backtest"
    ens = _read_json(d / "ensemble_result.json")
    if not ens or not ens.get("ensemble"):
        return {"has_result": False}
    e = ens["ensemble"]
    cfg = ens.get("config", {})
    return {
        "has_result": True,
        "source": str((d / "ensemble_result.json").relative_to(ROOT)),
        "period": " ~ ".join(cfg.get("test", [])),
        "capital": cfg.get("capital"),
        "metrics": {
            "total_return": e.get("total_return"),
            "annual_return": e.get("annual_return"),
            "max_drawdown": _norm_dd(e.get("max_drawdown")),
            "sharpe": e.get("sharpe"),
        },
        "n_trades": e.get("total_trades"),
        "seeds": ens.get("singles", []),
        "config": cfg,
    }


def _load_lgb_agents() -> dict:
    d = ROOT / "strategies" / "lgb_agents_ppo" / "backtest"
    rob = _read_json(d / "robustness.json")
    siz = _read_json(d / "sizing.json")
    if not rob and not siz:
        return {"has_result": False}
    out = {"has_result": True,
           "source": str(d.relative_to(ROOT)),
           "robustness_raw": rob, "sizing_raw": siz}
    # robustness.json 的结构随实验而变，尽量抽取通用指标
    src = None
    for cand in (rob, siz):
        if isinstance(cand, dict):
            if any(k in cand for k in
                   ("annual_return", "total_return", "sharpe")):
                src = cand
                break
            for v in cand.values():
                if isinstance(v, dict) and "annual_return" in v:
                    src = v
                    break
            if src:
                break
    if src:
        out["metrics"] = {
            "total_return": src.get("total_return"),
            "annual_return": src.get("annual_return"),
            "max_drawdown": _norm_dd(src.get("max_drawdown")),
            "sharpe": src.get("sharpe"),
        }
        out["period"] = src.get("period", "")
        out["n_trades"] = src.get("total_trades") or src.get("n_trades")
    return out


def _load_report(name: str, capital: float = 1_000_000):
    """从 reports/<name>/equity.csv 读净值并算指标。"""
    def _fn():
        p = ROOT / "reports" / name / "equity.csv"
        rows = _read_csv(p)
        if not rows:
            return {"has_result": False}
        dcol = "date" if "date" in rows[0] else list(rows[0])[0]
        vcol = next((c for c in ("equity", "total", "value", "nav")
                     if c in rows[0]), list(rows[0])[-1])
        dates, vals = [], []
        for r in rows:
            try:
                vals.append(float(r[vcol]))
                dates.append(r[dcol][:10])
            except (ValueError, KeyError, TypeError):
                continue
        if len(vals) < 2:
            return {"has_result": False}
        return {
            "has_result": True,
            "source": str(p.relative_to(ROOT)),
            "period": f"{dates[0]} ~ {dates[-1]}",
            "capital": capital,
            "metrics": _metrics_from_equity(dates, vals, capital),
            "equity": {"dates": dates, "values": vals},
        }
    return _fn


def _load_intraday_920368() -> dict:
    d = ROOT / "strategies" / "intraday_t_920368"
    s = _read_json(d / "backtest" / "summary.json")
    if not s:
        return {"has_result": False}
    rows = _read_csv(d / "backtest" / "equity.csv")
    eq = None
    if rows:
        dcol = list(rows[0])[0]
        vcol = next((c for c in ("equity", "total", "value")
                     if c in rows[0]), list(rows[0])[-1])
        try:
            eq = {"dates": [r[dcol][:10] for r in rows],
                  "values": [float(r[vcol]) for r in rows]}
        except (ValueError, KeyError):
            eq = None
    return {
        "has_result": True,
        "source": str((d / "backtest" / "summary.json").relative_to(ROOT)),
        "metrics": {
            "total_return": s.get("total_return"),
            "annual_return": s.get("annual_return"),
            "max_drawdown": _norm_dd(s.get("max_drawdown")),
            "sharpe": s.get("sharpe"),
        },
        "n_trades": s.get("n_trades") or s.get("trades"),
        "period": s.get("period", ""),
        "equity": eq,
        "raw": s,
    }



def _load_t0_constrained() -> dict:
    """回撤约束下的做 T 搜索 —— 底仓规模和策略参数一起搜。

    口径提醒（这两处最容易读反）：
      - annual_return / max_drawdown / sharpe 是**整个账户**（底仓市值 +
        现金）的，底仓跟着股票涨的钱全算在里面。判断做 T 有没有 alpha
        只能看 t_annual —— 看前者会把 beta 记成做 T 的战绩。
      - max_drawdown 产物里已是负数，_norm_dd 只做兜底。

    metrics 给约束内最好的一组（上限），feasible 给可行域的中位数。
    只报前者会让人把上限当成典型表现。
    """
    d = ROOT / "models" / "t0_constrained"
    s = _read_json(d / "summary.json")
    if not s:
        return {"has_result": False}

    best = s.get("best_under_constraint") or {}
    bh = s.get("best_buyhold_under_constraint") or {}
    limit = s.get("max_drawdown_limit")

    feasible = {}
    rows = _read_csv(d / "grid.csv")
    if rows and limit is not None:
        ok = []
        for r in rows:
            dd = _num(r.get("max_drawdown"))
            if dd is not None and abs(dd) <= abs(float(limit)):
                ok.append(r)
        if ok:
            feasible = {
                "n": len(ok),
                "annual_median": _median(_col(ok, "annual_return")),
                "t_annual_median": _median(_col(ok, "t_annual")),
                "max_drawdown_median": _norm_dd(
                    _median(_col(ok, "max_drawdown"))),
                "gross_edge_median": _median(_col(ok, "gross_edge")),
                "n_annual_positive": _count_pos(ok, "annual_return"),
                "n_t_positive": _count_pos(ok, "t_annual"),
            }

    return {
        "has_result": True,
        "source": str((d / "summary.json").relative_to(ROOT)),
        "symbol": s.get("symbol"),
        "period": " ~ ".join(s.get("date_range") or []),
        "n_days": s.get("n_days"),
        "capital": s.get("total_capital"),
        "drawdown_limit": _norm_dd(limit),
        "metrics": {
            "total_return": best.get("total_return"),
            "annual_return": best.get("annual_return"),
            "max_drawdown": _norm_dd(best.get("max_drawdown")),
            "sharpe": best.get("sharpe"),
        },
        "alpha_beta": {
            "t_annual": best.get("t_annual"),
            "t_return": best.get("t_return"),
            "buyhold_annual": bh.get("annual_return"),
        },
        "feasible": feasible,
        "n_trades": int(best["n_trades"]) if best.get("n_trades") else None,
        "search": {
            "n_combos": s.get("n_combos"),
            "n_satisfying": s.get("n_satisfying"),
            "n_positive_t": s.get("n_positive_t"),
            "t_annual_mean": s.get("t_annual_mean"),
            "t_vs_buyhold": s.get("t_vs_buyhold"),
        },
        "stock_max_drawdown": _norm_dd(s.get("stock_max_drawdown")),
        "max_base_for_constraint": s.get("max_base_for_constraint"),
        "cost": s.get("cost_model", {}),
        "cost_drift": _cost_drift(s.get("cost_model")),
        "equity": None,          # 这个实验不产净值曲线，不硬造
        "generated_at": s.get("generated_at"),
    }


def _load_t0_analysis() -> dict:
    """盈利日归因 + 门控对照 + 与大盘/行业的关系。

    没有净值曲线，也没有「年化收益」这类指标 —— 它是一份分析，
    不是一次回测。所以整个 metrics 不给，页面显示「无回测指标」，
    而不是把 None 渲染成 0。
    """
    d = ROOT / "models" / "t0_analysis"
    a = _read_json(d / "deep_analysis.json")
    if not a:
        return {"has_result": False}

    w = a.get("winning_days") or {}
    mk = a.get("market") or {}
    ind = a.get("industry") or {}
    gates = (a.get("gating") or {}).get("paired") or []

    # per_trade_ratio > 1 = 门控后每笔反而亏得更多，即门控只减少了交易
    # 次数而没有改善交易质量。这是判断门控有没有用的关键列。
    worse = 0
    for g in gates:
        v = g.get("per_trade_ratio")
        if v is not None and v > 1:
            worse += 1

    return {
        "has_result": True,
        "source": str((d / "deep_analysis.json").relative_to(ROOT)),
        "symbol": a.get("symbol"),
        "kind": "analysis",       # 分析型产物：没有 metrics / equity
        "equity": None,
        "winning_days": {
            "n_days": w.get("n_days"),
            "win_days": w.get("win_days"),
            "win_rate": w.get("win_rate"),
            "total_t_pnl": w.get("total_t_pnl"),
            "top7_share_of_gains": w.get("top7_share_of_gains"),
            "features": w.get("features") or [],
        },
        "gating": {
            "n_gates": len(gates),
            "n_worse_per_trade": worse,
            "rows": gates,
        },
        "market": {
            "corr": mk.get("corr"), "r2": mk.get("r2"),
            "beta": mk.get("beta"),
            "same_direction_pct": mk.get("same_direction_pct"),
            "n_days": mk.get("n_days"),
            "buckets": mk.get("buckets") or [],
        },
        "industry": {
            "industry": ind.get("industry"), "name": ind.get("name"),
            "corr": ind.get("corr"), "r2": ind.get("r2"),
            "beta": ind.get("beta"),
            "n_peers": ind.get("n_peers"), "n_used": ind.get("n_used"),
            "n_days": ind.get("n_days"),
        },
        "generated_at": a.get("generated_at"),
    }


def _load_t0_downday() -> dict:
    """「只在下跌日做反 T」这条假设的检验。

    lookahead 这一列是关键：用当日实际涨跌决定要不要做，是**未来函数**，
    只能作上界参考。实时可实现的只有 lookahead=0 那批。
    两者混在一起取最优，会得到一个漂亮但做不到的结论。
    """
    d = ROOT / "models" / "t0_downday"
    rows = _read_csv(d / "variants.csv")
    if not rows:
        return {"has_result": False}

    # lookahead 这一列是 pandas 写出来的布尔值，落成字符串 "True"/"False"
    # 而不是 0/1。用 _num() 解析它会全部返回 None，于是 1440 个实时可实现
    # 的变体被整批误判成含未来函数 —— 代表值直接算不出来（页面显示 —），
    # 而如果反过来判错方向，就会拿前视结果当可实现结果展示。
    def _is_lookahead(v) -> bool:
        t = str(v).strip().lower()
        if t in ("true", "1", "yes"):
            return True
        if t in ("false", "0", "no", "", "none", "nan"):
            return False
        n = _num(v)
        return bool(n) if n is not None else False

    live, look = [], []
    for r in rows:
        (look if _is_lookahead(r.get("lookahead")) else live).append(r)

    def _stats(rs):
        if not rs:
            return {}
        best = [x for x in _col(rs, "t_annual") if x is not None]
        return {
            "n": len(rs),
            "annual_median": _median(_col(rs, "annual_return")),
            "t_annual_median": _median(_col(rs, "t_annual")),
            "t_annual_best": max(best) if best else None,
            "max_drawdown_median": _norm_dd(
                _median(_col(rs, "max_drawdown"))),
            "gross_edge_median": _median(_col(rs, "gross_edge")),
            "n_t_positive": _count_pos(rs, "t_annual"),
        }

    lv = _stats(live)
    return {
        "has_result": True,
        "source": str((d / "variants.csv").relative_to(ROOT)),
        "kind": "experiment",
        "equity": None,
        # 代表值取**实时可实现**那批的中位数，不取全体最优
        "metrics": {
            "total_return": None,
            "annual_return": lv.get("annual_median"),
            "max_drawdown": lv.get("max_drawdown_median"),
            "sharpe": None,
        },
        "alpha_beta": {
            "t_annual": lv.get("t_annual_median"),
            "buyhold_annual": _median(_col(rows, "buyhold_annual")),
        },
        "realtime": lv,
        "lookahead": _stats(look),
        "n_variants": len(rows),
        "note": ("代表值是实时可实现变体（lookahead=0）的中位数。"
                 "含未来函数的变体单列，只作上界参考。"),
        "cost_drift": None,      # 该产物未记录 cost_model
    }


def _load_t0_market() -> dict:
    """市场状态门控的对照实验：只在特定行情下做 T 是否更好。

    判断门控有没有用不能看总年化 —— 门控减少交易次数，总年化会跟着
    底仓 beta 走，看起来像「改善」。要看的是单笔质量。
    """
    d = ROOT / "models" / "t0_market"
    rows = _read_csv(d / "gating_experiment.csv")
    if not rows:
        return {"has_result": False}

    by_gate: dict[str, list] = {}
    for r in rows:
        by_gate.setdefault(str(r.get("gate") or "?"), []).append(r)

    gates = []
    for g, rs in sorted(by_gate.items()):
        gates.append({
            "gate": g, "n": len(rs),
            "t_annual_median": _median(_col(rs, "t_annual")),
            "annual_median": _median(_col(rs, "annual_return")),
            "gross_edge_median": _median(_col(rs, "gross_edge")),
            "n_trades_median": _median(_col(rs, "n_trades")),
            "n_t_positive": _count_pos(rs, "t_annual"),
        })

    base = None
    for g in gates:
        if g["gate"] in ("none", "无", "", "baseline"):
            base = g
            break

    return {
        "has_result": True,
        "source": str((d / "gating_experiment.csv").relative_to(ROOT)),
        "kind": "experiment",
        "equity": None,
        "metrics": {
            "total_return": None,
            "annual_return": _median(_col(rows, "annual_return")),
            "max_drawdown": _norm_dd(_median(_col(rows, "max_drawdown"))),
            "sharpe": None,
        },
        "alpha_beta": {
            "t_annual": _median(_col(rows, "t_annual")),
            "buyhold_annual": _median(_col(rows, "buyhold_annual")),
        },
        "gates": gates,
        "baseline": base,
        "n_runs": len(rows),
        "n_t_positive": _count_pos(rows, "t_annual"),
        "note": ("门控是否有用要看单笔质量，不是总年化 —— 门控减少交易"
                 "次数，总年化会跟着底仓 beta 走，看起来像改善。"),
        "cost_drift": None,      # 该产物未记录 cost_model
    }


def _load_t0_allocation() -> dict:
    """仓位与资金配置搜索：底仓多少、留多少现金。

    这一维不改变 edge，只改变风险敞口。真正有信息量的是
    buyhold_frontier（纯持有在各仓位下的风险收益），
    以及做 T 能不能在同样回撤下胜过它。
    """
    d = ROOT / "models" / "allocation"
    s = _read_json(d / "summary.json")
    if not s:
        return {"has_result": False}

    st = s.get("stock") or {}
    front = s.get("frontier") or []
    t_rows = [f for f in front if f.get("kind") == "做T"]
    best_t = None
    if t_rows:
        best_t = max(t_rows, key=lambda f: f.get("annual_return") or -9)

    return {
        "has_result": True,
        "source": str((d / "summary.json").relative_to(ROOT)),
        "symbol": s.get("symbol"),
        "period": " ~ ".join(s.get("date_range") or []),
        "n_days": s.get("n_days"),
        "capital": s.get("capital"),
        "equity": None,
        # 这是一次 7776 组的**搜索**，不是一次回测 —— 没有「这一组」的
        # 年化。拿最优组当代表值，会在列表卡片上显示成绿色的 +17.8%，
        # 而那既是全域最大值、又含底仓 beta，和「做 T 无 alpha」的结论
        # 正好相反。所以账户口径留空，代表值用做 T 的全域均值。
        "metrics": {
            "total_return": None,
            "annual_return": None,
            "max_drawdown": _norm_dd(st.get("max_drawdown")),
            "sharpe": None,
        },
        "alpha_beta": {
            "t_annual": s.get("t_annual_mean"),
            "buyhold_annual": st.get("annual_return"),
        },
        # 最优组单独列出，标明它是上限而不是典型值
        "best_t": best_t,
        "stock": {
            "annual_return": st.get("annual_return"),
            "max_drawdown": _norm_dd(st.get("max_drawdown")),
            "volatility": st.get("volatility"),
        },
        "buyhold_frontier": s.get("buyhold_frontier") or [],
        "frontier": front,
        "search": {
            "n_combos": s.get("n_combos"),
            "n_positive_t": s.get("n_positive_t"),
            "pct_positive_t": s.get("pct_positive_t"),
            "t_annual_mean": s.get("t_annual_mean"),
            "t_wins": s.get("t_wins"),
        },
        "cost": s.get("cost_model", {}),
        "cost_drift": _cost_drift(s.get("cost_model")),
        "generated_at": s.get("generated_at"),
    }


# --------------------------------------------------------------- 注册表

#: 手写的核心清单。**保留**是刻意的 —— 这 13 条里只有 4 条指向
#: strategies/<目录>/，另外 9 条（t0_*、qlib_ml…）的产物散在 models/ 与
#: reports/，**没有 manifest 可供发现**。若改成"用发现结果替换本清单"，
#: 会静默丢掉这 9 个能用的页面。发现结果只做**追加**（见 _merge_discovered）。
_CORE: list[Strategy] = [
    Strategy(
        id="intraday_gbm",
        name="全市场日内 GBM 选股",
        category="日内",
        summary="用 1 分钟 K 线训练横截面模型，盘中对全市场打分选股，"
                "当日买入、次一交易日卖出（A 股 T+1）。",
        how=[
            "读取全市场 1m 清洗数据（已排除北交所、科创板、股价 >500 元的标的）",
            "计算 23 维日内特征：多周期动量、均线偏离、量能 z-score、ATR、"
            "RSI、VWAP 偏离、订单不平衡、Amihud 非流动性、日内位置、时间编码",
            "标签为「当日买入、次日开盘卖出的收益是否超过阈值」，LightGBM 二分类",
            "按日期 70/30 分割，训练集与验证集不重叠，避免前视",
            "盘中对全市场打分排序，按概率选前 N 名买入；持仓次日才可卖",
        ],
        inputs=["data/clean/1m/ 全市场分钟线"],
        risk=["单票止损（T+1 下顺延至次日执行）",
              "组合回撤分档停止开仓",
              "持仓次日才可卖，隔夜跳空风险无法规避"],
        caveat="2026-09-17 前的回测未加 T+1 约束，同日买卖回合在 A 股无法执行，"
               "历史上那份 +222%/夏普 31.97 的结果不可引用。"
               "当前口径：夏普 1.50、年化 27.11%、回撤 -11.53%（prob_buy=0.40）。",
        code="strategies/intraday_gbm/strategy.py",
        output_dir="models/intraday_gbm",
        backtest_task="backtest_intraday_gbm",
        live_task="run_intraday_gbm",
        live_service="intraday_gbm",
        status="live_ready",
        loader=_load_intraday_gbm,
    ),
    Strategy(
        id="t0_single",
        name="单票做 T（VWAP 偏离）",
        category="日内",
        summary="底仓 + 现金做 T+0 回转，以 VWAP 偏离为信号。"
                "已验证：样本外无 alpha。",
        how=[
            "严格实现 T+1 约束：区分 base_shares（可卖）与 locked_shares（今日买入锁定）",
            "正T = 低点用现金买入、高点从底仓卖出；反T = 高点先卖底仓、低点买回",
            "信号为价格相对日内累计 VWAP 的偏离幅度",
            "止盈/止损/尾盘强平三重出场",
            "内置样本外验证：前半段选参数，后半段检验",
        ],
        inputs=["单标的 1m 数据"],
        risk=["单笔止损", "最长持有时长", "尾盘强制平仓"],
        code="scripts/optimize_t0_600711.py",
        output_dir="models/t0_single",
        backtest_task="",
        status="research",
        caveat="3600 组参数中仅 5 组做T为正（0.1%），样本内前 10 组样本外无一存活。"
               "毛 edge 0.298% 对成本 0.25%，净 edge 仅 0.048%。不建议实盘。",
        loader=_load_t0_single,
    ),
    Strategy(
        id="t0_divergence",
        name="单票做 T（量价背离）",
        category="日内",
        summary="以量价背离为信号的 T+0 回转，测试 corr / OBV / 量能脉冲三类背离。",
        how=[
            "corr 背离：滚动窗口内价格变化与成交量变化的相关系数显著为负",
            "OBV 背离：能量潮与价格的短期斜率符号相反",
            "量能脉冲：量能 z-score 突增但价格反向走",
            "每类信号都测正向与反向（背离后是反转还是延续）",
            "加入最长持有时长，避免腿死等到尾盘",
        ],
        inputs=["单标的 1m 数据"],
        risk=["止盈止损", "最长持有时长", "尾盘强制平仓"],
        code="scripts/optimize_t0_divergence.py",
        output_dir="models/t0_divergence",
        backtest_task="",
        status="research",
        caveat="7776 组参数中仅 1 组做T为正（0.01%），三类信号全部为负均值。"
               "样本内前 10 组样本外 0/10 存活。毛 edge 0.266% 对成本 0.25%，"
               "距年化 100% 所需的 1.07% 差 4.0 倍。"
               "交易越频繁亏得越多 —— 覆盖率 >50% 的组合做T年化均值 -26.2%。"
               "不建议实盘。",
        loader=_load_t0_divergence,
    ),
    Strategy(
        id="intraday_t_920368",
        name="920368 日内做 T",
        category="日内",
        summary="单票 LightGBM 做 T 模型，9 个基础特征预测短期方向。",
        how=[
            "针对 920368.BSE 单只标的训练",
            "9 维基础特征（动量、量能、波动）",
            "LightGBM 二分类预测未来短期方向",
        ],
        inputs=["920368 1m 数据"],
        risk=["单票止损", "尾盘平仓"],
        code="strategies/intraday_t_920368/strategy.py",
        output_dir="strategies/intraday_t_920368/backtest",
        backtest_task="",
        status="backtest_only",
        loader=_load_intraday_920368,
    ),
    Strategy(
        id="alstm_ensemble",
        name="ALSTM 八模型集成",
        category="日频",
        summary="ALSTM 多 seed 集成选股 + PPO 仓位管理，中证 1000 池。",
        how=[
            "8 个不同 seed 的 ALSTM 模型对全池打分",
            "集成打分排序选股，固定持仓数与调仓周期",
            "PPO 智能体负责仓位管理",
        ],
        inputs=["日线行情", "因子数据"],
        risk=["回撤控制", "单票权重上限"],
        code="strategies/alstm_ppo_csi1000/",
        output_dir="strategies/alstm_ppo_csi1000/backtest",
        dir="alstm_ppo_csi1000",
        backtest_task="train_ensemble",
        live_task="generate_signal",
        status="live_ready",
        loader=_load_alstm_ensemble,
    ),
    Strategy(
        id="lgb_agents_ppo",
        name="四层选股系统",
        category="日频",
        summary="LightGBM 初筛 → TradingAgents 精研 → PPO 仓位 → 风控，"
                "中证 1000 池。已产出实盘信号。",
        how=[
            "初筛层：LightGBM + Qlib 全量因子打分，取 Top-N 候选池（约 100 只）",
            "精研层：TradingAgents 用研报/新闻逻辑验真，排除风险标的"
            "（此层无法历史回测，存在前视问题，见 DESIGN.md）",
            "执行层：PPO 输出动态仓位水平 [0,1]",
            "风控层：低过拟合设计贯穿 1、3 层，叠加回撤三档与下单前置检查",
        ],
        inputs=["data/clean/ 日线行情", "因子数据", "研报/新闻（精研层）"],
        risk=["回撤三档控制", "下单前置检查", "单票权重上限"],
        code="strategies/lgb_agents_ppo/",
        output_dir="strategies/lgb_agents_ppo/backtest",
        backtest_task="",
        live_task="generate_signal",
        status="live_ready",
        caveat="精研层依赖研报/新闻，无法历史回测 —— 该层的贡献无法用回测衡量，"
               "整体回测数据仅覆盖初筛+执行层。",
        loader=_load_lgb_agents,
    ),
    Strategy(
        id="lgb_enhanced",
        name="LightGBM 增强选股",
        category="日频",
        summary="LightGBM 因子模型选股，50 只持仓 / 20 日调仓。",
        how=[
            "多因子输入训练 LightGBM 排序模型",
            "按预测收益排序取前 N 只",
            "固定周期调仓",
        ],
        inputs=["日线行情", "财务数据"],
        risk=["回撤控制"],
        code="scripts/run_portfolio_backtest.py",
        backtest_task="sweep_portfolio",
        status="backtest_only",
        loader=_load_report("lgb_enhanced"),
    ),
    Strategy(
        id="qlib_ml",
        name="Qlib ML 基线",
        category="日频",
        summary="Qlib 框架下的机器学习选股基线，用于对照。",
        how=["Qlib 标准 Alpha158 因子集", "GBDT 模型", "TopK 选股"],
        inputs=["Qlib 格式行情"],
        risk=["标准回撤控制"],
        code="scripts/run_portfolio_backtest.py",
        backtest_task="",
        status="backtest_only",
        loader=_load_report("qlib_ml"),
    ),
    Strategy(
        id="t0_constrained",
        name="做 T · 回撤约束下的联合搜索",
        category="研究",
        summary="在「回撤不超过 15%」这条硬约束下，把底仓规模和策略参数"
                "一起搜。结论是约束内做 T 不如什么都不做。",
        how=[
            "先算约束的物理上限：组合回撤 ≈ 标的自身回撤 × 底仓占比，"
            "600711 自身最大回撤 48.62%，所以底仓超过 6.17 万就不可能"
            "满足 15% —— 这不是调参数能解决的",
            "底仓 7 档（2~10 万）× 3 类量价背离信号 × 参数网格 = 6048 组",
            "每一档底仓另跑一条「纯持有不做 T」基准做同约束对照",
            "账户口径与做 T 口径分开记：annual_return 含底仓 beta，"
            "t_annual 才是做 T 单独的贡献",
        ],
        inputs=["data/clean/1m/600711 分钟线"],
        risk=["回撤约束作为搜索的硬过滤，不满足的组合直接排除"],
        code="scripts/optimize_t0_constrained.py",
        output_dir="models/t0_constrained",
        status="research",
        caveat="否定结论：可行域 484 组里，做 T 年化为正 0 组、账户年化为正 "
               "0 组。最好的一组账户年化仍是 -1.28%，而同约束下什么都不做"
               "（4 万底仓纯持有）是 +2.87% —— 做 T 比不做差 4.15 个百分点。"
               "另：这批数字是用旧成本模型跑的（印花税 0.001、佣金万 2.5），"
               "偏悲观，但方向不会因重跑而反转。",
        loader=_load_t0_constrained,
    ),
    Strategy(
        id="t0_analysis",
        name="做 T · 盈利来源归因",
        category="研究",
        summary="拆开看做 T 的钱到底从哪来：哪些天赚、赚的集中度、"
                "门控有没有用、这只票跟大盘和行业的关系。",
        how=[
            "逐日拆解做 T 盈亏，找出盈利日与亏损日在特征上的差异，"
            "用分布重叠度衡量区分能力 —— 重叠越高说明这个特征越没用",
            "配对差异法做门控对照：同一组参数，加门控与不加门控跑两遍，"
            "比的是同一批交易日，避免样本不同带来的假差异",
            "关键指标是单笔盈亏比而非总收益：门控会减少交易次数，"
            "总收益跟着底仓 beta 变动，容易看成「门控有效」",
            "与上证指数、与申万小金属行业分别做回归，拆 beta 与 R²",
        ],
        inputs=["data/clean/1m/600711", "指数日线", "同行业成分股日线"],
        code="scripts/analyze_t0_deep.py",
        output_dir="models/t0_analysis",
        status="research",
        caveat="这是分析不是回测，没有净值曲线和年化指标。核心发现："
               "241 个交易日里做 T 盈利只有 22 天，最赚的 7 天贡献了全部"
               "盈利日收益的 58.6%；唯一有区分度的特征是「当日涨幅」，"
               "而它是结果不是原因，事前无法知道。",
        loader=_load_t0_analysis,
    ),
    Strategy(
        id="t0_downday",
        name="做 T · 下跌日反 T 假设检验",
        category="研究",
        summary="检验「只在下跌日做反 T」这条常见说法。"
                "结论是实时可实现的版本无效，有效的版本用了未来函数。",
        how=[
            "把「今天是不是下跌日」拆成两种取法：用当日实际涨跌"
            "（lookahead，未来函数，只作上界）与用开盘至今的涨跌"
            "（lookahead=0，实时可实现）",
            "两种取法各跑一遍完整参数网格，共 1728 个变体",
            "只有 lookahead=0 那批能代表真实可执行的结果",
        ],
        inputs=["data/clean/1m/600711 分钟线"],
        code="scripts/test_downday_reverse_t.py",
        output_dir="models/t0_downday",
        status="research",
        caveat="假设被证伪：实时可实现（lookahead=0）的变体里，做 T 年化"
               "中位数为负；只有引入未来函数的版本才转正。"
               "含底仓的总年化看起来回升，那是底仓 beta，不是做 T 的功劳。",
        loader=_load_t0_downday,
    ),
    Strategy(
        id="t0_market",
        name="做 T · 市场状态门控实验",
        category="研究",
        summary="只在特定市场状态下才做 T（大盘涨跌、宽度、波动率），"
                "看能不能筛掉坏交易。结论是门控只减少次数，不提高质量。",
        how=[
            "三类门控条件：市场宽度下限、大盘当日涨幅下限、波动率 z 下限",
            "每类多档阈值 × 全参数网格，共 4608 次模拟",
            "判据是**单笔**盈亏质量而非总收益 —— 门控必然减少交易次数，"
            "总收益会跟着底仓 beta 走，看总数会得出相反结论",
        ],
        inputs=["data/clean/1m/600711", "指数日线", "全市场日线（算宽度）"],
        code="scripts/test_market_gating.py",
        output_dir="models/t0_market",
        status="research",
        caveat="门控无效：所有档位的单笔平均亏损比值都大于 1，"
               "即门控之后每笔反而亏得更多。表面上的「改善」全部来自"
               "交易次数下降带来的底仓 beta 占比上升。",
        loader=_load_t0_market,
    ),
    Strategy(
        id="t0_allocation",
        name="做 T · 仓位与资金配置搜索",
        category="研究",
        summary="底仓放多少、留多少现金。这一维不改变 edge，"
                "只缩放风险敞口 —— 夏普在所有仓位上几乎是平的。",
        how=[
            "先画纯持有的风险收益前沿：仓位 10%~100% 各跑一遍，"
            "得到「什么都不做」在各风险档位上的基准",
            "再对每个回撤档位找做 T 的最优组合，看能否胜过同档的纯持有",
            "7776 组参数 × 多档底仓；成本用修正后的模型"
            "（万 0.854 + 最低 5 元 + 印花税万 5 + 过户费万 0.1）",
        ],
        inputs=["data/clean/1m/600711 分钟线"],
        code="scripts/optimize_allocation.py",
        output_dir="models/allocation",
        status="research",
        caveat="7776 组里做 T 年化为正只有 19 组（0.24%），全体均值 -14.07%。"
               "有 4 组在某个回撤档位上胜过纯持有，但参数几乎相同，"
               "在这个基数上更像噪声 —— 未做样本外验证前不应采信。"
               "夏普在 10%~100% 仓位区间只有 0.20~0.47，且不随仓位改善，"
               "说明仓位只在缩放风险，没有改善风险调整后收益。",
        loader=_load_t0_allocation,
    ),
]

# --------------------------------------------------------------- 发现层

#: 排障开关：设为 0 则退回「与手写清单完全一致」的旧行为
_DISCOVERY_ON = os.environ.get("QMT_WEBUI_DISCOVERY", "1") not in ("0", "false", "False")


def _dig(obj, path: str):
    """按 'a.b.c' 取值。任一段缺失返回 None，不抛异常。"""
    cur = obj
    for part in str(path).split("."):
        if not part:
            continue
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _numeric(v):
    """metrics 里只允许数字或 None —— test_console_registry 守着这条。

    字符串会让模板的 |pct 过滤器吐出「—」，看起来像「没测过」，
    其实是有值但类型错了。"""
    if v is None or isinstance(v, bool):
        return None
    return v if isinstance(v, (int, float)) else None


def _generic_loader(spec, base: Path = ROOT) -> object:
    """按 manifest 的 results 声明读结果。

    只走声明的点路径，绝不猜：文件缺失或字段对不上就返回空态，
    任何异常都吞成空态 —— 编造数字比空白危险得多。

    ``base`` 可覆盖（测试用），默认仓库根。"""
    def load() -> dict:
        try:
            res = spec.result
            if res.kind == "none" or not spec.output_dir or not res.source_json:
                return {"has_result": False}
            data = _read_json(base / spec.output_dir / res.source_json)
            if data is None:
                return {"has_result": False}

            metrics = {k: _numeric(_dig(data, p))
                       for k, p in (res.metrics or {}).items()}
            if res.normalize_drawdown and "max_drawdown" in metrics:
                metrics["max_drawdown"] = _norm_dd(metrics["max_drawdown"])

            period = _dig(data, res.period) if res.period else None
            if isinstance(period, (list, tuple)):
                period = " ~ ".join(str(x) for x in period)
            return {
                "has_result": True,
                "metrics": metrics,
                "period": "" if period is None else str(period),
                "source": f"{spec.output_dir}/{res.source_json}",
                "n_trades": (_numeric(_dig(data, res.n_trades))
                             if res.n_trades else None),
            }
        except Exception as e:                    # noqa: BLE001
            logger.warning("通用 loader 读 %s 失败: %s", spec.id, e)
            return {"has_result": False}
    return load


def _strategy_from_spec(spec) -> Strategy:
    is_code = spec.source in ("builtin", "code")
    # output_dir 只在真的存在时才声明 —— test_console_registry 要求
    # 「声明了就必须存在」，否则一个新策略还没跑出产物就把测试打红。
    out_dir = (spec.output_dir
               if spec.output_dir and (ROOT / spec.output_dir).is_dir() else "")
    return Strategy(
        id=spec.id,
        name=spec.name or spec.id,
        category=spec.category or "未分类",
        summary=spec.summary or "",
        how=list(spec.how), inputs=list(spec.inputs), risk=list(spec.risk),
        code=spec.code,
        dir=spec.dir,
        output_dir=out_dir,
        backtest_task=spec.backtest_task,
        live_task=spec.live_task,
        status=spec.status or "research",
        caveat=spec.caveat,
        # 代码策略没有产物 loader，走空态；项目策略走通用 loader
        loader=None if is_code else _generic_loader(spec),
    )


def _merge_discovered(core: list[Strategy]) -> list[Strategy]:
    """核心清单 + 发现结果，三键去重（核心优先）。

    核心优先是刻意的：那 9 个没有 manifest 的实验必须留在页面上。"""
    if not _DISCOVERY_ON:
        return list(core)
    claimed: set[str] = set()
    for s in core:
        claimed |= discovery.keys(s.id, s.dir, s.output_dir)
    out = list(core)
    try:
        specs = discovery.discovered_specs()
    except Exception as e:                        # noqa: BLE001
        logger.warning("策略发现失败，退回手写清单: %s", e, exc_info=True)
        return out
    for spec in specs:
        if discovery.keys(spec.id, spec.dir, spec.output_dir) & claimed:
            continue
        try:
            out.append(_strategy_from_spec(spec))
        except Exception as e:                    # noqa: BLE001
            logger.warning("构造策略 %s 失败: %s", spec.id, e)
    return out


STRATEGIES: list[Strategy] = _merge_discovered(_CORE)
BY_ID = {s.id: s for s in STRATEGIES}
CATEGORIES = ["日内", "日频", "组合", "研究"] + [
    c for c in sorted({s.category for s in STRATEGIES})
    if c not in ("日内", "日频", "组合", "研究")]



# ----------------------------------------------------- 产物自动发现

#: 只扫结果产物会落地的位置。全量扫 strategies/ 会把模型权重、
#: 信号文件、运行状态一并报出来 —— 那些是运行时产物不是回测结果，
#: 报了只会让人学会忽略这个区块，等于没有。
_OUTPUT_ROOTS = ("models", "strategies/*/backtest", "strategies/*/results")

#: 这些目录名不是结果产物
_SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "cache", "tmp", "logs",
              "models", "signals", "state", "executions", "reconcile",
              "checkpoints", "raw"}

#: 算作「结果」的文件后缀
_RESULT_SUFFIXES = {".json", ".csv", ".parquet", ".html"}


def _registered_dirs() -> set[str]:
    return {s.output_dir.replace("\\", "/").strip("/")
            for s in STRATEGIES if s.output_dir}


def unregistered_outputs() -> list[dict]:
    """磁盘上有产物、但没有任何策略声明它的目录。

    ## 为什么需要这个

    STRATEGIES 是手写列表。跑完一个实验不补注册项，**不会有任何报错** ——
    页面照常渲染，只是少一块。这样积压过 5 个目录（t0_constrained、
    t0_market、t0_analysis、t0_downday、allocation）没人发现，
    直到用户问「今天训练的怎么没同步到后台」。

    ## 为什么只报缺口，不自动生成 loader

    每个实验的字段语义都不一样：有的 max_drawdown 是负数有的是正数，
    有的 annual_return 含底仓 beta 有的不含，有的 lookahead 列是未来函数。
    自动猜会把亏损显示成盈利 —— **猜错比不显示更糟**。
    所以这里只负责喊「这儿有东西没接」，接的时候人来读字段。
    """
    out = []
    reg = _registered_dirs()
    cands: list[Path] = []
    for pat in _OUTPUT_ROOTS:
        if "*" in pat:
            cands.extend(d for d in ROOT.glob(pat) if d.is_dir())
        else:
            base = ROOT / pat
            if base.is_dir():
                cands.extend(d for d in base.iterdir() if d.is_dir())

    for d in sorted(set(cands)):
        if d.name in _SKIP_DIRS:
            continue
        files = [f for f in d.iterdir()
                 if f.is_file() and f.suffix.lower() in _RESULT_SUFFIXES]
        if not files:
            continue
        rel = d.relative_to(ROOT).as_posix()
        # 已注册，或落在某个已注册目录之内/之上 —— 都不算缺口。
        # 少了「之内」这一条，models/intraday_gbm/backtest 会被误报，
        # 而它其实由父目录的 loader 读着。
        if rel in reg:
            continue
        if any(rel.startswith(r + "/") or r.startswith(rel + "/")
               for r in reg):
            continue

        newest = max(f.stat().st_mtime for f in files)
        out.append({
            "dir": rel,
            "n_files": len(files),
            "files": sorted(f.name for f in files)[:8],
            "mtime": _dt.datetime.fromtimestamp(newest)
                     .strftime("%Y-%m-%d %H:%M"),
            "size_kb": round(
                sum(f.stat().st_size for f in files) / 1024, 1),
        })
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


def stale_cost_strategies() -> list[dict]:
    """产物记录的成本模型与当前不一致的策略。

    研究结果是快照，成本模型会改。把两套成本下的结论并排显示
    而不加区分，等于拿不可比的数字互相比较。
    """
    out = []
    for s in STRATEGIES:
        r = s.result()
        drift = r.get("cost_drift")
        if drift:
            out.append({"id": s.id, "name": s.name,
                        "drift": drift, "source": r.get("source")})
    return out


def list_strategies(category: str = "") -> list[dict]:
    """列表页数据：策略元信息 + 结果摘要。"""
    out = []
    for s in STRATEGIES:
        if category and s.category != category:
            continue
        r = s.result()
        out.append({"s": s, "r": r})
    return out


# --------------------------------------------------------------- 实盘

def _signal_status(dirname: str) -> dict:
    """某策略的信号产出情况。

    实盘链路是三段：**生成信号 → 快照持仓 → 下单执行**。
    早先只看 state/positions.json，把「信号已跑通、只是还没下单」误判成
    「实盘从未运行」—— 这三段要分开报，才能看出卡在哪一步。
    """
    d = ROOT / "strategies" / dirname / "signals"
    if not d.exists():
        return {"has": False, "dates": [], "latest": None, "n": 0}
    dates = sorted(
        (p.stem.replace("target_", "") for p in d.glob("target_*.csv")
         if p.stem != "target_latest"), reverse=True)
    latest_p = d / "target_latest.csv"
    rows, mtime = [], None
    if latest_p.exists():
        try:
            with latest_p.open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        except (OSError, csv.Error):
            rows = []
        try:
            import datetime
            mtime = datetime.datetime.fromtimestamp(
                latest_p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            mtime = None
    return {"has": bool(dates or rows), "dates": dates,
            "latest": dates[0] if dates else None,
            "n": len(rows), "rows": rows[:30], "updated_at": mtime,
            "dir": str(d.relative_to(ROOT))}


def live_status() -> dict:
    """实盘运行状态。只读产物文件，不连 miniQMT。

    按三段链路分别汇报，缺哪段说哪段，不把「信号跑通」说成「实盘在跑」，
    也不把「还没下单」说成「什么都没跑」。
    """
    out = {"strategies": [], "has_signal": False, "has_position": False,
           "has_execution": False}

    for s in STRATEGIES:
        if not s.live_task:
            continue
        sdir = s.dir or s.id
        sig = _signal_status(sdir)

        # 只认策略自己目录下的快照。曾写过「找不到就退回 alstm 的目录」，
        # 结果每个策略都显示有快照，实际读的是别人的数据 —— 实盘页上
        # 这种张冠李戴比空白危险得多。
        sd = ROOT / "strategies" / sdir / "state"
        pos = _read_json(sd / "positions.json") if sd.exists() else None
        eq = _read_csv(sd / "equity.csv") if sd.exists() else []

        ed = ROOT / "strategies" / sdir / "executions"
        ex_files = sorted(ed.glob("*.csv"), reverse=True) if ed.exists() else []

        out["strategies"].append({
            "id": s.id, "name": s.name, "category": s.category,
            "live_task": s.live_task,
            "signal": sig,
            "positions": pos,
            "equity_rows": len(eq),
            "n_executions": len(ex_files),
        })
        out["has_signal"] |= sig["has"]
        out["has_position"] |= pos is not None
        out["has_execution"] |= bool(ex_files)

    return out


def live_capable() -> list[Strategy]:
    """具备实盘条件的策略（登记了 live_task）。"""
    return [s for s in STRATEGIES if s.live_task]
