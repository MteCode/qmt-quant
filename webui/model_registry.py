"""模型版本管理 —— 训练产物的自动发现、对比、部署。

## 设计目标

每次训练自动产出标准 manifest.json，管理台自动发现并展示，
用户在界面上对比指标、选择模型、一键部署到实盘 —— 全程不改后台代码。

## 目录约定

    strategies/<策略>/runs/<run_id>/
        manifest.json       # 标准格式，训练脚本自动生成
        *.zip / *.pt        # 模型权重
        equity.csv          # 回测净值曲线
        daily_returns.csv   # 日收益 + 仓位

## 部署机制

用户在 UI 点「部署」→ 写入 strategies/<策略>/state/active_model.json，
实盘引擎启动时读此文件决定加载哪个模型权重。
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES_DIR = ROOT / "strategies"


# ---------------------------------------------------------------- manifest 格式

def create_manifest(
    *,
    run_id: str,
    model_type: str,
    strategy_id: str,
    params: dict,
    metrics: dict,
    artifacts: dict,
    kind: str = "model",
    train_period: list[str] | None = None,
    test_period: list[str] | None = None,
    train_time_sec: float | None = None,
    notes: str = "",
) -> dict:
    """构造标准 manifest dict，供训练/回测脚本调用后写入 JSON。

    `kind` 区分产物类型："model" 是训练产物，"backtest" 是回测产物。
    两者共用同一套 runs/ 目录与 manifest 结构，但在 UI 上分开展示 ——
    没有这个字段的话，回测 run 会被模型管理页当成可部署的模型。
    """
    return {
        "run_id": run_id,
        "kind": kind,
        "model_type": model_type,
        "strategy_id": strategy_id,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "completed",
        "params": params,
        "metrics": metrics,
        "artifacts": artifacts,
        "train_period": train_period or [],
        "test_period": test_period or [],
        "train_time_sec": train_time_sec,
        "notes": notes,
    }


def save_manifest(run_dir: Path, manifest: dict) -> Path:
    """把 manifest 写到 run 目录。训练脚本最后一步调用。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "manifest.json"
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def generate_run_id(model_type: str, seed: int | None = None) -> str:
    """生成唯一 run_id：<type>_<seed?>_<timestamp>。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed is not None:
        return f"{model_type}_s{seed}_{ts}"
    return f"{model_type}_{ts}"


# ---------------------------------------------------------------- 发现

def _read_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    try:
        with p.open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def _metrics_from_equity(values: list[float], capital: float) -> dict:
    """从净值序列算指标。"""
    if len(values) < 2:
        return {}
    total = values[-1] / capital - 1
    n = len(values)
    peak, mdd = values[0], 0.0
    rets = []
    for i, v in enumerate(values):
        peak = max(peak, v)
        dd = v / peak - 1
        mdd = min(mdd, dd)
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
    }


def kind_of(manifest: dict) -> str:
    """manifest 的产物类型。早于 kind 字段的 manifest 一律是训练产物。"""
    return manifest.get("kind") or "model"


def discover_runs(root: Path = ROOT, kind: str | None = None) -> list[dict]:
    """扫描所有 strategies/*/runs/*/manifest.json。kind 为 None 时不过滤。"""
    runs = []
    strategies_dir = root / "strategies"
    if not strategies_dir.is_dir():
        return runs
    for strat_dir in sorted(strategies_dir.iterdir()):
        runs_dir = strat_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = _read_json(run_dir / "manifest.json")
            if m is None:
                continue
            if kind is not None and kind_of(m) != kind:
                continue
            m["_dir"] = str(run_dir)
            m["_strategy_dir"] = strat_dir.name
            runs.append(m)
    runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return runs


def discover_models(root: Path = ROOT) -> list[dict]:
    """训练产物列表 —— 模型管理页与部署只认这一类。"""
    return discover_runs(root, kind="model")


def discover_backtests(root: Path = ROOT,
                       strategy_id: str | None = None) -> list[dict]:
    """回测产物列表，可按策略过滤。"""
    rows = discover_runs(root, kind="backtest")
    if strategy_id:
        rows = [r for r in rows
                if (r.get("strategy_id") or r.get("_strategy_dir")) == strategy_id]
    return rows


def _norm_dd(v):
    """回撤统一为负数 —— 各产物符号约定不一致，不在这里统一，
    页面就得逐处判断，迟早有一处把 -27% 显示成 +27%。"""
    if v is None:
        return None
    try:
        return -abs(float(v))
    except (TypeError, ValueError):
        return None


#: 回测 run 里进「各模式对比」表的字段
_MODE_FIELDS = ("total_return", "annual_return", "monthly_return",
                "sharpe", "volatility", "n_trades", "win_rate",
                "trading_days", "targets_passed", "targets_total")


def result_from_runs(strategy_dir: str, root: Path = ROOT) -> dict:
    """把该策略最新的回测 run 与模型 run 合成策略页要的统一结构。

    这一份取代 strategies.py 里「一个策略一个」的 _load_* 函数。那些函数
    存在的唯一原因是每个策略的产物格式都不同 —— 产物一旦统一成 manifest，
    读取逻辑就只剩这一份，与策略数量无关。
    """
    mine = [r for r in discover_runs(root)
            if (r.get("strategy_id") or r.get("_strategy_dir")) == strategy_dir]
    # discover_runs 已按 created_at 倒序，取首个即最新
    bt = next((r for r in mine if kind_of(r) == "backtest"), None)
    md = next((r for r in mine if kind_of(r) == "model"), None)
    if bt is None and md is None:
        return {"has_result": False}

    out: dict = {"has_result": True, "model": {}, "modes": []}

    if md:
        mm = md.get("metrics") or {}
        mp = md.get("params") or {}
        out["model"] = {
            "test_auc": mm.get("test_auc"),
            "test_accuracy": mm.get("test_accuracy"),
            "train_auc": mm.get("train_auc"),
            "n_features": mm.get("n_features"),
            "train_samples": mm.get("train_samples"),
            "horizon": mp.get("horizon_bars"),
            "period": " ~ ".join(str(x)[:10]
                                 for x in (md.get("train_period") or [])),
        }
        imp = (md.get("artifacts") or {}).get("feature_importance")
        if imp:
            out["importance"] = _read_csv(Path(md["_dir"]) / imp)[:10]
        out["source"] = str(Path(md["_dir"]).relative_to(root))

    if bt:
        bm = bt.get("metrics") or {}
        for mode, r in (bm.get("by_mode") or {}).items():
            row = {k: r.get(k) for k in _MODE_FIELDS}
            row["mode"] = mode
            row["max_drawdown"] = _norm_dd(r.get("max_drawdown"))
            out["modes"].append(row)
        # 顶层指标已由回测脚本取夏普最高的模式，这里直接用
        out["metrics"] = {
            "total_return": bm.get("total_return"),
            "annual_return": bm.get("annual_return"),
            "max_drawdown": _norm_dd(bm.get("max_drawdown")),
            "sharpe": bm.get("sharpe"),
            "volatility": bm.get("volatility"),
            "n_trades": bm.get("n_trades"),
            "win_rate": bm.get("win_rate"),
            "trading_days": bm.get("trading_days"),
        }
        out["config"] = bt.get("params") or {}
        out["generated_at"] = bt.get("created_at")
        out["run_id"] = bt.get("run_id")
        out.setdefault("source", str(Path(bt["_dir"]).relative_to(root)))

    return out


def get_run(run_id: str, root: Path = ROOT) -> dict | None:
    """按 run_id 查找任意类型的 run。"""
    for r in discover_runs(root):
        if r.get("run_id") == run_id:
            return r
    return None


def get_model(run_id: str, root: Path = ROOT) -> dict | None:
    """按 run_id 查找单个模型。"""
    for m in discover_models(root):
        if m.get("run_id") == run_id:
            return m
    return None


def get_model_equity(run_id: str, root: Path = ROOT) -> dict | None:
    """读模型的净值曲线数据，返回 {dates, values}。"""
    m = get_model(run_id, root)
    if not m:
        return None
    run_dir = Path(m["_dir"])
    eq_name = (m.get("artifacts") or {}).get("equity_csv", "equity.csv")
    rows = _read_csv(run_dir / eq_name)
    if not rows:
        return None
    dates, values = [], []
    for r in rows:
        d = r.get("date", "")
        v = r.get("equity") or r.get("value") or r.get("nav")
        if d and v:
            try:
                values.append(float(v))
                dates.append(str(d)[:10])
            except (ValueError, TypeError):
                continue
    if len(values) < 2:
        return None
    return {"dates": dates, "values": values}


def get_model_daily_returns(run_id: str, root: Path = ROOT) -> list[dict]:
    """读模型的日收益数据。"""
    m = get_model(run_id, root)
    if not m:
        return []
    run_dir = Path(m["_dir"])
    csv_name = (m.get("artifacts") or {}).get("daily_returns_csv",
                                               "daily_returns.csv")
    return _read_csv(run_dir / csv_name)


# ---------------------------------------------------------------- 部署管理

def _active_path(strategy_dir: str, root: Path = ROOT) -> Path:
    return root / "strategies" / strategy_dir / "state" / "active_model.json"


def active_model(strategy_dir: str, root: Path = ROOT) -> dict | None:
    """当前已部署的模型信息。"""
    return _read_json(_active_path(strategy_dir, root))


def deploy_model(run_id: str, root: Path = ROOT) -> dict:
    """部署指定模型到实盘。写入 active_model.json 供引擎读取。"""
    m = get_model(run_id, root)
    if not m:
        return {"ok": False, "error": "模型不存在"}

    strat_dir = m["_strategy_dir"]
    run_dir = Path(m["_dir"])

    artifacts = m.get("artifacts") or {}
    model_file = artifacts.get("model", "")
    model_path = str(run_dir / model_file) if model_file else ""

    state = {
        "run_id": run_id,
        "model_type": m.get("model_type", ""),
        "model_path": model_path,
        "metrics": m.get("metrics", {}),
        "params": m.get("params", {}),
        "deployed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "deployed_by": "webui",
    }

    p = _active_path(strat_dir, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    logger.info("模型已部署: %s → %s", run_id, p)
    return {"ok": True, "deployed": state}


def undeploy_model(strategy_dir: str, root: Path = ROOT) -> dict:
    """取消部署（移除 active_model.json）。"""
    p = _active_path(strategy_dir, root)
    if p.exists():
        p.unlink()
        logger.info("已取消部署: %s", strategy_dir)
    return {"ok": True}


def list_deployed(root: Path = ROOT) -> list[dict]:
    """所有策略中已部署的模型。"""
    out = []
    strategies_dir = root / "strategies"
    if not strategies_dir.is_dir():
        return out
    for d in sorted(strategies_dir.iterdir()):
        if not d.is_dir():
            continue
        am = active_model(d.name, root)
        if am:
            am["strategy_dir"] = d.name
            out.append(am)
    return out


# ---------------------------------------------------------------- 汇总

def models_summary(root: Path = ROOT) -> dict:
    """模型管理页面需要的全部数据。"""
    models = discover_models(root)
    deployed = {d["run_id"]: d for d in list_deployed(root)}

    for m in models:
        m["is_deployed"] = m.get("run_id", "") in deployed

    by_strategy: dict[str, list] = {}
    for m in models:
        sid = m.get("strategy_id") or m.get("_strategy_dir", "unknown")
        by_strategy.setdefault(sid, []).append(m)

    best_sharpe = None
    for m in models:
        s = (m.get("metrics") or {}).get("sharpe")
        if s is not None and (best_sharpe is None or s > best_sharpe):
            best_sharpe = s

    return {
        "models": models,
        "total": len(models),
        "deployed_count": sum(1 for m in models if m.get("is_deployed")),
        "by_strategy": by_strategy,
        "best_sharpe": best_sharpe,
        "strategies": sorted(by_strategy.keys()),
    }
