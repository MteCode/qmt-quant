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
    train_period: list[str] | None = None,
    test_period: list[str] | None = None,
    train_time_sec: float | None = None,
    notes: str = "",
) -> dict:
    """构造标准 manifest dict，供训练脚本调用后写入 JSON。"""
    return {
        "run_id": run_id,
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


def discover_models(root: Path = ROOT) -> list[dict]:
    """扫描所有 strategies/*/runs/*/manifest.json，返回模型列表。"""
    models = []
    strategies_dir = root / "strategies"
    if not strategies_dir.is_dir():
        return models
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
            m["_dir"] = str(run_dir)
            m["_strategy_dir"] = strat_dir.name
            models.append(m)
    models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return models


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
