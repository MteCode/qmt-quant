"""策略与模型总目录 —— config/catalog.yaml 的加载器。

## 为什么要有这个文件

加一个策略原先要改三四个 Python 文件：webui/strategies.py 的元信息、
webui/registry.py 的任务与参数、webui/services.py 的常驻服务，还要记得
在 strategies/<id>/strategy.yaml 里也写一份给自动发现用。漏掉任何一处，
表现都是「策略在某个页面看得到、在另一个页面看不到」。

catalog.yaml 是这些的统一手工入口：一条记录同时描述策略元信息、模型、
回测任务和实盘服务，加策略只改那一个文件。

## 与既有登记的关系

**补充，不替换。** strategies.py 里手写的那十几个条目继续有效，
catalog 里的条目追加进去；id 相同时以 catalog 为准，方便逐个迁移而不必
一次性重写 1400 行。

## 不做什么

不负责发现磁盘上的产物 —— 那是 model_registry 扫 runs/*/manifest.json
的事。catalog 只管「有哪些策略、各自的入口在哪」。
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "catalog.yaml"

_VALID_STATUS = {"research", "backtest_only", "live_ready"}


def _load_raw(path: Path | None = None) -> dict:
    p = path or CATALOG_PATH
    if not p.exists():
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("未安装 PyYAML，catalog.yaml 不生效")
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:                           # noqa: BLE001
        # 写坏了要吵出来。静默忽略会让「我明明加了策略却不显示」
        # 变成一个没人查得动的问题
        logger.error("catalog.yaml 解析失败，全部条目未生效: %s", e)
        return {}
    if not isinstance(data, dict):
        logger.error("catalog.yaml 顶层应是映射，实际是 %s", type(data).__name__)
        return {}
    return data


def entries(path: Path | None = None) -> list[dict]:
    """返回 catalog 中的策略条目，逐条校验。

    单条写错只跳过那一条并记日志，不影响其余 —— 一个手滑不该让整个
    策略中心空掉。
    """
    raw = _load_raw(path)
    items = raw.get("strategies") or []
    if not isinstance(items, list):
        logger.error("catalog.yaml 的 strategies 应是列表")
        return []

    out = []
    seen: set[str] = set()
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            logger.error("catalog.yaml strategies[%d] 不是映射，已跳过", i)
            continue
        sid = str(it.get("id") or "").strip()
        if not sid:
            logger.error("catalog.yaml strategies[%d] 缺 id，已跳过", i)
            continue
        if sid in seen:
            logger.error("catalog.yaml 中 id 重复: %s，后一条已跳过", sid)
            continue
        status = it.get("status") or "research"
        if status not in _VALID_STATUS:
            logger.warning("catalog.yaml %s 的 status=%r 不认识，按 research 处理",
                           sid, status)
            it = {**it, "status": "research"}
        seen.add(sid)
        out.append(it)
    return out


def _as_list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def build_strategies(strategy_cls, path: Path | None = None) -> list:
    """把 catalog 条目转成 strategies.py 的 Strategy 对象。

    loader 留空 —— 回测结果统一由 model_registry.result_from_runs 读
    manifest，不需要每个策略再写一个专用读取函数。
    """
    from . import model_registry

    out = []
    for it in entries(path):
        sid = it["id"]
        bt = it.get("backtest") or {}
        live = it.get("live") or {}
        model = it.get("model") or {}

        def _loader(_sid=sid):
            return model_registry.result_from_runs(_sid)

        out.append(strategy_cls(
            id=sid,
            name=it.get("name") or sid,
            category=it.get("category") or "研究",
            summary=it.get("summary") or "",
            how=_as_list(it.get("how")),
            inputs=_as_list(it.get("inputs")),
            risk=_as_list(it.get("risk")),
            code=it.get("code") or model.get("train_script") or "",
            output_dir=it.get("output_dir") or "",
            backtest_task=f"catalog_bt_{sid}" if bt.get("script") else "",
            live_task=live.get("script") or "",
            live_service=live.get("service") or "",
            status=it.get("status") or "research",
            caveat=it.get("caveat") or "",
            loader=_loader,
        ))
    return out


def build_tasks(task_cls, param_cls, path: Path | None = None) -> list:
    """把 catalog 里的 backtest 段转成 registry.py 的 Task 对象。"""
    out = []
    for it in entries(path):
        bt = it.get("backtest") or {}
        script = bt.get("script")
        if not script:
            continue
        params = []
        for p in _as_list(bt.get("params")):
            if not isinstance(p, dict) or not p.get("name"):
                logger.error("catalog.yaml %s 的 backtest.params 有条目缺 name",
                             it["id"])
                continue
            params.append(param_cls(
                name=p["name"],
                label=p.get("label") or p["name"],
                kind=p.get("kind") or "str",
                default=p.get("default"),
                choices=_as_list(p.get("choices")),
                flag=p.get("flag") or "",
                help=p.get("help") or "",
            ))
        out.append(task_cls(
            id=f"catalog_bt_{it['id']}",
            name=f"{it.get('name') or it['id']} 回测",
            script=script,
            desc=it.get("summary") or "",
            eta=bt.get("eta") or "",
            params=params,
            dangerous=bool(bt.get("dangerous")),
            outputs=_as_list(bt.get("outputs")),
        ))
    return out


def model_label_mode(strategy_id: str, path: Path | None = None) -> str:
    """该策略登记的模型口径，用于和实际加载的模型比对。"""
    for it in entries(path):
        if it["id"] == strategy_id:
            return ((it.get("model") or {}).get("label_mode") or "")
    return ""
