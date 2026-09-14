"""策略自动发现 —— 让「加一个策略」不必改 webui 代码。

## 发现两种东西

1. **策略项目**：``strategies/<目录>/strategy.yaml``（可选的 ``webui:`` 块）
2. **代码策略**：``StrategyBase`` 具体子类 ——
   ``qmtquant/research/loader.py::BUILTIN`` 里的内置短名，
   以及 ``strategies/*/strategy.py`` 里定义的类

## 只返回中性 spec，不 import webui 其他模块

本模块 import ``manifest`` 与 ``qmtquant``，**不** import ``webui.strategies`` 或
``webui.registry`` —— 那两个模块反过来 import 本模块，双向 import 会成环。
转换（spec → Strategy / Task）由它们各自做。

## 容错

一个坏 manifest、一个 import 失败的策略文件，都只跳过它自己并记日志，
不能连累其余策略被发现。上层还会再包一层 try/except 兜底。
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from pathlib import Path

from . import manifest
from .manifest import ParamSpec, StrategySpec, TaskSpec

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIRNAME = "strategies"

#: 约定脚本：项目根目录下这些名字的 .py 会被自动列成可跑任务
_CONVENTION_NAMES = {
    "generate_signal.py", "paper_trade.py", "run_backtest.py",
    "rebalance.py", "train_signal.py",
}
#: 约定前缀
_CONVENTION_PREFIXES = ("train_", "sweep_")
#: 会真下单或不可逆，前端需二次确认
_DANGEROUS_STEMS = {"paper_trade", "risk_monitor", "scheduled_trade"}


# --------------------------------------------------------------- 安全

def resolve_script(project_dir: Path, script: str) -> Path:
    """把 manifest 里声明的脚本解析成项目目录下的真实路径。

    这是**安全边界**：manifest 是仓库内的文件、与源码同等可信，但仍要挡住
    「声明一个绝对路径或 ../.. 去跑仓库外的程序」这种情况 —— 发现出来的任务
    必须落在项目目录内、且是 .py。

    :raises ValueError: 绝对路径 / 含 ``..`` / 非 ``.py`` / 越出项目目录
    """
    raw = str(script or "").strip()
    if not raw:
        raise ValueError("脚本名为空")
    p = Path(raw)
    if p.is_absolute():
        raise ValueError(f"脚本不能是绝对路径: {raw}")
    if p.suffix != ".py":
        raise ValueError(f"脚本必须是 .py: {raw}")
    if any(part == ".." for part in p.parts):
        raise ValueError(f"脚本不能含 ..: {raw}")

    base = project_dir.resolve()
    full = (base / p).resolve()
    if full != base and base not in full.parents:
        raise ValueError(f"脚本越出项目目录: {raw}")
    return full


# --------------------------------------------------------------- 参数派生

def _kind_of(annotation, default) -> str:
    """从类型注解或默认值推参数种类。**bool 必须排在 int 之前** ——
    ``bool`` 是 ``int`` 的子类，先判 int 会把 True/False 认成整数。"""
    if isinstance(annotation, str):          # from __future__ import annotations
        s = annotation.lower()
        for k in ("bool", "int", "float", "str"):
            if k in s:
                return k
    else:
        if annotation is bool:
            return "bool"
        if annotation is int:
            return "int"
        if annotation is float:
            return "float"
        if annotation is str:
            return "str"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    return "str"


def derive_params(cls: type) -> list[ParamSpec]:
    """由 ``StrategyBase.parameters`` + 类默认值推通用参数表单。

    只读 ``parameters`` 里声明过的名字（与 ``check_params`` 的口径一致）；
    类型优先取注解（走 MRO 收集，子类覆盖父类），取不到则看默认值类型。
    可选约定：类属性 ``<name>_choices`` 存在时视为枚举参数。
    """
    names = list(getattr(cls, "parameters", []) or [])
    anns: dict = {}
    for klass in reversed(getattr(cls, "__mro__", [])):
        anns.update(klass.__dict__.get("__annotations__", {}) or {})
    out: list[ParamSpec] = []
    for n in names:
        default = getattr(cls, n, None)
        choices = list(getattr(cls, f"{n}_choices", []) or [])
        kind = "choice" if choices else _kind_of(anns.get(n), default)
        out.append(ParamSpec(name=n, label=n, kind=kind,
                             default=default, choices=choices))
    return out


# --------------------------------------------------------------- 项目发现

def _read_yaml(path: Path) -> dict | None:
    import yaml
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        logger.warning("读不了 %s: %s", path, e)
        return None
    if not isinstance(raw, dict):
        logger.warning("%s 顶层不是键值表，跳过", path)
        return None
    return raw


def discover_projects(root: Path = ROOT) -> list[StrategySpec]:
    """扫 ``strategies/*/strategy.yaml``。坏文件只跳过它自己。"""
    base = root / PROJECTS_DIRNAME
    out: list[StrategySpec] = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        y = d / "strategy.yaml"
        if not (d.is_dir() and y.exists()):
            continue
        raw = _read_yaml(y)
        if raw is None:
            continue
        try:
            out.append(manifest.parse_webui_block(
                raw, d.name, f"{PROJECTS_DIRNAME}/{d.name}"))
        except Exception as e:                   # noqa: BLE001
            logger.warning("解析 %s 失败: %s", y, e)
    return out


def project_tasks(spec: StrategySpec, root: Path = ROOT) -> list[TaskSpec]:
    """该项目的可跑任务 = manifest 显式声明 + 约定脚本。

    约定脚本让**没有 webui 块**的既有项目（intraday_gbm / intraday_t_920368）
    也能列出任务，不必逐个手写。
    """
    proj_rel = f"{PROJECTS_DIRNAME}/{spec.dir}"
    proj = root / PROJECTS_DIRNAME / spec.dir
    out = list(spec.tasks)
    have = {t.script for t in out}

    if proj.is_dir():
        for f in sorted(proj.glob("*.py")):
            stem = f.stem
            if not (stem.startswith(_CONVENTION_PREFIXES)
                    or f.name in _CONVENTION_NAMES):
                continue
            if f.name in have:
                continue
            out.append(TaskSpec(
                id=f"{spec.id}.{stem}",
                name=stem.replace("_", " "),
                script=f.name,
                desc=f"自动发现：{f.name}",
                dangerous=stem in _DANGEROUS_STEMS,
            ))

    for t in out:
        t.project_dir = proj_rel
    return out


# --------------------------------------------------------------- 代码发现

def _is_concrete_strategy(cls) -> bool:
    from qmtquant.strategy.base import StrategyBase
    from qmtquant.strategy.portfolio import PortfolioStrategy
    if not (isinstance(cls, type) and issubclass(cls, StrategyBase)):
        return False
    if cls in (StrategyBase, PortfolioStrategy):
        return False
    try:
        if inspect.isabstract(cls):
            return False
    except Exception:                            # noqa: BLE001
        return False
    return True


def _spec_from_cls(cls, dotted: str, ident: str, source: str,
                   dir_name: str = "") -> StrategySpec:
    doc = (cls.__doc__ or "").strip().splitlines()
    summary = doc[0].strip() if doc else ""
    return StrategySpec(
        id=ident,
        name=cls.__name__,
        category="策略库",
        summary=summary,
        code=f"{cls.__module__.replace('.', '/')}.py",
        dir=dir_name,
        output_dir="",
        status="backtest_only",
        dotted=dotted,
        params=derive_params(cls),
        source=source,
    )


def _classes_in_file(path: Path, root: Path) -> list[type]:
    """import 一个策略文件并取出其中定义的 StrategyBase 子类。

    import 失败（缺依赖、语法错、相对 import 断裂）一律吞掉并返回空 ——
    一个坏文件不该让整个发现停摆。
    """
    rel = path.relative_to(root)
    pkg_dir = path.parent
    modname = ".".join(rel.with_suffix("").parts)
    try:
        if (pkg_dir / "__init__.py").exists():
            mod = importlib.import_module(modname)
        else:
            spec = importlib.util.spec_from_file_location(modname, path)
            if spec is None or spec.loader is None:
                return []
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    except Exception as e:                       # noqa: BLE001
        logger.info("跳过策略文件 %s: %s", rel, e)
        return []

    return [v for v in vars(mod).values()
            if _is_concrete_strategy(v) and getattr(v, "__module__", "") == modname]


def discover_code_strategies(root: Path = ROOT) -> list[StrategySpec]:
    """内置短名 + ``strategies/*/strategy.py`` 里定义的策略类。"""
    from qmtquant.research import loader

    out: list[StrategySpec] = []
    seen_cls: set[type] = set()

    for short, dotted in loader.BUILTIN.items():
        try:
            cls = loader.load_strategy(short)
        except Exception as e:                   # noqa: BLE001
            logger.info("内置策略 %s 加载失败: %s", short, e)
            continue
        if not _is_concrete_strategy(cls):
            continue
        seen_cls.add(cls)
        out.append(_spec_from_cls(cls, dotted, short, source="builtin"))

    base = root / PROJECTS_DIRNAME
    if base.is_dir():
        for d in sorted(base.iterdir()):
            f = d / "strategy.py"
            if not (d.is_dir() and f.exists()):
                continue
            for cls in _classes_in_file(f, root):
                if cls in seen_cls:
                    continue
                seen_cls.add(cls)
                dotted = f"{cls.__module__}.{cls.__name__}"
                out.append(_spec_from_cls(cls, dotted, cls.__name__,
                                          source="code", dir_name=d.name))
    return out


# --------------------------------------------------------------- 合并

def keys(ident: str, dir_: str = "", output_dir: str = "") -> set[str]:
    """去重键。核心清单与发现结果用**同一套**键，才能正确判重。"""
    k = {ident}
    if dir_:
        k.add(f"dir:{dir_}")
    if output_dir:
        k.add("out:" + output_dir.replace("\\", "/").strip("/"))
    return k


def discovered_specs(root: Path = ROOT) -> list[StrategySpec]:
    """项目 + 代码策略，按 id 去重，按 (分类, 名称) 排序。"""
    specs = discover_projects(root) + discover_code_strategies(root)
    seen: set[str] = set()
    out: list[StrategySpec] = []
    for s in specs:
        if s.id in seen:
            continue
        seen.add(s.id)
        out.append(s)
    out.sort(key=lambda s: (s.category, s.name))
    return out
