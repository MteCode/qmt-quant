"""策略清单（manifest）的解析。

## 放在哪、读什么

策略项目目录下的 ``strategy.yaml``。里面已经有一组**扁平键**
（``market`` / ``index`` / ``hold_k`` / ``rebalance_days`` / ``capital``），
由 ``strategies/<dir>/paths.py::load_params()`` 读取 —— 那些是策略自己的参数，
**本模块不碰**。

本模块只读一个**可选的 ``webui:`` 块**，声明管理台需要的东西：标题、分类、
说明、结果产物在哪、有哪些可跑的任务。没有这个块也能被发现 —— 走目录默认值。

## 向后兼容是硬要求

``alstm_ppo_csi1000`` 与 ``lgb_agents_ppo`` 的 yaml 已经存在且被 ``paths.py``
使用。新增字段只能是「加一层」：不动现有键、不让旧文件失效。

## 分类默认值不能是「研究」

``tests/test_console_registry.py`` 要求 ``category == "研究"`` 的策略必须给出
非空 ``caveat``。发现层若把没写分类的策略默认成「研究」，会让一个新策略在
没写 caveat 时直接把那条测试打红 —— 默认成「未分类」。
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: 未声明分类时的默认值（**不能**是「研究」，见模块 docstring）
DEFAULT_CATEGORY = "未分类"


@dataclass
class ParamSpec:
    name: str
    label: str = ""
    kind: str = "str"          # int / float / str / bool / choice
    default: object = None
    choices: list = field(default_factory=list)
    flag: str = ""
    help: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name


@dataclass
class TaskSpec:
    id: str
    name: str
    script: str                       # 相对项目目录
    desc: str = ""
    eta: str = ""
    params: list[ParamSpec] = field(default_factory=list)
    dangerous: bool = False
    outputs: list[str] = field(default_factory=list)
    param_style: str = "flag"         # flag | set（set 走 --set k=v）
    extra_args: list[str] = field(default_factory=list)
    #: 该任务所属项目的仓库相对目录，由 discovery 填入
    project_dir: str = ""


@dataclass
class ResultSpec:
    kind: str = "none"                # summary | equity | none
    source_json: str = ""             # 相对 output_dir
    metrics: dict = field(default_factory=dict)   # 统一键 -> JSON 点路径
    period: str = ""                  # JSON 点路径
    n_trades: str = ""                # JSON 点路径
    normalize_drawdown: bool = True   # 回撤归一到 <= 0
    equity_csv: str = ""
    equity_date_col: str = "date"
    equity_value_col: str = "equity"


@dataclass
class StrategySpec:
    id: str
    name: str = ""
    category: str = DEFAULT_CATEGORY
    summary: str = ""
    how: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    risk: list[str] = field(default_factory=list)
    code: str = ""
    dir: str = ""
    output_dir: str = ""
    status: str = "research"
    caveat: str = ""
    backtest_task: str = ""
    live_task: str = ""
    result: ResultSpec = field(default_factory=ResultSpec)
    tasks: list[TaskSpec] = field(default_factory=list)
    #: 代码策略（StrategyBase 子类）才有：完整点路径 "pkg.mod.Class"
    dotted: str = ""
    params: list[ParamSpec] = field(default_factory=list)
    #: project（strategies/ 目录）| builtin（loader.BUILTIN）| code（策略文件里的类）
    source: str = "project"


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v else []
    return [str(x) for x in v]


def _param_from_dict(d: dict) -> ParamSpec:
    return ParamSpec(
        name=str(d["name"]),
        label=str(d.get("label", "")),
        kind=str(d.get("kind", "str")),
        default=d.get("default"),
        choices=list(d.get("choices", []) or []),
        flag=str(d.get("flag", "")),
        help=str(d.get("help", "")),
    )


def parse_task(d: dict, project_dir: str) -> TaskSpec:
    return TaskSpec(
        id=str(d["id"]),
        name=str(d.get("name", d["id"])),
        script=str(d["script"]),
        desc=str(d.get("desc", "")),
        eta=str(d.get("eta", "")),
        params=[_param_from_dict(p) for p in d.get("params", []) or []],
        dangerous=bool(d.get("dangerous", False)),
        outputs=_as_list(d.get("outputs")),
        param_style=str(d.get("param_style", "flag")),
        extra_args=_as_list(d.get("extra_args")),
        project_dir=project_dir,
    )


def parse_result(d: dict) -> ResultSpec:
    return ResultSpec(
        kind=str(d.get("kind", "none")),
        source_json=str(d.get("source_json", "")),
        metrics=dict(d.get("metrics", {}) or {}),
        period=str(d.get("period", "")),
        n_trades=str(d.get("n_trades", "")),
        normalize_drawdown=bool(d.get("normalize_drawdown", True)),
        equity_csv=str(d.get("equity_csv", "")),
        equity_date_col=str(d.get("equity_date_col", "date")),
        equity_value_col=str(d.get("equity_value_col", "equity")),
    )


def parse_webui_block(raw: dict, dir_name: str,
                      project_dir: str = "") -> StrategySpec:
    """从一份 strategy.yaml 的完整 dict 里取出 webui 块，构造 StrategySpec。

    ``raw`` 是 ``yaml.safe_load`` 的结果（含那组扁平键 —— 本函数只读 ``webui``）。
    所有字段都可缺省，缺省值按 ``dir_name`` 推导。
    非 dict 的 ``webui`` 值（用户写错）按「没写」处理，不让整份清单失效。
    """
    project_dir = project_dir or f"strategies/{dir_name}"
    block = raw.get("webui")
    if not isinstance(block, dict):
        block = {}

    spec = StrategySpec(
        id=str(block.get("id", dir_name)),
        name=str(block.get("name", dir_name)),
        category=str(block.get("category", DEFAULT_CATEGORY)),
        summary=str(block.get("summary", "")),
        how=_as_list(block.get("how")),
        inputs=_as_list(block.get("inputs")),
        risk=_as_list(block.get("risk")),
        code=str(block.get("code", f"{project_dir}/")),
        dir=dir_name,
        output_dir=str(block.get("output_dir", f"{project_dir}/backtest")),
        status=str(block.get("status", "research")),
        caveat=str(block.get("caveat", "")),
        backtest_task=str(block.get("backtest_task", "")),
        live_task=str(block.get("live_task", "")),
        source="project",
    )
    res = block.get("results")
    if isinstance(res, dict):
        spec.result = parse_result(res)
    spec.tasks = [parse_task(t, project_dir)
                  for t in block.get("tasks", []) or []
                  if isinstance(t, dict) and "id" in t and "script" in t]
    return spec
