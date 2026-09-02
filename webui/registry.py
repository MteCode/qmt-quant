"""可运行任务的白名单。

## 为什么用白名单而不是让前端传命令

管理台能触发的每一件事都在这里显式登记。前端只提交 `task_id` 和已声明的
参数，服务端据此拼命令 —— 前端传什么都无法执行未登记的程序。

管理台默认只监听 127.0.0.1，但白名单这层不能省：浏览器上任何一个页面都能
向 localhost 发请求（CSRF），若接口能执行任意命令，一个恶意网页就能在你机器上
跑任意程序。

## 关于下单类任务

`paper_trade` 会真实下单。它在这里登记时 `dry_run` 默认为真，且标记
`dangerous=True`，前端必须二次确认才能提交非预览的执行。
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
STRATEGY = "strategies/alstm_ppo_csi1000"


@dataclass
class Param:
    name: str
    label: str
    kind: str = "int"          # int / float / str / bool / choice
    default: object = None
    choices: list = field(default_factory=list)
    flag: str = ""             # 命令行开关，留空则用 --{name}
    help: str = ""

    def cli_flag(self) -> str:
        return self.flag or f"--{self.name.replace('_', '-')}"


@dataclass
class Task:
    id: str
    name: str
    script: str
    desc: str
    #: 预计耗时，用于前端提示
    eta: str = ""
    params: list = field(default_factory=list)
    #: 会产生真实委托或不可逆副作用，前端需二次确认
    dangerous: bool = False
    #: 该任务产出哪些结果文件，跑完后前端可直接跳转查看
    outputs: list = field(default_factory=list)


TASKS = [
    Task(
        id="train_alstm",
        name="训练 ALSTM",
        script=f"{STRATEGY}/train_alstm.py",
        desc="重新训练选股模型并跑组合回测。不加复用参数时会覆盖已有权重与分数。",
        eta="约 22 分钟",
        params=[
            Param("reuse_weights", "复用已有权重（不重训）", "bool", False,
                  help="加载 alstm_weights.pt 直接推理，约 1 分钟"),
            Param("holdings", "持仓只数", "int", 50),
            Param("rebalance", "调仓周期（日）", "int", 20),
            Param("n_epochs", "训练轮数", "int", 200),
        ],
        outputs=["alstm_test_ic.csv", "alstm_only_equity.csv"],
    ),
    Task(
        id="train_ppo",
        name="训练 PPO 择时",
        script=f"{STRATEGY}/train_ppo.py",
        desc="在当前分数面板上训练仓位控制模型。改过选股分数后必须重跑，否则两者不匹配。",
        eta="约 16 分钟",
        outputs=["ppo_equity.csv", "daily_returns.csv"],
    ),
    Task(
        id="seed_experiment",
        name="多种子方差实验",
        script=f"{STRATEGY}/seed_experiment.py",
        desc="固定种子重复训练，给出指标分布而非单点估计，用于判断成绩是真本事还是运气。",
        eta="约 90 分钟",
        params=[Param("seeds", "种子个数", "int", 8)],
        outputs=["seed_experiment.json"],
    ),
    Task(
        id="train_ensemble",
        name="集成训练",
        script=f"{STRATEGY}/train_ensemble.py",
        desc="多种子截面排名平均，消除初始化随机性。会覆盖 alstm_scores.parquet。",
        eta="约 95 分钟",
        params=[Param("seeds", "种子个数", "int", 8)],
        outputs=["ensemble_result.json"],
    ),
    Task(
        id="ensemble_scaling",
        name="集成规模分析",
        script=f"{STRATEGY}/ensemble_scaling.py",
        desc="复用已存分数面板，测量方差随集成规模的收敛情况。不重训。",
        eta="约 5 分钟",
        params=[Param("samples", "每规模抽样组合数", "int", 12)],
        outputs=["ensemble_scaling.json"],
    ),
    Task(
        id="sweep_portfolio",
        name="组合参数扫描",
        script=f"{STRATEGY}/sweep_portfolio.py",
        desc="扫持仓只数 x 调仓周期网格。支持指定区间做分段检验。",
        eta="约 1 分钟",
        params=[
            Param("start", "起始日", "str", "2022-01-01"),
            Param("end", "结束日", "str", "2026-08-27"),
            Param("tag", "结果文件后缀", "str", "",
                  help="留空覆盖主结果，填 _h1 之类可保留分段结果"),
        ],
        outputs=["sweep_portfolio.json"],
    ),
    Task(
        id="generate_signal",
        name="生成交易信号",
        script=f"{STRATEGY}/generate_signal.py",
        desc="ALSTM 选股 + PPO 择时，产出目标持仓文件。",
        eta="约 1 分钟",
        params=[Param("date", "信号日期", "str", "")],
        outputs=["target_latest.csv"],
    ),
    Task(
        id="paper_trade",
        name="执行下单",
        script=f"{STRATEGY}/paper_trade.py",
        desc="读取信号文件，经风控校验后通过 miniQMT 下单。需 miniQMT 已启动登录。",
        eta="约 1 分钟",
        dangerous=True,
        params=[
            Param("dry_run", "仅预览（不下单）", "bool", True,
                  help="预览同样跑完整风控校验，只跳过最后的委托提交"),
        ],
    ),
]

TASK_BY_ID = {t.id: t for t in TASKS}


def build_command(task_id: str, values: dict) -> list:
    """按白名单拼命令。未登记的参数一律忽略。"""
    task = TASK_BY_ID.get(task_id)
    if task is None:
        raise ValueError(f"未登记的任务: {task_id}")

    cmd = [str(PYTHON), "-u", task.script]
    for p in task.params:
        if p.name not in values:
            continue
        v = values[p.name]
        if p.kind == "bool":
            # 布尔型只在为真时附加开关
            if v in (True, "true", "True", "1", 1, "on"):
                cmd.append(p.cli_flag())
        else:
            if v is None or v == "":
                continue
            cmd.extend([p.cli_flag(), str(v)])
    return cmd
