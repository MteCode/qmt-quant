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
        id="download_full_market",
        name="下载全市场行情",
        script="scripts/download_full_market.py",
        desc="从 miniQMT 下载当前沪深A股行情。支持日线及分钟线；分钟线受券商限制，通常只有最近约 1 年。",
        eta="分钟线约数十分钟至数小时",
        params=[
            Param("intervals", "周期（逗号分隔）", "str", "5m",
                  help="例如 5m，或 1d,5m"),
            Param("start", "起始日期", "str", "2025-09-01"),
            Param("end", "结束日期", "str", "2026-09-06"),
            Param("include_st", "包含 ST/退市风险标的", "bool", False,
                  help="默认排除当前名称含 ST 或退字样的标的"),
            Param("rebuild", "重新下载已有文件", "bool", False),
        ],
    ),
    Task(
        id="update_market_data",
        name="更新行情",
        script="scripts/update_market_data.py",
        desc="下载最新日线并导出 Qlib 格式。两步必须成对执行 —— "
             "只下载不导出，模型读到的仍是旧数据。需 miniQMT 已启动。",
        eta="约 5-15 分钟",
        params=[
            Param("mode", "模式", "choice", "post",
                  choices=["post", "pre"],
                  help="post=盘后更新（当日日线已定稿）；pre=盘前补齐昨夜遗漏"),
            Param("sector", "板块", "str", "中证1000"),
            Param("index", "导出指数", "str", "000852.SH"),
            Param("all", "全量导出（消除跨指数数据撕裂，较慢）", "bool", False),
            Param("skip_download", "只导出，不下载", "bool", False),
        ],
    ),
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
        id="snapshot_positions",
        name="刷新持仓快照",
        script=f"{STRATEGY}/snapshot_positions.py",
        desc="从 miniQMT 拉取账户实际持仓与资金，供「实盘执行」页展示。"
             "只读不下单。需 miniQMT 已启动登录。",
        eta="约 10 秒",
    ),
    Task(
        id="track_equity",
        name="记录实盘净值",
        script=f"{STRATEGY}/track_equity.py",
        desc="每日记一笔总资产与回撤，并与回测对照。只读不下单。"
             "漏记的日子补不回来 —— 券商查不到历史净值序列。",
        eta="约 10 秒",
        params=[Param("date", "记录日期", "str", "", help="留空为今天")],
    ),
    Task(
        id="risk_monitor",
        name="盘中风控巡检",
        script=f"{STRATEGY}/risk_monitor.py",
        desc="跟踪回撤与当日盈亏，触线自动减仓/清仓。与回测共用同一套 "
             "DrawdownController，口径一致。默认 --once 只查一次；"
             "去掉 --once 会变成常驻守护。",
        eta="--once 约 10 秒",
        params=[
            Param("once", "只检查一次", "bool", True,
                  help="关闭则常驻轮询，适合手动长开"),
            Param("dry_run", "只告警不下单", "bool", True,
                  help="触线时是否真的减仓"),
            Param("interval", "轮询间隔（秒）", "int", 30,
                  help="仅常驻模式有效"),
        ],
        # 非 dry-run 时会真实减仓下单
        dangerous=True,
    ),
    Task(
        id="reconcile",
        name="成交回报对账",
        script=f"{STRATEGY}/reconcile.py",
        desc="核对下单意图与实际成交：成交率、滑点、废单。只读不下单。"
             "不对账等于蒙眼交易 —— 持续负滑点会悄悄吃掉收益。",
        eta="约 10 秒",
        params=[Param("date", "对账日期", "str", "",
                      help="留空为今天")],
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

# 盘中行情增量更新
TASKS.append(
    Task(
        id="update_intraday",
        name="盘中行情增量更新",
        script="scripts/update_intraday.py",
        desc="增量追加全市场 1m/5m K 线，供日内策略实时计算特征。"
             "需 miniQMT 已启动。单次运行约 1-3 分钟。",
        eta="约 1-3 分钟",
        params=[
            Param("intervals", "周期（逗号分隔）", "str", "1m",
                  help="例如 1m，或 1m,5m"),
            Param("clean", "更新后自动清洗", "bool", True),
            Param("loop", "持续运行间隔（秒，0=单次）", "int", 0,
                  help="盘中每 N 秒自动刷新，非交易时段自动休眠"),
        ],
    ),
)

# 全市场日内 GBM
TASKS.extend([
    Task(
        id="train_intraday_gbm",
        name="训练全市场日内GBM",
        script="scripts/train_intraday_gbm.py",
        desc="读取全市场 1m 清洗数据，计算 24 维日内特征，训练 LightGBM 选股模型。"
             "输出模型文件、指标和特征重要性。",
        eta="视标的数量，500 只约 5-10 分钟，全市场约 30-60 分钟",
        params=[
            Param("horizon", "预测窗口（bar 数）", "int", 10),
            Param("threshold", "正类阈值", "float", 0.0005),
            Param("max_symbols", "最多使用标的数（0=全部）", "int", 0,
                  help="快速实验时限制标的数量"),
            Param("min_bars", "最少 bar 数", "int", 1000),
        ],
        outputs=["models/intraday_gbm/metrics.json",
                 "models/intraday_gbm/feature_importance.csv"],
    ),
    Task(
        id="backtest_intraday_gbm",
        name="日内GBM策略回测",
        script="scripts/backtest_intraday_gbm.py",
        desc="用训练好的模型回测三种日内模式（做T/均值回归/打板），"
             "计算收益、回撤、夏普，并对照硬性指标（月收益≥30%、"
             "回撤≤10%、夏普>1）。含单票止损与组合回撤控制。",
        eta="800 只标的约 10-20 分钟",
        params=[
            Param("mode", "交易模式", "str", "all",
                  help="t_plus_0 / mean_reversion / momentum / all"),
            Param("capital", "初始资金", "float", 50000.0),
            Param("max_positions", "同时持仓标的数", "int", 5),
            Param("prob_buy", "买入概率阈值", "float", 0.60),
            Param("prob_sell", "卖出概率阈值", "float", 0.40),
            Param("max_intraday_loss", "单票日内止损线", "float", 0.02),
            Param("max_drawdown_stop", "组合回撤停止开仓线", "float", 0.10),
            Param("max_symbols", "回测标的数", "int", 800),
            Param("start", "起始日期", "str", "",
                  help="留空则用全部历史"),
        ],
        outputs=["models/intraday_gbm/backtest/summary.json"],
    ),
    Task(
        id="predict_intraday",
        name="盘中全市场预测",
        script="scripts/predict_intraday.py",
        desc="用训练好的全市场日内 GBM 模型对所有标的实时打分，"
             "输出每只股票的上涨概率，供日内策略筛选标的。",
        eta="约 1-2 分钟",
        params=[
            Param("top", "只输出 top N（0=全部）", "int", 50),
            Param("loop", "持续运行间隔（秒，0=单次）", "int", 0),
        ],
        outputs=["predictions/"],
    ),
])

# 920368 1分钟日内做T：模型训练与样本外回测
TASKS.extend([
    Task(id="train_intraday_t_920368", name="训练920368做T模型",
         script="strategies/intraday_t_920368/train_gbm.py",
         desc="用920368.BSE清洗后的1分钟数据训练LightGBM短线方向模型。",
         eta="约1分钟", outputs=["strategies/intraday_t_920368/models/gbm_metrics.json"]),
    Task(id="backtest_intraday_t_920368", name="回测920368日内做T",
         script="strategies/intraday_t_920368/run_backtest.py",
         desc="底仓10万元、T仓10万元，每日调仓并执行15%最大回撤风控。",
         eta="约1分钟", outputs=["strategies/intraday_t_920368/backtest/report.html", "strategies/intraday_t_920368/backtest/summary.json"]),
])

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
