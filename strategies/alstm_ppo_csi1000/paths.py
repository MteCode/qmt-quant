"""本策略的路径与参数解析。

所有脚本从这里取路径，不写死相对路径 —— 这样从任意工作目录调用都不会
找错文件，复制整个策略目录到新名字也能直接跑。

新增策略时：复制本目录 → 改 `strategy.yaml` → 重训模型，不需要改代码。
"""
import sys
from pathlib import Path

#: 本策略目录（本文件所在处）
STRATEGY_DIR = Path(__file__).resolve().parent
#: 策略代号，即目录名
STRATEGY_NAME = STRATEGY_DIR.name
#: 项目根目录
ROOT_DIR = STRATEGY_DIR.parent.parent

# 让脚本能 import qmtquant
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

#: venv 解释器，供 subprocess 调用子脚本
PYTHON = str(ROOT_DIR / ".venv" / "Scripts" / "python.exe")
#: 共享的数据管道脚本（下载、导出 Qlib），不属于任何单一策略
SHARED_SCRIPTS = ROOT_DIR / "scripts"

# ---------------------------------------------------------------- 模型产物
MODELS_DIR = STRATEGY_DIR / "models"
#: ALSTM 选股分数面板（LFS 跟踪，重训会覆盖 —— 覆盖前务必先提交）
ALSTM_SCORES = MODELS_DIR / "alstm_scores.parquet"
#: ALSTM 网络权重。有了它就能不重训直接推理，分数可复现。
#: 曾经只存分数不存权重，分数被覆盖后对应的回测永久无法复现
ALSTM_WEIGHTS = MODELS_DIR / "alstm_weights.pt"
#: 权重对应的超参与训练时间，加载时用来校验网络结构没被改过
ALSTM_META = MODELS_DIR / "alstm_meta.json"
#: PPO 择时模型权重（LFS 跟踪）
PPO_MODEL = MODELS_DIR / "ppo_model.zip"

#: 集成模型目录。单个种子的结果方差过大（8 种子实测 Sharpe -0.491 ~ +0.532，
#: 中位数 -0.027），集成用多个种子的截面排名平均消掉这层随机性
ENSEMBLE_DIR = MODELS_DIR / "ensemble"


def ensemble_weights(seed: int):
    return ENSEMBLE_DIR / f"alstm_seed{seed}.pt"


def ensemble_scores(seed: int):
    return ENSEMBLE_DIR / f"scores_seed{seed}.parquet"

# ---------------------------------------------------------------- 回测结果
BACKTEST_DIR = STRATEGY_DIR / "backtest"

# ---------------------------------------------------------------- 每日运行
SIGNALS_DIR = STRATEGY_DIR / "signals"
#: 最新目标持仓，下单脚本读这个
LATEST_SIGNAL = SIGNALS_DIR / "target_latest.csv"

#: 运行时状态。回撤峰值存在这里，**跨重启必须保留** ——
#: DrawdownController.reset() 的文档写明实盘不应调用，删掉等于抹掉回撤记忆
STATE_DIR = STRATEGY_DIR / "state"
RISK_STATE = STATE_DIR / "risk_state.json"


def signal_file(date: str) -> Path:
    """某日的目标持仓文件"""
    return SIGNALS_DIR / f"target_{date}.csv"


def ensure_dirs() -> None:
    for d in (MODELS_DIR, ENSEMBLE_DIR, BACKTEST_DIR, SIGNALS_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_params() -> dict:
    """读 strategy.yaml。找不到就用默认值，避免新策略忘了建配置就跑不起来。"""
    import yaml

    path = STRATEGY_DIR / "strategy.yaml"
    if not path.exists():
        return dict(_DEFAULTS)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = dict(_DEFAULTS)
    merged.update(data)
    return merged


_DEFAULTS = {
    "market": "csi1000",
    "index": "000852.SH",
    "hold_k": 10,
    "rebalance_days": 20,
    "capital": 500_000,
}
