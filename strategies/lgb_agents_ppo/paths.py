"""本策略的路径常量。集中在这里，避免各脚本各拼各的。"""
import json
from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parent
ROOT = STRATEGY_DIR.parents[1]

MODELS_DIR = STRATEGY_DIR / "models"
BACKTEST_DIR = STRATEGY_DIR / "backtest"
SIGNALS_DIR = STRATEGY_DIR / "signals"
STATE_DIR = STRATEGY_DIR / "state"

#: 第 1 层产出：全量打分面板（日期 x 标的）
SCREEN_SCORES = MODELS_DIR / "screen_scores.parquet"
#: 第 1 层产出：每个调仓日的候选池
CANDIDATES = MODELS_DIR / "candidates.parquet"
#: 模型权重与元信息
SCREEN_MODEL = MODELS_DIR / "screen_model.pkl"
SCREEN_META = MODELS_DIR / "screen_meta.json"
#: 特征筛选结果 —— 哪些因子被保留、各自贡献多少
FEATURE_REPORT = BACKTEST_DIR / "feature_selection.json"
#: 多种子稳健性检验报告
ROBUSTNESS = BACKTEST_DIR / "robustness.json"

CONFIG_FILE = STRATEGY_DIR / "strategy.yaml"


def ensure_dirs() -> None:
    for d in (MODELS_DIR, BACKTEST_DIR, SIGNALS_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_params() -> dict:
    """读策略参数。文件缺失时回落到默认值，不让脚本因此跑不起来。"""
    defaults = {
        "market": "csi1000",
        "index": "000852.SH",
        "candidate_k": 100,
        "hold_k": 30,
        "rebalance_days": 20,
        "capital": 500_000,
    }
    if not CONFIG_FILE.exists():
        return defaults
    try:
        import yaml
        cfg = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}
    except Exception:
        return defaults
    return {**defaults, **cfg}
