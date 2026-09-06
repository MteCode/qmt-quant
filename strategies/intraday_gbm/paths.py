"""全市场日内 GBM 策略的路径定义。"""
import sys
from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parent
STRATEGY_NAME = STRATEGY_DIR.name
ROOT_DIR = STRATEGY_DIR.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PYTHON = str(ROOT_DIR / ".venv" / "Scripts" / "python.exe")

# 模型在项目级 models/ 下，因为是全市场共享的
MODEL_DIR = ROOT_DIR / "models" / "intraday_gbm"
MODEL_FILE = MODEL_DIR / "model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

# 预测输出
PREDICTIONS_DIR = ROOT_DIR / "predictions"

# 策略运行时状态
STATE_DIR = STRATEGY_DIR / "state"
RISK_STATE = STATE_DIR / "risk_state.json"
