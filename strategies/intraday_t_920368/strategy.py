"""可注入 qmtquant 的 920368 日内做 T 策略定义。"""
from pathlib import Path
import joblib
import pandas as pd
from qmtquant.strategy.base import StrategyBase


class IntradayT920368Strategy(StrategyBase):
    parameters = ["base_value", "trade_value", "max_drawdown", "sell_time", "buy_time"]
    variables = StrategyBase.variables + ["base_volume", "trade_volume", "halted"]

    def __init__(self, engine, strategy_name, vt_symbols, setting=None):
        self.base_value, self.trade_value, self.max_drawdown = 100000.0, 100000.0, 0.15
        self.sell_time, self.buy_time = "10:15", "14:30"
        self.base_volume = self.trade_volume = 0.0
        self.halted = False
        self.model = None
        super().__init__(engine, strategy_name, vt_symbols, setting)
        p = Path(__file__).parent / "models" / "gbm_t_model.joblib"
        if p.exists():
            self.model = joblib.load(p)["model"]

    def on_bar(self, bar):
        # 实盘接入时由外部调度器调用；底仓与 T 仓位状态显式分离。
        return None

    @staticmethod
    def signal(features: pd.DataFrame, model) -> pd.Series:
        return pd.Series(model.predict_proba(features)[:, 1], index=features.index)
