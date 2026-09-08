"""全市场日内 GBM 策略 —— 根据模型打分筛选标的并执行日内交易。

支持多种日内交易模式（通过 trade_mode 参数切换）：

- **t_plus_0**：做 T 策略。持有底仓的票，模型看多时加仓、看空时减仓，
  日内回转赚价差。A 股 T+1 限制下需要先有底仓。
- **mean_reversion**：均值回归。价格偏离日内 VWAP 超过阈值时反向操作，
  模型概率作为过滤器（只做模型认可方向的均值回归）。
- **momentum**：打板/追涨。模型高概率 + 量能放大 + 日内涨幅突破时买入，
  尾盘或止损时卖出。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from qmtquant.core.constants import Direction, OrderType
from qmtquant.core.objects import BarData
from qmtquant.features.intraday import INTRADAY_FEATURES, make_intraday_features
from qmtquant.strategy.base import StrategyBase

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parents[2] / "models" / "intraday_gbm" / "model.joblib"


class IntradayGBMStrategy(StrategyBase):
    """全市场日内 GBM 策略。

    工作流：
    1. on_bars() 收到全市场同一时刻的 1m K 线
    2. 计算特征 → 模型打分 → 按概率排序
    3. 根据 trade_mode 决定操作：做T/均值回归/打板
    4. 风控检查后下单
    """

    parameters = [
        "trade_mode",           # t_plus_0 / mean_reversion / momentum
        "max_positions",        # 同时持仓标的数
        "position_size",        # 每只标的的仓位金额
        "prob_buy_threshold",   # 买入概率阈值
        "prob_sell_threshold",  # 卖出概率阈值
        "max_intraday_loss",    # 单票日内最大亏损比例
        "entry_time",           # 最早入场时间 HH:MM
        "exit_time",            # 最晚持仓时间 HH:MM（到时强制平仓）
        "vol_z_threshold",      # momentum 模式的量能放大阈值
        "vwap_deviation",       # mean_reversion 模式的 VWAP 偏离阈值
        "use_rank",             # 按概率排序选前 N 名，而非用绝对阈值
    ]

    variables = StrategyBase.variables + [
        "today_entries",        # 今日已入场标的
        "entry_prices",         # 入场价格记录
    ]

    def __init__(self, engine: Any, strategy_name: str,
                 vt_symbols: list[str], setting: dict | None = None):
        # 默认参数
        self.trade_mode = "t_plus_0"
        self.max_positions = 10
        self.position_size = 50000.0
        self.prob_buy_threshold = 0.6
        self.prob_sell_threshold = 0.4
        self.max_intraday_loss = 0.02
        self.entry_time = "09:35"
        self.exit_time = "14:50"
        self.vol_z_threshold = 1.5
        self.vwap_deviation = 0.005
        # 按排序选股而非绝对阈值。横截面模型的绝对概率取决于训练时的
        # 正类比例，换个 horizon 重训一次就全变了；而**排序是稳定的**——
        # 这也正是横截面选股模型该用的方式。
        self.use_rank = False

        self.today_entries: dict[str, float] = {}
        self.entry_prices: dict[str, float] = {}

        self.model = None
        self.features = INTRADAY_FEATURES
        self._bar_buffers: dict[str, list] = {}

        super().__init__(engine, strategy_name, vt_symbols, setting)

    def on_init(self) -> None:
        if not MODEL_PATH.exists():
            self.write_log(f"模型不存在: {MODEL_PATH}，策略将不产生信号")
            return

        data = joblib.load(MODEL_PATH)
        self.model = data["model"]
        self.features = data.get("features", INTRADAY_FEATURES)
        self.write_log(f"模型加载成功，{len(self.features)} 维特征")
        self._check_threshold_reachable(data)

    def _check_threshold_reachable(self, model_data: dict) -> None:
        """检查买入阈值对当前模型是否可达。

        ## 为什么必须查

        概率阈值是绝对值，而模型输出的概率分布取决于**训练时的正类比例**。
        重训换了 horizon / threshold 之后，正类比例从 38% 掉到 16.5%，
        模型输出整体下移（实测中位数 0.058、最大 0.361），
        而策略里写死的 0.6 永远达不到。

        表现是最糟的那种：引擎正常、行情正常、打分正常、
        **策略一整天不下一笔单，且不报任何错**。
        查起来要从行情一路追到策略内部才能发现。

        所以启动时就用训练集的正类比例估一下阈值是否离谱，
        离谱就大声说出来，而不是等交易日结束才发现什么都没做。
        """
        import json
        from pathlib import Path as _P

        metrics_path = _P(MODEL_PATH).parent / "metrics.json"
        pos_rate = None
        if metrics_path.exists():
            try:
                m = json.loads(metrics_path.read_text(encoding="utf-8"))
                pos_rate = m.get("test_positive_rate")
                self.write_log(
                    f"模型: horizon={m.get('horizon_bars')} bar, "
                    f"阈值={m.get('threshold', 0) * 10000:.0f}bp, "
                    f"正类率={pos_rate:.1%}, AUC={m.get('test_auc', 0):.4f}")
            except (OSError, ValueError, TypeError):
                pass

        if pos_rate is None:
            return

        # 二分类模型的输出大致以正类比例为中心。买入阈值远高于它时，
        # 能触发的样本会少到实际为零。这里用 3 倍作为「明显离谱」的界线 ——
        # 精确阈值不重要，重要的是把静默失效变成显式告警。
        if self.prob_buy_threshold > pos_rate * 3:
            self.write_log(
                f"【警告】买入阈值 {self.prob_buy_threshold} 相对模型正类率 "
                f"{pos_rate:.1%} 过高，很可能一整天触发不了任何买入。"
                f"建议设到 {pos_rate * 1.5:.2f} 附近，"
                f"或改用 top_k 排序选股（设 use_rank=true）。")

    def on_start(self) -> None:
        self.today_entries = {}
        self.entry_prices = {}
        self._bar_buffers = {}

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """全市场截面推送 —— 核心决策入口。"""
        if not self.model or not bars:
            return

        first_bar = next(iter(bars.values()))
        now = first_bar.datetime
        t = now.strftime("%H:%M")

        # 尾盘强制平仓
        if t >= self.exit_time:
            self._close_all_intraday(bars)
            return

        # 未到入场时间
        if t < self.entry_time:
            return

        # 更新 bar 缓冲
        for vt, bar in bars.items():
            if vt not in self._bar_buffers:
                self._bar_buffers[vt] = []
            self._bar_buffers[vt].append({
                "open": bar.open_price, "high": bar.high_price,
                "low": bar.low_price, "close": bar.close_price,
                "volume": bar.volume, "amount": bar.turnover,
            })
            if len(self._bar_buffers[vt]) > 120:
                self._bar_buffers[vt] = self._bar_buffers[vt][-120:]

        # 计算全市场特征和概率
        scores = self._score_all(bars)
        if scores.empty:
            return

        # 按交易模式分发
        if self.trade_mode == "t_plus_0":
            self._execute_t0(scores, bars)
        elif self.trade_mode == "mean_reversion":
            self._execute_mean_reversion(scores, bars)
        elif self.trade_mode == "momentum":
            self._execute_momentum(scores, bars)

    def _score_all(self, bars: dict[str, BarData]) -> pd.DataFrame:
        """对当前截面所有标的打分。"""
        rows = []
        for vt, buf in self._bar_buffers.items():
            if len(buf) < 30:
                continue
            df = pd.DataFrame(buf, index=pd.date_range(
                end=bars[vt].datetime if vt in bars else pd.Timestamp.now(),
                periods=len(buf), freq="min"))
            feat = make_intraday_features(df)
            last = feat.iloc[-1:]
            x = last[self.features].fillna(0)
            prob = self.model.predict_proba(x)[0, 1]
            rows.append({
                "symbol": vt,
                "prob_up": prob,
                "close": float(df["close"].iloc[-1]),
                "day_ret": float(feat["day_ret"].iloc[-1]),
                "vol_z": float(feat["vol_z_30"].iloc[-1]),
                "vwap_gap": float(feat["vwap_gap"].iloc[-1]),
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("prob_up", ascending=False)

    def _execute_t0(self, scores: pd.DataFrame,
                    bars: dict[str, BarData]) -> None:
        """做T：在持有底仓的票上日内高抛低吸。"""
        for _, row in scores.iterrows():
            vt = row["symbol"]
            pos = self.get_pos(vt)
            price = row["close"]

            if pos <= 0:
                continue

            # 止损检查
            if vt in self.entry_prices:
                pnl = (price - self.entry_prices[vt]) / self.entry_prices[vt]
                if pnl < -self.max_intraday_loss:
                    vol = min(pos, self.position_size / price)
                    if vol > 0:
                        self.sell(vt, price, vol, OrderType.LIMIT)
                        self.write_log(f"T0 止损 {vt} pnl={pnl:.2%}")
                    continue

            if row["prob_up"] > self.prob_buy_threshold:
                vol = self.position_size / price
                if vol > 0:
                    self.buy(vt, price, vol, OrderType.LIMIT)
                    self.entry_prices[vt] = price
                    self.write_log(f"T0 加仓 {vt} prob={row['prob_up']:.3f}")
            elif row["prob_up"] < self.prob_sell_threshold:
                vol = min(pos, self.position_size / price)
                if vol > 0:
                    self.sell(vt, price, vol, OrderType.LIMIT)
                    self.write_log(f"T0 减仓 {vt} prob={row['prob_up']:.3f}")

    def _execute_mean_reversion(self, scores: pd.DataFrame,
                                bars: dict[str, BarData]) -> None:
        """均值回归：VWAP 偏离 + 模型过滤。"""
        active = len([v for v in self.today_entries.values() if v > 0])

        for _, row in scores.iterrows():
            vt = row["symbol"]
            price = row["close"]
            pos = self.get_pos(vt)
            vwap_gap = row["vwap_gap"]

            # 止损
            if vt in self.entry_prices and pos > 0:
                pnl = (price - self.entry_prices[vt]) / self.entry_prices[vt]
                if pnl < -self.max_intraday_loss:
                    self.sell(vt, price, pos, OrderType.LIMIT)
                    self.write_log(f"MR 止损 {vt} pnl={pnl:.2%}")
                    continue

            # 价格低于 VWAP + 模型看多 → 买入（预期回归均值上方）
            if (vwap_gap < -self.vwap_deviation
                    and row["prob_up"] > self.prob_buy_threshold
                    and pos <= 0
                    and active < self.max_positions):
                vol = self.position_size / price
                if vol > 0:
                    self.buy(vt, price, vol, OrderType.LIMIT)
                    self.entry_prices[vt] = price
                    self.today_entries[vt] = price
                    active += 1
                    self.write_log(f"MR 买入 {vt} vwap_gap={vwap_gap:.4f}")

            # 价格高于 VWAP + 模型看空 → 卖出
            elif (vwap_gap > self.vwap_deviation
                  and row["prob_up"] < self.prob_sell_threshold
                  and pos > 0):
                self.sell(vt, price, pos, OrderType.LIMIT)
                self.write_log(f"MR 卖出 {vt} vwap_gap={vwap_gap:.4f}")

    def _execute_momentum(self, scores: pd.DataFrame,
                          bars: dict[str, BarData]) -> None:
        """打板/追涨：高概率 + 量能放大。"""
        active = len([v for v in self.today_entries.values() if v > 0])

        for rank, (_, row) in enumerate(
                scores.head(self.max_positions * 2).iterrows()):
            vt = row["symbol"]
            price = row["close"]
            pos = self.get_pos(vt)

            # 止损
            if vt in self.entry_prices and pos > 0:
                pnl = (price - self.entry_prices[vt]) / self.entry_prices[vt]
                if pnl < -self.max_intraday_loss:
                    self.sell(vt, price, pos, OrderType.LIMIT)
                    self.write_log(f"MOM 止损 {vt} pnl={pnl:.2%}")
                    continue

            # 高概率 + 量能放大 + 日内正收益
            # use_rank 下，prob 条件由「进入前 max_positions 名」代替 ——
            # scores 已按 prob_up 降序，循环又只取前 max_positions*2 个，
            # 所以名次由 rank 变量给出
            prob_ok = (rank < self.max_positions if self.use_rank
                       else row["prob_up"] > self.prob_buy_threshold)
            if (prob_ok
                    and row["vol_z"] > self.vol_z_threshold
                    and row["day_ret"] > 0
                    and pos <= 0
                    and active < self.max_positions):
                vol = self.position_size / price
                if vol > 0:
                    self.buy(vt, price, vol, OrderType.LIMIT)
                    self.entry_prices[vt] = price
                    self.today_entries[vt] = price
                    active += 1
                    self.write_log(
                        f"MOM 买入 {vt} prob={row['prob_up']:.3f} "
                        f"vol_z={row['vol_z']:.1f}")

            # 已持仓但模型转空
            elif pos > 0 and row["prob_up"] < self.prob_sell_threshold:
                self.sell(vt, price, pos, OrderType.LIMIT)
                self.write_log(f"MOM 卖出 {vt} prob={row['prob_up']:.3f}")

    def _close_all_intraday(self, bars: dict[str, BarData]) -> None:
        """尾盘平掉所有日内仓位。"""
        for vt in list(self.today_entries.keys()):
            pos = self.get_pos(vt)
            if pos > 0 and vt in bars:
                self.sell(vt, bars[vt].close_price, pos, OrderType.LIMIT)
                self.write_log(f"尾盘平仓 {vt}")
        self.today_entries.clear()
        self.entry_prices.clear()
