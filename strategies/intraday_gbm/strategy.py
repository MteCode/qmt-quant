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
        "t_plus_1",             # A股T+1：当日买入不可卖，止损顺延到次日
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
        #: A 股股票必须开着。关掉只适用于 T+0 品种（ETF、可转债）
        self.t_plus_1 = True

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
        # today_entries 是「今天买的」，换日/重启后自然清空
        self.today_entries = {}
        self._bar_buffers = {}
        # entry_prices 不清：T+1 下隔夜仓要靠它兜底做止损。
        # 正常情况下 stop_ref_price 会优先用券商成本价，这里只是退路。
        if not hasattr(self, "entry_prices") or self.entry_prices is None:
            self.entry_prices = {}

    def sellable(self, vt: str) -> float:
        """能卖多少 —— 所有卖出路径（止损/信号/尾盘）都必须先过这道闸。

        A 股 T+1：当日买入部分被冻结，按持仓量直接下卖单会被券商逐笔拒。
        实盘出现过每分钟发单、每分钟被「可卖数量不足」拒掉的情况，
        连 -2.4% 的止损单都执行不了 —— 日志刷满拒单，实际什么也没做。

        与其发出去让券商拒，不如自己先判断：卖不了就不发，日志才有信噪比。
        """
        pos = self.get_pos(vt)
        if pos <= 0:
            return 0.0
        if not self.t_plus_1:
            return pos
        return max(0.0, min(pos, self.get_available(vt)))

    def stop_ref_price(self, vt: str) -> float:
        """止损参考价：优先用券商成本价，退回自记的入场价。

        隔夜仓必须靠成本价 —— 策略每天重启会清空 entry_prices，
        只认内存的话隔夜仓就完全失去止损保护，而那恰恰是风险最大的仓位。
        """
        return self.get_cost_price(vt) or self.entry_prices.get(vt, 0.0)

    def warmup(self, history: dict[str, list[dict]]) -> int:
        """用当日已发生的 1m bar 预热缓冲区，返回预热成功的标的数。

        `_score_all` 要求每只标的攒够 30 根 bar 才打分，而缓冲区只从实时
        推送累积。不预热的话盘中任何时刻启动都要再等 30 分钟才可能出第一个
        信号 —— 盘中重启一次就等于半小时不交易，且不报任何错。

        只接受当日 bar：day_ret、日内位置、时间编码这几维特征都以当天开盘
        为基准，掺进昨天的 bar 会把特征算错，比不预热更糟。
        """
        n = 0
        for vt, rows in history.items():
            if not rows:
                continue
            self._bar_buffers[vt] = list(rows[-120:])
            n += 1
        if n:
            ready = sum(1 for b in self._bar_buffers.values() if len(b) >= 30)
            self.write_log(f"预热 {n} 只标的，其中 {ready} 只已满足 30 根 bar")
        return n

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

            # 止损检查。做 T 模式卖的本就是昨仓，sellable 正好等于可回转的量
            ref = self.stop_ref_price(vt)
            if ref > 0:
                pnl = (price - ref) / ref
                if pnl < -self.max_intraday_loss:
                    vol = min(self.sellable(vt), self.position_size / price)
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
                vol = min(self.sellable(vt), self.position_size / price)
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

            # 止损（T+1 下当日买入卖不掉，自然顺延到次日）
            ref = self.stop_ref_price(vt)
            if ref > 0 and pos > 0:
                pnl = (price - ref) / ref
                if pnl < -self.max_intraday_loss:
                    vol = self.sellable(vt)
                    if vol > 0:
                        self.sell(vt, price, vol, OrderType.LIMIT)
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
                vol = self.sellable(vt)
                if vol > 0:
                    self.sell(vt, price, vol, OrderType.LIMIT)
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

            # 止损。T+1 下当日买入的卖不掉，止损自然顺延到次日 ——
            # 参考价用券商成本价，保证隔夜仓跨重启仍受保护
            ref = self.stop_ref_price(vt)
            if ref > 0 and pos > 0:
                pnl = (price - ref) / ref
                if pnl < -self.max_intraday_loss:
                    vol = self.sellable(vt)
                    if vol > 0:
                        self.sell(vt, price, vol, OrderType.LIMIT)
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
                vol = self.sellable(vt)
                if vol > 0:
                    self.sell(vt, price, vol, OrderType.LIMIT)
                    self.write_log(f"MOM 卖出 {vt} prob={row['prob_up']:.3f}")

    def _close_all_intraday(self, bars: dict[str, BarData]) -> None:
        """尾盘清掉**可卖**的持仓。

        T+1 下这不再是「日内平仓」：当日买入的卖不掉，能平的只有昨仓。
        原先的实现按 today_entries 遍历并按持仓量下单，结果是每只都被
        券商以「可卖数量不足」拒掉 —— 声称的「不留隔夜」从来没做到过。

        因此这里改为遍历全部持仓、只卖 sellable 的部分，并且**不再清空
        entry_prices**：今天买的要留到明天才能卖，清掉就失去止损参考了。
        """
        closed = 0
        for vt in list(set(list(self.today_entries) + list(self.pos))):
            if vt not in bars:
                continue
            vol = self.sellable(vt)
            if vol > 0:
                self.sell(vt, bars[vt].close_price, vol, OrderType.LIMIT)
                self.write_log(f"尾盘平仓 {vt} 量={vol:.0f}")
                closed += 1
                self.today_entries.pop(vt, None)
                self.entry_prices.pop(vt, None)

        held = [v for v in self.pos if self.get_pos(v) > 0]
        if held:
            self.write_log(
                f"尾盘：平掉 {closed} 只，{len(held)} 只因 T+1 留隔夜")
