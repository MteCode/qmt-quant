"""回测绩效计算与报告。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: A 股一年约 242 个交易日
TRADING_DAYS = 242


@dataclass
class PerformanceStats:
    """绩效指标"""
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0
    initial_capital: float = 0
    final_capital: float = 0
    total_return: float = 0
    annual_return: float = 0
    max_drawdown: float = 0
    max_drawdown_duration: int = 0
    sharpe_ratio: float = 0
    calmar_ratio: float = 0
    volatility: float = 0
    total_trades: int = 0
    win_rate: float = 0
    profit_factor: float = 0
    total_commission: float = 0
    turnover_rate: float = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def summary(self) -> str:
        return "\n".join([
            "=" * 46,
            f"回测区间      : {self.start_date} ~ {self.end_date} ({self.trading_days} 交易日)",
            f"初始资金      : {self.initial_capital:,.2f}",
            f"期末资金      : {self.final_capital:,.2f}",
            f"总收益率      : {self.total_return * 100:.2f}%",
            f"年化收益率    : {self.annual_return * 100:.2f}%",
            f"最大回撤      : {self.max_drawdown * 100:.2f}%",
            f"最长回撤天数  : {self.max_drawdown_duration}",
            f"年化波动率    : {self.volatility * 100:.2f}%",
            f"Sharpe        : {self.sharpe_ratio:.3f}",
            f"Calmar        : {self.calmar_ratio:.3f}",
            f"成交笔数      : {self.total_trades}",
            f"胜率          : {self.win_rate * 100:.2f}%",
            f"盈亏比        : {self.profit_factor:.3f}",
            f"累计手续费    : {self.total_commission:,.2f}",
            "=" * 46,
        ])


def calculate_stats(equity: pd.Series, trades: list, initial_capital: float,
                    risk_free_rate: float = 0.02) -> PerformanceStats:
    """从净值曲线与成交明细计算绩效指标。

    :param equity: index 为日期的每日总资产序列
    :param trades: TradeData 列表
    """
    stats = PerformanceStats(initial_capital=initial_capital)
    if equity.empty:
        return stats

    equity = equity.sort_index()
    stats.start_date = str(equity.index[0])[:10]
    stats.end_date = str(equity.index[-1])[:10]
    stats.trading_days = len(equity)
    stats.final_capital = float(equity.iloc[-1])
    stats.total_return = stats.final_capital / initial_capital - 1

    years = max(stats.trading_days / TRADING_DAYS, 1e-9)
    # 期末资金可能为负（理论上不该发生），此时年化无意义，直接取总收益
    if stats.final_capital > 0:
        stats.annual_return = (stats.final_capital / initial_capital) ** (1 / years) - 1
    else:
        stats.annual_return = -1.0

    # 回撤
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    stats.max_drawdown = float(drawdown.min())
    # 最长回撤持续天数：连续处于水下的最长长度
    underwater = (drawdown < 0).astype(int)
    longest, cur = 0, 0
    for v in underwater:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    stats.max_drawdown_duration = longest

    daily_return = equity.pct_change().dropna()
    if len(daily_return) > 1:
        stats.volatility = float(daily_return.std() * np.sqrt(TRADING_DAYS))
        excess = daily_return.mean() * TRADING_DAYS - risk_free_rate
        stats.sharpe_ratio = float(excess / stats.volatility) if stats.volatility else 0.0
    if stats.max_drawdown < 0:
        stats.calmar_ratio = stats.annual_return / abs(stats.max_drawdown)

    stats.total_trades = len(trades)
    stats.total_commission = float(sum(getattr(t, "commission", 0) for t in trades))

    # 按标的 FIFO 配对计算单次盈亏，得到胜率与盈亏比
    pnls = _pair_trade_pnl(trades)
    if pnls:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        stats.win_rate = len(wins) / len(pnls)
        gross_loss = abs(sum(losses))
        stats.profit_factor = sum(wins) / gross_loss if gross_loss else float("inf")

    turnover = sum(t.price * t.volume for t in trades)
    stats.turnover_rate = turnover / initial_capital if initial_capital else 0
    return stats


def _pair_trade_pnl(trades: list) -> list[float]:
    """FIFO 配对买卖成交，返回每次平仓的净盈亏（已扣费）"""
    from ..core.constants import Direction

    queues: dict[str, list[list]] = {}
    pnls: list[float] = []

    for t in sorted(trades, key=lambda x: x.datetime):
        q = queues.setdefault(t.vt_symbol, [])
        if t.direction == Direction.LONG:
            # [剩余数量, 成本价, 分摊费用单价]
            fee_per_share = t.commission / t.volume if t.volume else 0
            q.append([t.volume, t.price, fee_per_share])
            continue

        remaining = t.volume
        sell_fee_per_share = t.commission / t.volume if t.volume else 0
        while remaining > 0 and q:
            lot = q[0]
            matched = min(lot[0], remaining)
            pnls.append((t.price - lot[1]) * matched
                        - lot[2] * matched - sell_fee_per_share * matched)
            lot[0] -= matched
            remaining -= matched
            if lot[0] <= 0:
                q.pop(0)
    return pnls
