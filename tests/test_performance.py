"""绩效统计测试。"""
import numpy as np
import pandas as pd
import pytest

from qmtquant.engine.performance import calculate_stats


class TestIntradayAnnualization:
    """真 bug 回归：分钟回测的绩效指标必须按自然日折算。

    曾把 58,434 根分钟 Bar 当成 58,434 个交易日，
    年化被开了 232 年的根号（真实 -6.70% 显示成 -0.03%），
    波动率、Sharpe、最长回撤天数同样全部失真，且不报错。
    """

    def _minute_equity(self, days: int = 10, bars_per_day: int = 240,
                       final: float = 1_100_000.0) -> pd.Series:
        n = days * bars_per_day
        idx = []
        for d in pd.bdate_range("2024-01-02", periods=days):
            idx += [d + pd.Timedelta(minutes=570 + i) for i in range(bars_per_day)]
        values = np.linspace(1_000_000.0, final, n)
        return pd.Series(values, index=pd.DatetimeIndex(idx))

    def test_trading_days_counts_dates_not_bars(self):
        eq = self._minute_equity(days=10)
        stats = calculate_stats(eq, [], 1_000_000.0)
        assert stats.trading_days == 10, "应为 10 个交易日而非 2400 根 Bar"

    def test_annual_return_not_diluted(self):
        """10 个交易日赚 10%，年化应远大于 10%，而不是被摊薄到接近 0"""
        eq = self._minute_equity(days=10, final=1_100_000.0)
        stats = calculate_stats(eq, [], 1_000_000.0)
        assert stats.total_return == pytest.approx(0.10)
        assert stats.annual_return > 1.0, (
            f"年化 {stats.annual_return:.4f} 被按 Bar 数摊薄了")

    def test_drawdown_duration_in_days(self):
        eq = self._minute_equity(days=10, final=900_000.0)
        stats = calculate_stats(eq, [], 1_000_000.0)
        assert stats.max_drawdown_duration <= 10, (
            f"回撤天数 {stats.max_drawdown_duration} 超过总交易日数")

    def test_max_drawdown_uses_full_resolution(self):
        """日内那一下真实发生过，最大回撤不应被日频重采样抹平"""
        idx = pd.DatetimeIndex([
            pd.Timestamp("2024-01-02 09:31"),
            pd.Timestamp("2024-01-02 11:00"),   # 日内暴跌
            pd.Timestamp("2024-01-02 15:00"),   # 收盘回到原位
        ])
        eq = pd.Series([1_000_000.0, 700_000.0, 1_000_000.0], index=idx)
        stats = calculate_stats(eq, [], 1_000_000.0)
        assert stats.max_drawdown == pytest.approx(-0.30), (
            "日内回撤被抹平，风险被低估")

    def test_daily_backtest_unchanged(self):
        """日线回测下重采样必须是恒等变换"""
        idx = pd.bdate_range("2024-01-02", periods=50)
        eq = pd.Series(np.linspace(1_000_000, 1_200_000, 50), index=idx)
        stats = calculate_stats(eq, [], 1_000_000.0)
        assert stats.trading_days == 50
