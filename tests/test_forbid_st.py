"""ST 标的禁买。

## 起因：一个承诺了但不存在的保护

`forbid_st` 在 config.yaml 里写着 true、在 RiskConfig 里定义了、
被三个测试文件引用（都传 False），但 **RiskManager._do_check 从来没读过它**。
风控里只有黑名单检查，没有 ST 检查。

于是「实盘不碰 ST 股」这个保护根本不存在，而配置让人以为它存在。
这和 NotifyConfig 是同一种病（定义了、加载了、没人用），
但这个更危险 —— 它是风控，而且失效方向是**放行**。

顺带查出来的规模：1m 训练池里有 75 只标的在 2025-09~2026-09 期间是 ST。
模型在这些标的上训练、回测在这些标的上成交，而实盘本该一只都不碰。

## 为什么卖出不拦

拦卖出会让手上已有的 ST 股清不掉 —— 风控要出场就必须出得去。
这与回撤强平的设计一致（卖出始终放行）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from qmtquant.config import RiskConfig
from qmtquant.core.constants import Direction, Exchange, RejectReason
from qmtquant.core.objects import AccountData, OrderRequest
from qmtquant.risk.risk_manager import RiskManager


def _rm(forbid: bool = True, is_st=None, **kw) -> RiskManager:
    cfg = RiskConfig(
        forbid_st=forbid, blacklist=[], drawdown_enabled=False,
        max_order_value=1_000_000, max_position_ratio=1.0,
        max_total_position_ratio=1.0, max_order_count_per_day=10_000,
        max_turnover_per_day=1e12, **kw)
    rm = RiskManager(cfg, is_st=is_st)
    rm.account = AccountData(balance=1_000_000, available=1_000_000)
    return rm


def _req(vt: str, direction=Direction.LONG, volume=100, price=11.0):
    code, _, ex = vt.rpartition(".")
    return OrderRequest(symbol=code, exchange=Exchange[ex],
                        direction=direction, price=price, volume=volume)


ST_ONE = staticmethod(lambda vt, dt: vt == "600711.SSE")


class TestBuyBlocked:
    def test_st_buy_rejected(self):
        rm = _rm(is_st=lambda vt, dt: vt == "600711.SSE")
        ok, why = rm.check(_req("600711.SSE"))
        assert not ok
        assert why is RejectReason.ST_FORBIDDEN

    def test_non_st_buy_allowed(self):
        rm = _rm(is_st=lambda vt, dt: vt == "600711.SSE")
        ok, why = rm.check(_req("000001.SZSE"))
        assert ok, f"非 ST 标的不该被拦: {why}"

    def test_disabled_lets_st_through(self):
        """forbid_st=false 时不拦 —— 配置要真的能关掉。"""
        rm = _rm(forbid=False, is_st=lambda vt, dt: True)
        assert rm.check(_req("600711.SSE"))[0]


class TestSellNotBlocked:
    """拦卖出会让手上已有的 ST 股清不掉。风控要出场就必须出得去。"""

    def test_st_sell_not_rejected_for_st_reason(self):
        rm = _rm(is_st=lambda vt, dt: True)
        rm.positions = {}
        ok, why = rm.check(_req("600711.SSE", Direction.SHORT))
        # 可能因为「无持仓」被拒，但绝不该是 ST_FORBIDDEN
        assert why is not RejectReason.ST_FORBIDDEN


class TestDegradesLoudly:
    """判定器缺失或抛异常时放行，但必须留痕。

    静默放行会让「ST 拦截生效中」和「拦截器一直在抛异常」长得一模一样
    —— 那正是 forbid_st 原来的状态。
    """

    def test_no_checker_allows_but_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            rm = _rm(is_st=None)
        # 构造时若加载不到 ST 数据会告警；加载得到则不告警，两者都可接受
        assert rm.check(_req("000001.SZSE"))[0]

    def test_checker_exception_allows_and_logs(self, caplog):
        import logging

        def boom(vt, dt):
            raise RuntimeError("ST 数据损坏")

        rm = _rm(is_st=boom)
        with caplog.at_level(logging.WARNING):
            ok, why = rm.check(_req("600711.SSE"))
        assert ok, "判定失败应当放行而不是拦死所有买入"
        assert any("ST 判定失败" in r.message for r in caplog.records), \
            "判定失败必须留痕，否则和「拦截正常工作」无法区分"


class TestRealData:
    """用真实 ST 历史验证边界。"""

    @pytest.fixture(scope="class")
    def checker(self):
        from qmtquant.engine.backtest_engine import _load_st_checker

        f = _load_st_checker()
        if f is None:
            pytest.skip("缺 ST 历史数据")
        return f

    @pytest.mark.parametrize("date,expect", [
        ("2024-07-01", False),   # ST 之前
        ("2024-09-01", True),    # ST 期间
        ("2025-08-13", True),    # ST 最后一天
        ("2025-08-20", False),   # 摘帽之后
        ("2025-09-08", False),   # T+0 分析期起点
    ])
    def test_600711_st_window(self, checker, date, expect):
        """600711 的 ST 区间是 2024-08-01 ~ 2025-08-13。

        这条同时钉住一个我踩过的坑：我随手拿 2025-06-30 去测，
        看到 ST=True 就以为判定器坏了 —— 其实那个日期正落在区间内。
        边界要用明确的日期钉住，不能凭印象抽查。
        """
        assert checker("600711.SSE", pd.Timestamp(date)) is expect

    def test_analysis_period_is_not_st(self, checker):
        """T+0 分析期（2025-09-08 ~ 2026-09-04）全程非 ST。

        若哪天数据变了让这段变成 ST，那批研究的涨跌停口径就得重看。
        """
        for d in pd.date_range("2025-09-08", "2026-09-04", freq="MS"):
            assert not checker("600711.SSE", d), f"{d.date()} 变成 ST 了"
