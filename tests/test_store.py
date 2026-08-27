"""状态持久化测试。

核心场景是「进程崩溃后重启」：策略内部状态（指标窗口、计数器）必须能恢复，
而持仓与资金**不能**从本地恢复 —— 券商是唯一真相来源。
"""
from datetime import datetime

import pytest

from qmtquant.core.constants import Direction, Exchange, Status
from qmtquant.core.objects import OrderData, TradeData
from qmtquant.store.database import StateStore
from qmtquant.strategy.base import StrategyBase


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "state.db")


class DemoStrategy(StrategyBase):
    variables = ["inited", "trading", "pos", "bar_count", "last_signal"]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bar_count = 0
        self.last_signal = ""


def make_strategy(name="S1"):
    return DemoStrategy(engine=None, strategy_name=name,
                        vt_symbols=["000001.SZSE"])


class TestStrategyState:
    def test_roundtrip(self, store):
        store.save_state("S1", {"bar_count": 42, "pos": {"000001.SZSE": 500}})
        loaded = store.load_state("S1")
        assert loaded["bar_count"] == 42
        assert loaded["pos"]["000001.SZSE"] == 500

    def test_missing_returns_empty(self, store):
        assert store.load_state("nope") == {}

    def test_save_overwrites_not_merges(self, store):
        """整体覆盖：删掉的字段不能残留，否则重启会读到早已失效的旧值"""
        store.save_state("S1", {"a": 1, "b": 2})
        store.save_state("S1", {"a": 9})
        loaded = store.load_state("S1")
        assert loaded == {"a": 9}

    def test_unserializable_field_skipped(self, store):
        """有字段存不下时不应整体失败，其余字段仍要落库"""
        store.save_state("S1", {"ok": 1, "bad": object()})
        loaded = store.load_state("S1")
        assert loaded["ok"] == 1
        assert "bad" not in loaded

    def test_updated_at_recorded(self, store):
        store.save_state("S1", {"a": 1})
        assert store.state_updated_at("S1") is not None
        assert store.state_updated_at("nope") is None

    def test_clear_and_list(self, store):
        store.save_state("S1", {"a": 1})
        store.save_state("S2", {"a": 1})
        assert store.list_strategies() == ["S1", "S2"]
        store.clear_state("S1")
        assert store.list_strategies() == ["S2"]


class TestRestoreVariables:
    def test_restores_declared_fields(self):
        s = make_strategy()
        s.restore_variables({"bar_count": 77, "last_signal": "up",
                             "pos": {"000001.SZSE": 300}})
        assert s.bar_count == 77
        assert s.last_signal == "up"
        assert s.pos["000001.SZSE"] == 300

    def test_never_restores_trading_flag(self):
        """从磁盘读回 trading=True 会让策略在未初始化时就开始交易"""
        s = make_strategy()
        s.restore_variables({"trading": True, "inited": True})
        assert s.trading is False
        assert s.inited is False

    def test_ignores_undeclared_fields(self):
        s = make_strategy()
        s.restore_variables({"bar_count": 5, "sneaky": 999})
        assert s.bar_count == 5
        assert not hasattr(s, "sneaky")

    def test_partial_state_ok(self):
        s = make_strategy()
        s.bar_count = 3
        s.restore_variables({"last_signal": "down"})
        assert s.last_signal == "down"
        assert s.bar_count == 3

    def test_full_crash_recovery_cycle(self, store):
        """完整的崩溃恢复：跑一段 -> 存盘 -> 新建实例 -> 恢复"""
        s1 = make_strategy()
        s1.bar_count = 128
        s1.last_signal = "up"
        s1.pos = {"000001.SZSE": 700}
        store.save_state("S1", s1.get_variables())

        # 模拟进程重启：全新实例，状态归零
        s2 = make_strategy()
        assert s2.bar_count == 0

        s2.restore_variables(store.load_state("S1"))
        assert s2.bar_count == 128
        assert s2.last_signal == "up"
        assert s2.pos["000001.SZSE"] == 700
        # 生命周期标志仍须由引擎设置
        assert s2.trading is False


def make_trade(tradeid="T1", volume=100.0):
    return TradeData(
        symbol="000001", exchange=Exchange.SZSE, orderid="O1", tradeid=tradeid,
        direction=Direction.LONG, price=11.6, volume=volume, commission=5.0,
        datetime=datetime(2026, 8, 27, 14, 36), reference="S1",
        gateway_name="MINIQMT")


def make_order(orderid="O1", traded=0.0, status=Status.NOTTRADED):
    return OrderData(
        symbol="000001", exchange=Exchange.SZSE, orderid=orderid,
        direction=Direction.LONG, price=11.62, volume=100.0, traded=traded,
        status=status, datetime=datetime(2026, 8, 27, 14, 36),
        reference="S1", gateway_name="MINIQMT")


class TestTradeLog:
    def test_save_and_load(self, store):
        store.save_trade(make_trade())
        rows = store.load_trades()
        assert len(rows) == 1
        assert rows[0]["vt_symbol"] == "000001.SZSE"
        assert rows[0]["price"] == 11.6

    def test_idempotent_by_tradeid(self, store):
        """断线重连会重复推送同一笔成交，去重失败会让对账金额翻倍"""
        for _ in range(3):
            store.save_trade(make_trade("SAME"))
        assert len(store.load_trades()) == 1

    def test_filter_by_date(self, store):
        store.save_trade(make_trade("T1"))
        assert len(store.load_trades("2026-08-27")) == 1
        assert len(store.load_trades("2026-08-26")) == 0


class TestOrderLog:
    def test_status_updates_in_place(self, store):
        """同一委托多次状态更新应覆盖，而不是堆成多行"""
        store.save_order(make_order(traded=0, status=Status.NOTTRADED))
        store.save_order(make_order(traded=100, status=Status.ALLTRADED))
        rows = store.load_orders()
        assert len(rows) == 1
        assert rows[0]["traded"] == 100
        assert rows[0]["status"] == Status.ALLTRADED.value

    def test_multiple_orders_kept(self, store):
        store.save_order(make_order("O1"))
        store.save_order(make_order("O2"))
        assert len(store.load_orders()) == 2


class TestEquity:
    def test_same_day_overwrites(self, store):
        """盘中会多次写入当日净值，应覆盖而非追加"""
        store.save_equity(1_000_000, 900_000, 100_000, "2026-08-27")
        store.save_equity(1_010_000, 890_000, 120_000, "2026-08-27")
        rows = store.load_equity()
        assert len(rows) == 1
        assert rows[0]["balance"] == 1_010_000

    def test_range_query(self, store):
        for d in ["2026-08-25", "2026-08-26", "2026-08-27"]:
            store.save_equity(1_000_000, 1_000_000, 0, d)
        assert len(store.load_equity(start="2026-08-26")) == 2
        assert len(store.load_equity(end="2026-08-25")) == 1


class TestSummary:
    def test_counts(self, store):
        store.save_state("S1", {"a": 1})
        store.save_trade(make_trade())
        store.save_order(make_order())
        store.save_equity(1_000_000, 1_000_000, 0)
        s = store.summary()
        assert s["strategies"] == 1
        assert s["trades"] == 1
        assert s["orders"] == 1
        assert s["equity_days"] == 1

    def test_reopen_persists(self, tmp_path):
        """真正的持久化：关掉再打开数据仍在"""
        path = tmp_path / "s.db"
        StateStore(path).save_state("S1", {"bar_count": 9})
        assert StateStore(path).load_state("S1")["bar_count"] == 9


class TestPeriodicSave:
    """硬崩溃兜底：进程被强杀时 finally 不执行，
    只靠「成交时保存 + 优雅退出保存」会丢掉自上次成交以来的全部状态。"""

    def test_timer_saves_state(self, store, monkeypatch):
        import time as _time

        from qmtquant.config import CostConfig, RiskConfig
        from qmtquant.engine import live_engine as le
        from qmtquant.engine.live_engine import LiveEngine
        from qmtquant.event.engine import Event, EventEngine
        from qmtquant.gateway.sim_gateway import SimGateway
        from qmtquant.risk.risk_manager import RiskManager

        ee = EventEngine(timer_interval=10)          # 不自动触发，手工调用
        gw = SimGateway(ee, "SIM", cost=CostConfig())
        engine = LiveEngine(ee, gw, RiskManager(RiskConfig()), store=store)

        s = engine.add_strategy(DemoStrategy, "S1", ["000001.SZSE"])
        s.bar_count = 55
        assert store.load_state("S1") == {}, "尚未触发保存"

        # 把上次保存时间推到很久以前，模拟已过定时间隔
        engine._last_state_save = _time.time() - le.STATE_SAVE_INTERVAL - 1
        engine._on_timer(Event("eTimer"))

        assert store.load_state("S1")["bar_count"] == 55

    def test_timer_does_not_save_too_often(self, store):
        from qmtquant.config import CostConfig, RiskConfig
        from qmtquant.engine.live_engine import LiveEngine
        from qmtquant.event.engine import Event, EventEngine
        from qmtquant.gateway.sim_gateway import SimGateway
        from qmtquant.risk.risk_manager import RiskManager

        ee = EventEngine(timer_interval=10)
        gw = SimGateway(ee, "SIM", cost=CostConfig())
        engine = LiveEngine(ee, gw, RiskManager(RiskConfig()), store=store)
        engine.add_strategy(DemoStrategy, "S1", ["000001.SZSE"])

        engine._on_timer(Event("eTimer"))
        assert store.load_state("S1") == {}, "间隔未到不应保存"
