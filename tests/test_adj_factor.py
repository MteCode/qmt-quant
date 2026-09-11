"""复权因子与整手取整。

## 起因：一条死掉的通道，静默剔掉了一批蓝筹

`_adj_factor()` 原先只从 `data/1d_raw/` 读不复权价。而那个目录在本仓库
**并不存在** —— 通道对所有标的恒返回 None，静默退回「按后复权价取整」。

本地行情是后复权价且锚在 IPO，因子差异极大：

    工商银行 1.6   贵州茅台 6.2   盛屯矿业 9.7   平安银行 104

整手约束作用在**真实股数**上。按后复权价取整会把高因子的标的算成
不足一手，实测沪深300：

    10 万/只 -> 剔除 33 只（11.0%）
     5 万/只 -> 剔除 69 只（23.0%）
     2 万/只 -> 剔除 123 只（41.0%）

被剔掉的正是平安银行、云南白药、泸州老窖、格力、五粮液这类老蓝筹
（上市久、分红送转多、因子高）。于是每个组合回测都系统性偏向小盘股，
而且不报任何错。

## 单位歧义怎么解决的

推真实价需要 turnover / (volume * 单位)，而 volume 是按股还是按手，
BarData 契约里没规定。单看一只股票两种都说得通 ——
平安银行按股算因子 1.04、按手算 104，都不离谱。

全市场一起看就不歧义：后复权价按定义不低于真实价，所以
「隐含真实价 > 后复权价」不可能。实测 579 只：按股 99.7% 违反，
按手 8.8% 违反（残余那部分是因子≈1 的新股，收盘价与日均价的日内噪声）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from qmtquant.core.constants import Exchange, Interval
from qmtquant.core.objects import BarData
from qmtquant.engine.backtest_engine import BacktestEngine


def _bar(code: str, ex: Exchange, day: int, adj_close: float,
         real_price: float, lots: float = 10_000.0) -> BarData:
    """造一根 bar：volume 按**手**计，turnover 按元计。"""
    return BarData(
        symbol=code, exchange=ex,
        datetime=pd.Timestamp(f"2026-09-{day:02d}").to_pydatetime(),
        interval=Interval.DAILY,
        open_price=adj_close, high_price=adj_close,
        low_price=adj_close, close_price=adj_close,
        volume=lots, turnover=real_price * lots * 100)


def _engine(specs: list[tuple[str, float, float]]) -> BacktestEngine:
    """specs: [(代码, 后复权价, 真实价)]"""
    bars = []
    for i, (code, adj, real) in enumerate(specs):
        ex = Exchange.SSE if code.startswith("6") else Exchange.SZSE
        for d in (3, 4):
            bars.append(_bar(code, ex, d, adj, real))
    e = BacktestEngine(initial_capital=1_000_000)
    e.load_data(bars)
    return e


#: 真实的四只，因子已独立核对过
REAL = [
    ("601398", 13.18, 8.152),      # 工商银行  1.6
    ("600519", 8288.98, 1326.10),  # 贵州茅台  6.2
    ("600711", 114.18, 11.786),    # 盛屯矿业  9.7
    ("000001", 1241.60, 11.910),   # 平安银行  104
]


class TestVolumeUnitResolution:
    def test_resolves_to_lots(self):
        e = _engine(REAL * 6)      # 凑够 min_cohort
        assert e._resolve_volume_unit() == 100

    def test_cached(self):
        e = _engine(REAL * 6)
        assert e._resolve_volume_unit() is e._resolve_volume_unit()

    def test_small_cohort_defaults_to_lots(self):
        """样本太少时判据不可靠，用仓库数据的约定。"""
        e = _engine(REAL[:1])
        assert e._resolve_volume_unit() == 100

    def test_share_unit_data_detected(self):
        """如果哪天数据换成按股计，判据要能认出来。"""
        bars = []
        for i in range(30):
            code = f"6{i:05d}"
            adj, real = 100.0, 10.0
            bars.append(BarData(
                symbol=code, exchange=Exchange.SSE,
                datetime=pd.Timestamp("2026-09-04").to_pydatetime(),
                interval=Interval.DAILY,
                open_price=adj, high_price=adj, low_price=adj,
                close_price=adj,
                volume=1_000_000,              # 股
                turnover=real * 1_000_000))
        e = BacktestEngine(initial_capital=1_000_000)
        e.load_data(bars)
        assert e._resolve_volume_unit() == 1


class TestFactorDerivation:
    @pytest.mark.parametrize("code,adj,real", REAL)
    def test_matches_known_factor(self, code, adj, real):
        e = _engine(REAL * 6)
        ex = "SSE" if code.startswith("6") else "SZSE"
        f = e._adj_factor(f"{code}.{ex}")
        assert f is not None, f"{code} 应当能推出因子"
        assert f == pytest.approx(adj / real, rel=0.01)

    def test_factor_at_least_one(self):
        """后复权价按定义不低于真实价，因子必 >= 1。"""
        e = _engine(REAL * 6)
        for code, adj, real in REAL:
            ex = "SSE" if code.startswith("6") else "SZSE"
            assert e._adj_factor(f"{code}.{ex}") >= 1.0

    def test_absurd_factor_rejected(self):
        """算出天文数字说明数据有问题，不猜 —— 退回并计数。"""
        e = _engine(REAL * 6 + [("600999", 1e9, 1.0)])
        assert e._adj_factor("600999.SSE") is None
        assert e.factor_fallbacks >= 1

    def test_missing_data_returns_none(self):
        e = _engine(REAL * 6)
        assert e._adj_factor("999999.SSE") is None


class TestLotRoundingKeepsBlueChips:
    """这是整件事的目的：高因子的蓝筹不该被静默剔除。"""

    @pytest.mark.parametrize("code,adj,real", REAL)
    def test_affordable_by_real_price(self, code, adj, real):
        """按真实价算，10 万买得起一手（100 股）的，就不该被剔掉。"""
        e = _engine(REAL * 6)
        ex = "SSE" if code.startswith("6") else "SZSE"
        cap = 100_000.0
        f = e._adj_factor(f"{code}.{ex}")
        real_shares = int((cap / adj) * f // 100) * 100

        affordable = real * 100 <= cap
        if affordable:
            assert real_shares >= 100, (
                f"{code} 真实价 {real}，一手 {real*100:.0f} 元，"
                f"10 万买得起却被算成 0 股")

    def test_maotai_genuinely_unaffordable(self):
        """茅台一手 13.3 万，10 万确实买不起 —— 这时剔除是**对的**。

        这条是反面用例：修复不能变成「什么都不剔」。
        """
        e = _engine(REAL * 6)
        f = e._adj_factor("600519.SSE")
        shares = int((100_000 / 8288.98) * f // 100) * 100
        assert shares == 0
        assert 1326.10 * 100 > 100_000


class TestRiskExitLotRounding:
    """回撤强平路径原先完全没有整手取整。

    会发出 137 股这种委托，实盘被券商拒掉，回测却照单成交 ——
    回测里的「减仓能力」强于实际。

    这几条**必须真正驱动 _enforce_drawdown**，不能自己复算一遍公式。
    第一版我就是复算的，结果把强平取整整段删掉，测试照样全绿 ——
    复算公式的测试只能证明我会算术，证明不了代码做了这件事。
    """

    def _engine_with_position(self, volume: float, available: float | None = None):
        from qmtquant.risk.drawdown import DrawdownController, DrawdownLevel

        e = _engine(REAL * 6)
        e.drawdown = DrawdownController()
        e.positions = {"601398.SSE": {
            "volume": volume,
            "available": volume if available is None else available,
            "price": 13.18,
        }}
        e.pending_orders = []
        e._last_enforced_level = DrawdownLevel.NORMAL
        return e

    def _force_reduce(self, e, ratio: float = 0.3):
        """把回撤控制器推到「强制减仓」档并执行一次。

        用**真的** DrawdownController，只改它的档位 —— 第一版我写了个
        _Stub，结果 stub 缺 target_position_ratio() 直接 AttributeError。
        自己造的替身会和真实接口悄悄脱节，测出来的东西就不作数了。
        """
        from qmtquant.risk.drawdown import DrawdownLevel

        e.drawdown.config.reduce_keep_ratio = ratio
        e.drawdown.state.level = DrawdownLevel.REDUCE
        bars = {"601398.SSE": _bar("601398", Exchange.SSE, 4, 13.18, 8.152)}
        e._enforce_drawdown(bars)
        return [o for o in e.pending_orders
                if o.reference == e.RISK_REFERENCE]

    def test_partial_reduction_is_whole_lots(self):
        """部分减仓的委托，换算成真实股数后必须是 100 的整数倍。"""
        e = self._engine_with_position(volume=1000.0)
        orders = self._force_reduce(e, ratio=0.3)
        assert orders, "应当发出减仓委托"
        f = e._adj_factor("601398.SSE") or 1.0
        real = orders[0].volume * f
        assert abs(real - round(real / 100) * 100) < 1e-6, (
            f"减仓 {orders[0].volume} (真实 {real:.2f} 股) 不是整手 —— "
            f"实盘会被拒单")

    def test_odd_lot_position_still_reducible(self):
        """持仓本身带零股时也要能减，不能因为取整变成 0 而放弃减仓。"""
        e = self._engine_with_position(volume=1537.0)
        orders = self._force_reduce(e, ratio=0.3)
        assert orders, "带零股的持仓也应当能减仓"
        assert orders[0].volume > 0

    def test_reduction_does_not_exceed_sellable(self):
        """T+1：当日买入的部分卖不掉。取整不能把量放大到超过可卖数。"""
        e = self._engine_with_position(volume=2000.0, available=500.0)
        orders = self._force_reduce(e, ratio=0.1)
        assert orders
        assert orders[0].volume <= 500.0 + 1e-9

    def test_tiny_position_not_forced_into_invalid_order(self):
        """小到不足一手的减仓量，宁可不发单也不能发非法委托。"""
        e = self._engine_with_position(volume=50.0)
        orders = self._force_reduce(e, ratio=0.9)
        for o in orders:
            f = e._adj_factor("601398.SSE") or 1.0
            real = o.volume * f
            assert real >= 100 - 1e-6 or abs(real - 50.0 * f) < 1e-6


class TestRawPricePathAlignsByDate:
    """路径1（data/1d_raw）必须取与后复权末根**同日**的不复权价。

    不复权文件常比 data/1d/ 新（今天仍在更新，后复权滞后一两天）。
    原先直接取 raw.iloc[-1]，与后复权末根不是同一天 —— 两道之间若跨了
    除权日，相除得到的因子就是错的，而且错得无声。

    这是 verify() 全量扫末根日期才暴露出来的：实测下载当日，
    600519 不复权到 20260911、后复权到 20260904。
    """

    def _engine_with_raw(self, tmp_path, raw_closes: dict[str, float]):
        e = _engine(REAL * 6)          # 末根为 2026-09-04
        (tmp_path / "SSE").mkdir()
        s = pd.Series(raw_closes, dtype=float)
        s.index = pd.Index(list(s.index))      # 'YYYYMMDD' 字符串，与真实数据一致
        s.sort_index().to_frame("close").to_parquet(tmp_path / "SSE" / "600519.parquet")
        e._raw_dir = tmp_path
        e._factor_cache.clear()
        return e

    def test_uses_same_day_row_not_last_row(self, tmp_path):
        """末根是不对齐的 20260911，必须仍取 20260904 那根。"""
        e = self._engine_with_raw(tmp_path, {
            "20260904": 1300.00,       # 与后复权末根同日 —— 应取这根
            "20260911": 1340.00,       # 更新的、不对齐的一根 —— 不能取
        })
        f = e._adj_factor("600519.SSE")
        assert f == pytest.approx(8288.98 / 1300.00, rel=1e-6), (
            "取错日期了：8288.98/1300=6.376，误取 20260911 则得 6.186")

    def test_no_aligned_row_does_not_cross_dates(self, tmp_path):
        """不复权全在末根之后 —— 宁可退回路径2，也不跨日相除。"""
        e = self._engine_with_raw(tmp_path, {"20260911": 1340.00})
        f = e._adj_factor("600519.SSE")
        # 退回路径2 = turnover/volume 反推的真实价 1326.10
        assert f == pytest.approx(8288.98 / 1326.10, rel=1e-3)
        assert f != pytest.approx(8288.98 / 1340.00, rel=1e-3)

