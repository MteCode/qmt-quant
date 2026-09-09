"""后复权价与真实价的换算，以及依赖它的过滤/取整。

## 起因：一条把便宜蓝筹当成高价股排除的过滤

规则是「1 手买入超过 5 万的排除」，即**真实价** > 500 元。
代码写的是 `feat["close"].iloc[-1] > 500`，而 close 是**后复权价**。

本地行情锚在 IPO 复权，因子从 1.6（工行）到 280（万科A）不等，
于是这条过滤排掉的是：

    海尔智家  真实 21.15 元（1 手 2,115 元）  后复权 906.8
    新和成    真实 26.46 元（1 手 2,646 元）  后复权 948.0
    北方稀土  真实 40.07 元（1 手 4,007 元）  后复权 897.7

实测 500 只里误排 11 只，而真正该排的只有 1 只 —— 错排的比对排的多 10 倍。
**训练和回测用的是同一条过滤，所以模型从一开始就没见过这批标的。**

同一个根因还有第二处：整手取整。一手是 100 **真实**股，
按后复权价算手数，高因子的标的会被算成不足一手而静默跳过
（实测 2 万/票时剔除 10.3%）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from qmtquant.datafeed.adjust import LOT_SIZE, adj_factor, real_price


def _df(close: float, real: float, lots: float = 10_000.0,
        amount_col: str = "amount") -> pd.DataFrame:
    """volume 按手计，amount 按元计 —— 仓库数据的约定。"""
    return pd.DataFrame({
        "close": [close] * 3,
        "volume": [lots] * 3,
        amount_col: [real * lots * LOT_SIZE] * 3,
    })


#: 四只真实标的，因子已独立核对
REAL = [
    ("工商银行", 13.18, 8.152, 1.6),
    ("贵州茅台", 8288.98, 1326.10, 6.3),
    ("盛屯矿业", 114.18, 11.786, 9.7),
    ("平安银行", 1241.60, 11.910, 104.2),
    ("万科A", 879.10, 3.13, 280.9),
]


class TestRealPrice:
    @pytest.mark.parametrize("name,adj,real,_f", REAL)
    def test_recovers_real_price(self, name, adj, real, _f):
        got = real_price(_df(adj, real))
        assert got == pytest.approx(real, rel=1e-6), name

    def test_accepts_turnover_column(self):
        """BarData 用 turnover，parquet 用 amount。两个都要认。"""
        assert real_price(_df(100, 10, amount_col="turnover")) == \
            pytest.approx(10.0)

    @pytest.mark.parametrize("bad", [
        pd.DataFrame(),
        pd.DataFrame({"close": [10.0]}),                       # 缺量额
        pd.DataFrame({"close": [10.0], "volume": [0.0], "amount": [1.0]}),
        pd.DataFrame({"close": [10.0], "volume": [1.0], "amount": [0.0]}),
    ])
    def test_missing_or_zero_returns_none(self, bad):
        """返回 None 比返回一个猜的数好：调用方能据此选保守行为，
        而一个错的价格会静默改变过滤与仓位。"""
        assert real_price(bad) is None


class TestAdjFactor:
    @pytest.mark.parametrize("name,adj,real,expect", REAL)
    def test_matches_known(self, name, adj, real, expect):
        f = adj_factor(_df(adj, real))
        assert f is not None, name
        assert f == pytest.approx(expect, rel=0.02), name

    def test_never_below_one(self):
        """后复权价按定义不低于真实价。"""
        for _n, adj, real, _e in REAL:
            assert adj_factor(_df(adj, real)) >= 1.0

    def test_intraday_noise_clamped_to_one(self):
        """真实价用的是当日均价，收盘价可能略低于它 ——
        这会算出 0.9x 的因子。夹到 1，不当成异常。"""
        assert adj_factor(_df(9.5, 10.0)) == pytest.approx(1.0)

    def test_absurd_factor_rejected(self):
        """算出天文数字说明数据有问题，不猜。"""
        assert adj_factor(_df(1e9, 1.0)) is None

    def test_wrong_direction_rejected(self):
        """真实价远高于后复权价 —— 单位搞错了，宁可返回 None。"""
        assert adj_factor(_df(10.0, 1000.0)) is None


class TestHighPriceFilter:
    """「1 手 > 5 万」这条规则必须作用在真实价上。"""

    MAX_PRICE = 500.0

    @pytest.mark.parametrize("name,adj,real,_f", REAL)
    def test_filter_follows_real_price(self, name, adj, real, _f):
        px = real_price(_df(adj, real))
        excluded = px > self.MAX_PRICE
        assert excluded == (real * LOT_SIZE > 50_000), (
            f"{name} 真实价 {real}，1 手 {real*LOT_SIZE:.0f} 元，"
            f"按真实价判定应当{'排除' if real*LOT_SIZE > 50_000 else '保留'}")

    def test_blue_chips_survive(self):
        """海尔智家/新和成/北方稀土这类老蓝筹不该被排除。"""
        for name, adj, real in (("海尔智家", 906.8, 21.15),
                                ("新和成", 948.0, 26.46),
                                ("北方稀土", 897.7, 40.07)):
            assert adj > self.MAX_PRICE, f"{name} 的后复权价确实 > 500"
            assert real_price(_df(adj, real)) <= self.MAX_PRICE, \
                f"{name} 真实 1 手只要 {real*100:.0f} 元，不该被当成高价股"

    def test_genuinely_expensive_still_excluded(self):
        """茅台真实价 1326 元，1 手 13.3 万 —— 这个该排除。

        反面用例：修复不能变成「什么都不排」。
        """
        assert real_price(_df(8288.98, 1326.10)) > self.MAX_PRICE


class TestLotRounding:
    """一手是 100 **真实**股。"""

    @pytest.mark.parametrize("name,adj,real,_f", REAL)
    def test_affordable_stocks_get_at_least_one_lot(self, name, adj,
                                                    real, _f):
        per_size = 100_000.0
        f = adj_factor(_df(adj, real))
        real_vol = int(per_size / adj * f / LOT_SIZE) * LOT_SIZE
        if real * LOT_SIZE <= per_size:
            assert real_vol >= LOT_SIZE, (
                f"{name} 一手 {real*LOT_SIZE:.0f} 元，{per_size:.0f} 买得起"
                f"却被算成 {real_vol} 股")

    def test_money_is_preserved(self):
        """换算回后复权口径后，price * vol 仍是正确的金额。"""
        adj, real = 1241.60, 11.910
        f = adj_factor(_df(adj, real))
        per_size = 100_000.0
        real_vol = int(per_size / adj * f / LOT_SIZE) * LOT_SIZE
        vol = real_vol / f
        spent_adjusted = adj * vol
        spent_real = real * real_vol
        assert spent_adjusted == pytest.approx(spent_real, rel=1e-6)
        assert spent_adjusted <= per_size

    def test_naive_rounding_drops_high_factor_stocks(self):
        """反证：不换算就直接取整，平安银行在 2 万/票时被算成 0 股。

        这条不测生产代码，是把「为什么必须换算」钉成可执行的事实。
        """
        adj, real = 1241.60, 11.910
        per_size = 20_000.0
        naive = int(per_size / adj / LOT_SIZE) * LOT_SIZE
        assert naive == 0
        f = adj_factor(_df(adj, real))
        correct = int(per_size / adj * f / LOT_SIZE) * LOT_SIZE
        assert correct >= LOT_SIZE
        assert real * LOT_SIZE < per_size      # 真实一手只要 1191 元
