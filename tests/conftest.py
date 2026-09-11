"""pytest 公共夹具。

## 为什么需要隔离 data/1d_raw

单测里大量用例用**合成价格 + 真实代码**（symbol="600519"、闭眼给 10 元）。
回测引擎会按 ``store_dir/1d_raw`` 找不复权价推导复权因子，而只要本机跑过一次
``scripts/download_adj_factor.py``，因子就会介入整手取整：合成价 10 配上
600519 的真实因子 6.2、或 000001 的 104，算出的「真实价」毫无意义，
委托量取整成 0 或不合理的值 —— 测 T+1、无前视、手续费、组合权重的用例
会随**机器状态**时红时绿，与被测代码无关。

所以默认把回测引擎的 ``_raw_dir`` 关掉，等价于「本机没有不复权数据」的
干净环境。需要专门测路径1的用例（见 tests/test_adj_factor.py 的
TestRawPricePathAlignsByDate），在构造后自行 ``engine._raw_dir = ...`` 覆盖即可。
"""
import pytest


@pytest.fixture(autouse=True)
def _hermetic_backtest_engine(monkeypatch):
    """回测引擎默认不读机器上的 data/1d_raw，让用例与机器数据解耦。"""
    import qmtquant.engine.backtest_engine as be

    original_init = be.BacktestEngine.__init__

    def init_without_raw_prices(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._raw_dir = None
        self._factor_cache.clear()

    monkeypatch.setattr(be.BacktestEngine, "__init__", init_without_raw_prices)
