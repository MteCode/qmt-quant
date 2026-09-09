"""滚动 walk-forward 的结构正确性。

## 起因：一个被我说大了的证据

原实现是**单次 50/50 切分**：前半段跑全部参数、按样本内排序取前 10、
在同一个后半段上各测一次，报「0/10 存活」。

那个数字看起来像 10 次独立检验，其实是 **10 组参数 x 1 个样本外区间**。
而且前 10 组往往是同一参数区域的邻域点（实测 t0_single 的 10 组里
momentum/max_trades/sell_gap 完全相同，只在另外 3 个参数上微调），
它们一起失败本来就该预期，不构成 10 个独立证据。

我拿这个数当主要结论讲了很多次，描述成「10 折」。这里把它做对：
n 折滚动，每折独立选参、检验区间互不重叠，「几折为正」才是几次独立观测。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def engine():
    spec = importlib.util.spec_from_file_location(
        "t0div", ROOT / "scripts" / "optimize_t0_divergence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bars(engine):
    p = ROOT / "data" / "clean" / "1m" / "SSE" / "600711.parquet"
    if not p.exists():
        pytest.skip("缺 600711 分钟数据")
    return engine.prep(engine.load_bars("600711", "SSE"))


COMBOS = [
    ("corr", 0.3, 1.0, 0.010, 0.012, 1, 60, True),
    ("corr", 0.1, 1.0, 0.010, 0.008, 1, 60, False),
    ("pulse", 0.01, 1.0, 0.010, 0.012, 2, 60, True),
]


@pytest.fixture(scope="module")
def folds(engine, bars):
    return engine.rolling_walkforward(
        bars, COMBOS, 100_000, 100_000, 575, 890, n_folds=5)


class TestFoldStructure:
    def test_produces_requested_folds(self, folds):
        assert len(folds) == 5

    def test_test_windows_do_not_overlap(self, folds):
        """检验区间不重叠 —— 这是「几折为正 = 几次独立观测」的前提。

        重叠的话同一段行情会被数很多次，分母虚高。
        原实现就是这个毛病：10 个「折」共用同一个样本外区间。
        """
        spans = []
        for f in folds:
            lo, hi = f["test"].split("~")
            spans.append((pd.Timestamp(lo), pd.Timestamp(hi)))
        spans.sort()
        for (a_lo, a_hi), (b_lo, b_hi) in zip(spans, spans[1:]):
            assert a_hi < b_lo, f"检验区间重叠: {a_hi} vs {b_lo}"

    def test_train_precedes_its_own_test(self, folds):
        """训练必须在检验之前 —— 否则就是拿未来选参数。"""
        for f in folds:
            tr_hi = pd.Timestamp(f["train"].split("~")[1])
            te_lo = pd.Timestamp(f["test"].split("~")[0])
            assert tr_hi < te_lo, f"第 {f['fold']} 折训练没有早于检验"

    def test_each_fold_picks_independently(self, folds):
        """每折独立选参。若所有折选出同一组参数，说明选参没起作用
        （或者参数空间太小）—— 这条不强制不同，只要求字段存在，
        由下一条断言检查实质。"""
        for f in folds:
            assert f["signal"] in ("corr", "obv", "pulse")
            assert f["n_train_days"] > 0 and f["n_test_days"] > 0

    def test_only_top_one_per_fold(self, folds):
        """每折只取样本内第 1 名。

        取前 10 会把同一参数区域的邻域点重复计数，让「0/N」的 N 虚高。
        这条通过「折数 == 请求的 n_folds」间接保证：
        若每折取 10 个，len(folds) 会是 50。
        """
        assert len(folds) == 5


class TestFoldsAreInformative:
    """只钉**结构**，不钉统计性质。

    我先后写了两条统计断言，两条都因为测试用的参数空间太小而误报：

      「样本内应当为正」  -> 3 组参数里最好的一组仍为负，完全合理
      「样本内优于样本外」-> 3 组根本没法过拟合（实测 2/5），
                            48 组才有 5/5

    这两条测的是**数据的性质**，不是代码的性质。过拟合的强弱取决于
    搜索空间大小和标的，随数据变；把它写成断言，只会在换数据时误报，
    然后被人加 skip 或放宽阈值，最后测试变成摆设。

    结论性的统计（0/5 折为正）属于分析报告，不属于单元测试。
    """

    def test_out_of_sample_actually_traded(self, folds):
        """样本外必须真的产生了交易。

        零交易的「样本外为负」是空的 —— 什么都没做怎么会亏。
        这条是结构性的：它保证每一折都构成一次真实的观测。
        """
        for f in folds:
            assert f["out_sample_trades"] > 0,                 f"第 {f['fold']} 折样本外零交易，这一折不构成证据"

    def test_reports_both_sides(self, folds):
        """每折都要同时给出样本内与样本外 —— 只报一边无法判断过拟合。"""
        for f in folds:
            assert "in_sample_t_annual" in f
            assert "out_sample_t_annual" in f
            assert isinstance(f["in_sample_t_annual"], float)
            assert isinstance(f["out_sample_t_annual"], float)


class TestGuards:
    def test_too_few_days_returns_empty(self, engine, bars):
        """数据不够分折时返回空，而不是硬切出没有意义的折。"""
        small = bars[bars["day"] <= bars["day"].unique()[20]]
        assert engine.rolling_walkforward(
            small, COMBOS, 100_000, 100_000, 575, 890, n_folds=5) == []
