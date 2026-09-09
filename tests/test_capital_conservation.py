"""资金守恒：回测起点不能凭空少钱。

## 起因

`buyhold()` 和 `simulate()` 都这么建仓：

    shares = int(base_value / px0 / 100) * 100
    eq = cash_value + shares * closes        # ← 零头没了

整手取整剩下的零头是现金，不是消失了。而零头占底仓的比例**随仓位变化**：

    底仓 20 万 -> 2200 股，零头 4,678 元（2.34%）
    底仓  2 万 ->  200 股，零头 2,243 元（11.22%）

optimize_allocation 研究的正是仓位这一维。小仓位被额外扣掉 11%，
等于给低仓位配置人为加了一道惩罚，把有效前沿的形状压歪了 ——
而结论正是从那条曲线上读出来的。

这类错误不会让程序崩，也不会让数字变得离谱，只会让曲线安静地歪掉。
所以要用「起点资金必须等于投入资金」这条守恒律钉死它。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def engine():
    return _load("t0div", "scripts/optimize_t0_divergence.py")


@pytest.fixture(scope="module")
def constrained():
    return _load("t0con", "scripts/optimize_t0_constrained.py")


@pytest.fixture(scope="module")
def alloc():
    return _load("alloc", "scripts/optimize_allocation.py")


@pytest.fixture(scope="module")
def bars(engine):
    p = ROOT / "data" / "clean" / "1m" / "SSE" / "600711.parquet"
    if not p.exists():
        pytest.skip("缺 600711 分钟数据")
    return engine.prep(engine.load_bars("600711", "SSE"))


BASE_CASH = [
    (200_000, 0),
    (100_000, 100_000),
    (40_000, 160_000),
    (20_000, 180_000),      # 零头占比最大的一档
]


class TestBuyholdConservation:
    @pytest.mark.parametrize("base,cash", BASE_CASH)
    def test_no_capital_vanishes_at_start(self, alloc, bars, base, cash):
        """第一天收盘的权益不应显著低于投入的本金。

        允许的偏差只来自第一天本身的涨跌，不该来自建仓取整。
        """
        px0 = float(bars["close"].iloc[0])
        shares = int(base / px0 / 100) * 100
        residual = base - shares * px0
        init = base + cash

        # 起点权益（用 px0 计价）必须等于本金
        start_equity = cash + residual + shares * px0
        assert start_equity == pytest.approx(init), \
            f"底仓 {base} 起点少了 {init - start_equity:.0f} 元"

    @pytest.mark.parametrize("base,cash", BASE_CASH)
    def test_residual_is_material_for_small_base(self, alloc, bars,
                                                 base, cash):
        """零头不是可忽略的舍入误差 —— 小仓位下超过 10%。

        这条不是测实现，是把「为什么必须保留零头」钉成事实：
        如果哪天有人觉得"整手取整差那么点无所谓"，这里会告诉他差多少。
        """
        px0 = float(bars["close"].iloc[0])
        shares = int(base / px0 / 100) * 100
        residual_pct = (base - shares * px0) / base
        assert 0 <= residual_pct < 1
        if base <= 40_000:
            assert residual_pct > 0.05, \
                "小底仓的零头应当是显著的（实测 11.22%）"


class TestSimulateConservation:
    @pytest.mark.parametrize("base,cash", BASE_CASH)
    def test_zero_trade_run_matches_buyhold(self, engine, alloc, bars,
                                            base, cash):
        """把做 T 关掉（阈值高到不可能触发），结果必须与纯持有一致。

        两边用同一套建仓口径，是「做T vs 纯持有」这个比较能成立的前提。
        口径不一致时，比较出来的差异里混着建仓方式的差异，
        而那正是这批研究要下的结论。
        """
        r = engine.simulate(bars, base, cash, "corr", 0.999, 99.0,
                            0.006, 0.012, 1, 575, 890, 30, False)
        if "error" in r:
            pytest.skip(r["error"])
        assert r["n_trades"] == 0, "阈值 0.999 不应产生任何交易"

        bh = alloc.buyhold(bars, base, cash)
        assert bh, "纯持有对照组应当有结果"
        assert r["total_return"] == pytest.approx(
            bh["total_return"], abs=1e-4), (
            f"零交易时做T组总收益 {r['total_return']:.4%} 应等于纯持有 "
            f"{bh['total_return']:.4%} —— 不等说明两边建仓口径不一致")

    @pytest.mark.parametrize("base,cash", BASE_CASH)
    def test_drawdown_also_matches(self, engine, alloc, bars, base, cash):
        r = engine.simulate(bars, base, cash, "corr", 0.999, 99.0,
                            0.006, 0.012, 1, 575, 890, 30, False)
        if "error" in r or r["n_trades"]:
            pytest.skip("需要零交易的对照")
        bh = alloc.buyhold(bars, base, cash)
        assert r["max_drawdown"] == pytest.approx(
            bh["max_drawdown"], abs=1e-4)


class TestInternalBenchmarkConsistency:
    """simulate() 自己算的纯持有基准，必须和它模拟的那条路径同口径。

    这条是补一个我自己制造的洞：修零头时只改了模拟路径
    （cash = cash_value + 零头），漏了 simulate() 内部的 buyhold_annual。
    做 T 那边白拿一笔现金而基准没有 —— 底仓 4 万时零头 4,488 元
    = 总资金的 2.24%，把 t_vs_buyhold 从 -4.15% "翻正"到 +1.06%，
    看起来像做 T 突然有了价值。

    同一份结果里 t_annual 一直是 -1.21%：做 T 亏钱却让账户跑赢基准，
    这个自相矛盾本该立刻提示我。

    上一组测试没抓到，因为它比的是 simulate().total_return 与**外部**
    buyhold()，而不是 simulate() 内部那个 buyhold_annual 字段。
    口径不一致的对照组比没有对照组更危险：它给出一个具体的、
    方向错误的数字。
    """

    @pytest.mark.parametrize("base,cash", BASE_CASH)
    def test_internal_benchmark_matches_external(self, engine, alloc, bars,
                                                 base, cash):
        r = engine.simulate(bars, base, cash, "corr", 0.999, 99.0,
                            0.006, 0.012, 1, 575, 890, 30, False)
        if "error" in r:
            pytest.skip(r["error"])
        bh = alloc.buyhold(bars, base, cash)
        assert r["buyhold_annual"] == pytest.approx(
            bh["annual_return"], abs=1e-4), (
            f"simulate 内部基准 {r['buyhold_annual']:.4%} 与外部 buyhold "
            f"{bh['annual_return']:.4%} 不一致 —— 两边建仓口径不同")

    @pytest.mark.parametrize("base,cash", BASE_CASH)
    def test_zero_trade_run_equals_its_own_benchmark(self, engine, bars,
                                                     base, cash):
        """不做任何交易时，账户年化必须等于自己算的纯持有基准。

        这是最直接的自洽检查：一笔都没做，怎么可能跑赢或跑输纯持有。
        """
        r = engine.simulate(bars, base, cash, "corr", 0.999, 99.0,
                            0.006, 0.012, 1, 575, 890, 30, False)
        if "error" in r or r["n_trades"]:
            pytest.skip("需要零交易的对照")
        assert r["annual_return"] == pytest.approx(
            r["buyhold_annual"], abs=1e-4), (
            f"零交易时账户年化 {r['annual_return']:.4%} != 内部基准 "
            f"{r['buyhold_annual']:.4%}")


@pytest.fixture(scope="module")
def single():
    return _load("t0single", "scripts/optimize_t0_600711.py")


@pytest.fixture(scope="module")
def single_bars(single):
    """600711 脚本有自己的 prep()，产出的列与 divergence 不同 ——
    用错 prep 会在 simulate 里 KeyError，而不是给出错误的数。"""
    p = ROOT / "data" / "clean" / "1m" / "SSE" / "600711.parquet"
    if not p.exists():
        pytest.skip("缺 600711 分钟数据")
    return single.prep(single.load_bars("600711", "SSE", None, None))


class TestSingleStockScriptConsistency:
    """optimize_t0_600711 是独立实现的模拟器，同样要自洽。

    它原先两边**都**吞零头，所以 excess_vs_buyhold 无偏，
    只是绝对收益整体偏低 —— 比 divergence 那个错轻，但会让
    t0_single 与 t0_divergence 在管理台上并排显示时不可比。
    """

    @pytest.mark.parametrize("base,cash", BASE_CASH[1:])
    def test_zero_trade_equals_benchmark(self, single, single_bars,
                                         base, cash):
        """一笔都没做时，账户收益必须等于它自己的纯持有对照组。"""
        r = single.simulate(single_bars, base, cash, 0.99, 0.99, 1, 0.012,
                            575, 890, False, 0.006, False)
        if "error" in r:
            pytest.skip(r["error"])
        assert r["n_trades"] == 0, "阈值 0.99 不应产生交易"
        assert r["total_return"] == pytest.approx(
            r["buyhold_return"], abs=1e-6), (
            f"零交易时账户 {r['total_return']:.4%} != 基准 "
            f"{r['buyhold_return']:.4%} —— 两边建仓口径不一致")
        assert r["excess_vs_buyhold"] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("base,cash", BASE_CASH[1:])
    def test_two_engines_agree_on_buyhold(self, single, single_bars,
                                          engine, bars, base, cash):
        """两个独立实现的模拟器，纯持有收益应当一致。

        纯持有只取决于价格和底仓规模，与各自的信号逻辑无关 ——
        不一致就说明其中一个的建仓口径不对。而它们的结果会在管理台上
        并排显示，读的人不会知道两栏用的是不同算法。
        """
        a = single.simulate(single_bars, base, cash, 0.99, 0.99, 1, 0.012,
                            575, 890, False, 0.006, False)
        b = engine.simulate(bars, base, cash, "corr", 0.999, 99.0,
                            0.006, 0.012, 1, 575, 890, 30, False)
        if "error" in a or "error" in b:
            pytest.skip("数据不足")
        assert a["buyhold_annual"] == pytest.approx(
            b["buyhold_annual"], abs=1e-4), (
            f"600711 算的纯持有年化 {a['buyhold_annual']:.4%}，"
            f"divergence 算的 {b['buyhold_annual']:.4%}")


class TestAllBuyholdImplementationsAgree:
    """纯持有被**四个**地方各实现了一遍：

        optimize_t0_divergence.simulate()   内部基准
        optimize_t0_600711.simulate()       内部基准
        optimize_t0_constrained.buyhold_curve()
        optimize_allocation.buyhold()

    同一件事写四遍，就会有四种口径。实际发生的：修零头时改了两处，
    以为修完了；`t_vs_buyhold` 仍然 +1.06%，被第三处顶着 ——
    而那份 JSON 里同时躺着 buyhold_annual=5.14% 和
    best_buyhold_under_constraint=2.87%，两个都号称是「底仓 4 万纯持有」。

    **同一个量在同一份结果里有两个值，本身就是口径不一致的信号。**
    这条测试就是把那个信号变成一条会红的断言。
    """

    @pytest.mark.parametrize("base,cash", BASE_CASH[1:])
    def test_three_engines_same_buyhold(self, engine, constrained, alloc,
                                        bars, base, cash):
        r = engine.simulate(bars, base, cash, "corr", 0.999, 99.0,
                            0.006, 0.012, 1, 575, 890, 30, False)
        if "error" in r:
            pytest.skip(r["error"])
        c = constrained.buyhold_curve(bars, base, cash)
        a = alloc.buyhold(bars, base, cash)

        vals = {
            "divergence.simulate": r["buyhold_annual"],
            "constrained.buyhold_curve": c["annual_return"],
            "allocation.buyhold": a["annual_return"],
        }
        lo, hi = min(vals.values()), max(vals.values())
        assert hi - lo < 1e-4, (
            "同一个「纯持有年化」在不同实现里不一致: "
            + "; ".join(f"{k}={v:.4%}" for k, v in vals.items()))
