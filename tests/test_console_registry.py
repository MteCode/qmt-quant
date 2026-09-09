"""管理台注册表的守卫测试。

## 为什么需要

`webui/strategies.py` 的 STRATEGIES 是手写列表。跑完一个实验不补注册项，
**不会有任何报错** —— 页面照常渲染，只是少一块。靠这个静默失败模式，
仓库里一度积压了 5 个结果目录（t0_constrained、t0_market、t0_analysis、
t0_downday、allocation）没接进管理台，直到用户直接问起才发现。

所以这里把「有产物就必须有人认领」变成一条会红的断言。
"""
from __future__ import annotations

import pytest

from qmtquant.core.costs import DEFAULT_COST
from webui.strategies import (
    BY_ID,
    STRATEGIES,
    ROOT,
    stale_cost_strategies,
    unregistered_outputs,
)


class TestNoOrphanOutputs:
    def test_every_result_dir_is_claimed(self):
        """磁盘上每个结果目录都要有策略认领。

        这条红了说明你刚跑完一个实验但没往 STRATEGIES 里加注册项。
        修法不是删目录，是补一个 Strategy(...) 和它的 loader ——
        补的时候要真的去读产物字段，别照抄别的 loader：
        每个实验的 max_drawdown 正负号、annual_return 含不含底仓 beta
        都不一样，抄错会把亏损显示成盈利。
        """
        orphans = unregistered_outputs()
        assert not orphans, (
            "以下目录有产物但没有策略认领：\n  "
            + "\n  ".join(f"{o['dir']}（{o['n_files']} 个文件，"
                          f"{o['mtime']}）" for o in orphans)
        )

    def test_discovery_actually_detects(self, tmp_path, monkeypatch):
        """自动发现本身要能抓到缺口。

        上一条测试「通过」有两种可能：真的没有缺口，或者判定写坏了
        永远返回空。这两者在测试结果里长得一模一样。
        实现里就真的出过一次 —— out.append 被留在 continue 之后，
        函数恒返回 []，而「0 个缺口」看起来像是干净的。
        """
        probe = ROOT / "models" / "_pytest_probe_dir"
        probe.mkdir(parents=True, exist_ok=True)
        (probe / "summary.json").write_text("{}", encoding="utf-8")
        try:
            found = [o["dir"] for o in unregistered_outputs()]
            assert any("_pytest_probe_dir" in d for d in found), (
                "注入了一个未注册的结果目录，但 unregistered_outputs() "
                "没有报出来 —— 判定过松或实现有误")
        finally:
            (probe / "summary.json").unlink(missing_ok=True)
            probe.rmdir()


class TestRegistryIntegrity:
    def test_ids_unique(self):
        ids = [s.id for s in STRATEGIES]
        assert len(ids) == len(set(ids)), f"策略 id 重复: {ids}"
        assert len(BY_ID) == len(STRATEGIES)

    @pytest.mark.parametrize("s", STRATEGIES, ids=lambda s: s.id)
    def test_loader_never_raises(self, s):
        """loader 抛异常会被 Strategy.result() 吞成 error，页面变空白。
        这里直接调，让它在测试里炸出来。"""
        r = s.result()
        assert isinstance(r, dict)
        assert "error" not in r, f"{s.id} 的 loader 抛了: {r.get('error')}"

    @pytest.mark.parametrize("s", STRATEGIES, ids=lambda s: s.id)
    def test_output_dir_exists_if_declared(self, s):
        if not s.output_dir:
            pytest.skip("未声明 output_dir")
        assert (ROOT / s.output_dir).is_dir(), \
            f"{s.id} 声明的 output_dir 不存在: {s.output_dir}"

    @pytest.mark.parametrize("s", STRATEGIES, ids=lambda s: s.id)
    def test_metrics_are_numbers_or_none(self, s):
        """metrics 里只能是数字或 None，不能是字符串。

        字符串会让模板的 |pct 过滤器吐出「—」，看起来像「没测过」，
        而实际上是有值但类型错了 —— 一种查起来很久的静默失真。
        """
        m = (s.result() or {}).get("metrics") or {}
        for k, v in m.items():
            assert v is None or isinstance(v, (int, float)), \
                f"{s.id}.metrics[{k}] 类型是 {type(v).__name__}: {v!r}"

    @pytest.mark.parametrize("s", STRATEGIES, ids=lambda s: s.id)
    def test_drawdown_is_negative(self, s):
        """回撤统一为负数。各脚本的符号约定不一致，
        不统一的话页面会把 -27% 显示成 +27%。"""
        m = (s.result() or {}).get("metrics") or {}
        dd = m.get("max_drawdown")
        if dd is not None:
            assert dd <= 0, f"{s.id} 的 max_drawdown 是正数: {dd}"

    def test_research_entries_state_their_conclusion(self):
        """研究类策略必须写 caveat。

        这批实验的结论全是否定的（做 T 无 alpha）。不写 caveat，
        页面上就只剩一串数字，读的人会自己脑补成正面结果。
        """
        for s in STRATEGIES:
            if s.category == "研究":
                assert s.caveat.strip(), f"{s.id} 是研究类但没写 caveat"


class TestStaleCostDetection:
    def test_stale_cost_is_surfaced(self):
        """用旧成本模型跑出来的结果必须被标出来。

        研究结果是快照，成本模型会改。2026-09 那次修正把印花税从
        0.001 改成 0.0005、佣金从万2.5 改成万0.854。之前跑的结果偏悲观，
        和新结果并排显示而不加区分，等于拿两套成本互相比较。
        """
        stale = stale_cost_strategies()
        for x in stale:
            assert x["drift"]["fields"], f"{x['id']} 标了 drift 但没有差异字段"
            assert x["drift"]["note"]

    def test_current_cost_is_the_reference(self):
        """drift 比较的基准必须是当前成本模型，不是某个写死的数。"""
        stale = stale_cost_strategies()
        for x in stale:
            for k, v in x["drift"]["fields"].items():
                expect = {
                    "commission": DEFAULT_COST.commission_rate,
                    "stamp_tax": DEFAULT_COST.stamp_tax_rate,
                    "slippage": DEFAULT_COST.slippage_rate,
                }[k]
                assert v["current"] == expect

    def test_no_drift_means_matching_cost(self):
        """没标 drift 的策略，要么没记 cost_model，要么真的一致。"""
        for s in STRATEGIES:
            r = s.result()
            if r.get("cost_drift"):
                continue
            cost = r.get("cost") or {}
            if "stamp_tax" in cost:
                assert float(cost["stamp_tax"]) == pytest.approx(
                    DEFAULT_COST.stamp_tax_rate), \
                    f"{s.id} 的成本与当前不一致却没标 drift"


class TestFieldSemantics:
    """字段语义是这类 loader 最容易出错的地方，而且错了不报错。

    真实踩过的：t0_downday 的 lookahead 列是 pandas 写出的布尔值，
    落盘成字符串 "True"/"False" 而不是 0/1。用数值解析会全部得到 None，
    1440 个实时可实现的变体被整批误判成含未来函数。
    页面上表现为「年化 —」，看起来像没数据；而如果判反了方向，
    就会拿前视结果当可实现结果展示。
    """

    def test_downday_splits_lookahead_correctly(self):
        r = BY_ID["t0_downday"].result()
        live, look = r["realtime"], r["lookahead"]
        assert live and look, "两组都必须非空 —— 有一组空说明分组判据错了"
        assert live["n"] > look["n"], \
            "实时可实现的变体应当远多于含未来函数的"
        assert live["n"] + look["n"] == r["n_variants"]

    def test_downday_lookahead_looks_better(self):
        """前视版本必然更好看 —— 这正是不能用它下结论的原因。

        这条断言反过来也是个哨兵：如果哪天实时版本反而更好，
        说明分组判据被写反了。
        """
        r = BY_ID["t0_downday"].result()
        live, look = r["realtime"], r["lookahead"]
        assert look["t_annual_median"] > live["t_annual_median"], \
            "含未来函数的中位数应当优于实时可实现的；反了说明分组搞错"

    def test_allocation_headline_is_not_the_maximum(self):
        """搜索类结果的代表值不能取全域最优。

        7776 组里挑最好的一组，年化 +17.8%（还含底仓 beta），
        显示在结论是「无 alpha」的卡片上，是纯粹的误导。
        """
        r = BY_ID["t0_allocation"].result()
        best = (r.get("best_t") or {}).get("annual_return")
        headline = r["metrics"]["annual_return"]
        assert headline is None, "搜索没有单一年化，应当留空而不是取最优"
        if best is not None:
            assert r["alpha_beta"]["t_annual"] < best, \
                "代表值不能等于全域最优"

    def test_analysis_has_no_fabricated_metrics(self):
        """归因分析没跑回测，就不该有 metrics。"""
        r = BY_ID["t0_analysis"].result()
        assert r.get("kind") == "analysis"
        assert not r.get("metrics"), "分析型产物不应编造回测指标"
        assert r.get("equity") is None
