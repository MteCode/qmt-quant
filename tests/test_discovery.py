"""策略自动发现的测试。

背景见 webui/discovery.py 与 webui/manifest.py 的模块 docstring。要点：

- 发现层**只追加、不替换**核心清单（否则会静默丢掉没有 manifest 的实验页）
- 一个坏 manifest / 一个 import 失败的策略文件，只跳过它自己
- manifest 声明的脚本必须落在项目目录内（安全边界）

风格与 tests/ 一致：类式组织，tmp_path 隔离，不碰真实仓库数据。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from webui import discovery, manifest
from webui.manifest import ResultSpec, StrategySpec


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------- manifest

class TestManifestParsing:
    def test_flat_keys_untouched(self):
        """现有扁平键必须原样可用 —— paths.py::load_params() 还读它们。"""
        raw = yaml.safe_load(
            "market: csi1000\nindex: '000852.SH'\nhold_k: 10\n"
            "rebalance_days: 20\ncapital: 500000\n")
        spec = manifest.parse_webui_block(raw, "alstm_ppo_csi1000")
        assert spec.id == "alstm_ppo_csi1000"
        # 默认分类**不能**是「研究」—— 那会让没写 caveat 的新策略把守卫测试打红
        assert spec.category == manifest.DEFAULT_CATEGORY
        assert spec.dir == "alstm_ppo_csi1000"
        assert spec.output_dir == "strategies/alstm_ppo_csi1000/backtest"

    def test_webui_block_overrides(self):
        raw = yaml.safe_load("""
market: csi1000
webui:
  id: my_id
  name: 我的策略
  category: 日频
  how: [a, b]
  output_dir: strategies/x/out
  backtest_task: train_x
  results:
    kind: summary
    source_json: r.json
    metrics: {sharpe: nested.sharpe}
  tasks:
    - {id: t1, name: T1, script: a.py,
       params: [{name: n, kind: int, default: 3}]}
""")
        spec = manifest.parse_webui_block(raw, "x")
        assert spec.id == "my_id" and spec.name == "我的策略"
        assert spec.category == "日频"
        assert spec.how == ["a", "b"]
        assert spec.result.source_json == "r.json"
        assert spec.result.metrics == {"sharpe": "nested.sharpe"}
        assert spec.tasks[0].id == "t1"
        assert spec.tasks[0].params[0].kind == "int"
        assert spec.tasks[0].params[0].default == 3

    def test_non_dict_webui_ignored(self):
        """webui 写错成标量时按「没写」处理，不让整份清单失效。"""
        spec = manifest.parse_webui_block({"webui": "oops"}, "d")
        assert spec.id == "d"

    def test_incomplete_task_skipped(self):
        raw = {"webui": {"tasks": [{"id": "no_script"}, "not_a_dict"]}}
        spec = manifest.parse_webui_block(raw, "d")
        assert spec.tasks == []

    def test_real_yamls_still_parse(self):
        base = discovery.ROOT / "strategies"
        seen = 0
        for d in ("alstm_ppo_csi1000", "lgb_agents_ppo"):
            y = base / d / "strategy.yaml"
            if not y.exists():
                continue
            raw = yaml.safe_load(y.read_text(encoding="utf-8"))
            assert manifest.parse_webui_block(raw, d).id == d
            seen += 1
        if not seen:
            pytest.skip("仓库里没有 strategy.yaml")


# --------------------------------------------------------------- 项目发现

class TestProjectDiscovery:
    def test_scans_strategy_yaml(self, tmp_path):
        _write(tmp_path / "strategies" / "demo" / "strategy.yaml",
               "webui:\n  name: 演示\n")
        assert [s.id for s in discovery.discover_projects(tmp_path)] == ["demo"]

    def test_missing_webui_uses_dir_defaults(self, tmp_path):
        _write(tmp_path / "strategies" / "bare" / "strategy.yaml", "market: x\n")
        spec = discovery.discover_projects(tmp_path)[0]
        assert spec.dir == "bare"
        assert spec.category == manifest.DEFAULT_CATEGORY

    def test_broken_yaml_skipped_others_survive(self, tmp_path):
        _write(tmp_path / "strategies" / "bad" / "strategy.yaml", "a: [unclosed\n")
        _write(tmp_path / "strategies" / "ok" / "strategy.yaml", "webui: {}\n")
        assert [s.id for s in discovery.discover_projects(tmp_path)] == ["ok"]

    def test_convention_tasks_generated(self, tmp_path):
        _write(tmp_path / "strategies" / "p" / "strategy.yaml", "webui: {}\n")
        _write(tmp_path / "strategies" / "p" / "train_foo.py", "x = 1\n")
        _write(tmp_path / "strategies" / "p" / "sweep_bar.py", "x = 1\n")
        _write(tmp_path / "strategies" / "p" / "paper_trade.py", "x = 1\n")
        _write(tmp_path / "strategies" / "p" / "helper.py", "x = 1\n")
        proj = discovery.discover_projects(tmp_path)[0]
        tasks = {t.id: t for t in discovery.project_tasks(proj, tmp_path)}
        assert "p.train_foo" in tasks and "p.sweep_bar" in tasks
        assert tasks["p.paper_trade"].dangerous is True
        assert not any("helper" in k for k in tasks), "非约定脚本不该被列成任务"

    def test_explicit_task_wins_over_convention(self, tmp_path):
        _write(tmp_path / "strategies" / "p" / "strategy.yaml",
               "webui:\n  tasks:\n    - {id: mine, name: M, script: train_foo.py}\n")
        _write(tmp_path / "strategies" / "p" / "train_foo.py", "x = 1\n")
        proj = discovery.discover_projects(tmp_path)[0]
        ids = [t.id for t in discovery.project_tasks(proj, tmp_path)]
        assert ids.count("mine") == 1 and "p.train_foo" not in ids


# --------------------------------------------------------------- 代码发现

class TestCodeStrategyDiscovery:
    def test_derive_params_kinds_and_defaults(self):
        from qmtquant.strategy.base import StrategyBase

        class Fake(StrategyBase):
            """假策略"""
            parameters = ["n", "z", "flag", "label"]
            n: int = 3
            z: float = 1.5
            flag: bool = True
            label: str = "hi"

            def on_bar(self, bar):        # pragma: no cover
                pass

        params = {p.name: p for p in discovery.derive_params(Fake)}
        assert {k: v.kind for k, v in params.items()} == {
            "n": "int", "z": "float", "flag": "bool", "label": "str"}
        assert params["n"].default == 3
        assert params["flag"].default is True

    def test_bool_not_confused_with_int(self):
        """bool 是 int 的子类 —— 先判 int 会把 True/False 认成整数。"""
        from qmtquant.strategy.base import StrategyBase

        class Fake(StrategyBase):
            """x"""
            parameters = ["flag"]
            flag = True                    # 无注解，只能看默认值

            def on_bar(self, bar):         # pragma: no cover
                pass

        assert discovery.derive_params(Fake)[0].kind == "bool"

    def test_choices_convention(self):
        from qmtquant.strategy.base import StrategyBase

        class Fake(StrategyBase):
            """x"""
            parameters = ["mode"]
            mode = "a"
            mode_choices = ["a", "b"]

            def on_bar(self, bar):         # pragma: no cover
                pass

        p = discovery.derive_params(Fake)[0]
        assert p.kind == "choice" and p.choices == ["a", "b"]

    def test_all_builtins_resolve(self):
        from qmtquant.research import loader
        ids = {s.id for s in discovery.discover_code_strategies()}
        assert set(loader.BUILTIN) <= ids

    def test_import_error_skipped(self, tmp_path):
        _write(tmp_path / "strategies" / "broken" / "strategy.py",
               "import definitely_not_a_module_xyz\n")
        specs = discovery.discover_code_strategies(tmp_path)
        assert not any(s.dir == "broken" for s in specs)


# --------------------------------------------------------------- 合并去重

class TestMergeAndDedup:
    def test_keys_shape(self):
        assert discovery.keys("a", "d", "strategies/x/backtest") == {
            "a", "dir:d", "out:strategies/x/backtest"}

    def test_core_claimed_project_not_duplicated(self, monkeypatch):
        from webui import strategies as strat
        core = [strat.Strategy(id="alstm_ensemble", name="x", category="日频",
                               summary="", dir="alstm_ppo_csi1000")]
        monkeypatch.setattr(discovery, "discovered_specs",
                            lambda root=discovery.ROOT: [
                                StrategySpec(id="alstm_ppo_csi1000",
                                             dir="alstm_ppo_csi1000")])
        merged = strat._merge_discovered(core)
        assert [s.id for s in merged] == ["alstm_ensemble"], "核心已认领的项目不该重复出现"

    def test_discovery_failure_falls_back_to_core(self, monkeypatch):
        from webui import strategies as strat

        def boom(root=discovery.ROOT):
            raise RuntimeError("坏了")

        core = [strat.Strategy(id="only", name="x", category="日频", summary="")]
        monkeypatch.setattr(discovery, "discovered_specs", boom)
        assert [s.id for s in strat._merge_discovered(core)] == ["only"]

    def test_kill_switch(self, monkeypatch):
        from webui import strategies as strat
        monkeypatch.setattr(strat, "_DISCOVERY_ON", False)
        core = [strat.Strategy(id="only", name="x", category="日频", summary="")]
        assert [s.id for s in strat._merge_discovered(core)] == ["only"]

    def test_real_registry_has_no_duplicates(self):
        from webui.strategies import STRATEGIES
        ids = [s.id for s in STRATEGIES]
        assert len(ids) == len(set(ids))


# --------------------------------------------------------------- 安全边界

class TestScriptSecurity:
    def test_rejects_parent_traversal(self, tmp_path):
        with pytest.raises(ValueError):
            discovery.resolve_script(tmp_path, "../../evil.py")

    def test_rejects_absolute(self, tmp_path):
        with pytest.raises(ValueError):
            discovery.resolve_script(tmp_path, "C:/Windows/system32/evil.py")

    def test_rejects_non_python(self, tmp_path):
        with pytest.raises(ValueError):
            discovery.resolve_script(tmp_path, "run.sh")

    def test_accepts_plain_name(self, tmp_path):
        assert discovery.resolve_script(tmp_path, "train_x.py").name == "train_x.py"


# --------------------------------------------------------------- 通用 loader

class TestGenericLoader:
    def _spec(self) -> StrategySpec:
        return StrategySpec(
            id="x", dir="x", output_dir="out",
            result=ResultSpec(
                kind="summary", source_json="r.json",
                metrics={"sharpe": "s.sharpe", "total_return": "s.ret",
                         "max_drawdown": "s.dd"},
                period="s.period", n_trades="s.n"))

    def test_reads_dotted_metrics_and_normalizes_dd(self, tmp_path):
        from webui import strategies as strat
        out = tmp_path / "out"
        out.mkdir()
        (out / "r.json").write_text(json.dumps({
            "s": {"sharpe": 1.2, "ret": 0.3, "dd": 0.17,
                  "period": "2022 ~ 2026", "n": 10}}), encoding="utf-8")
        r = strat._generic_loader(self._spec(), base=tmp_path)()
        assert r["has_result"] is True
        assert r["metrics"]["sharpe"] == pytest.approx(1.2)
        assert r["metrics"]["max_drawdown"] == pytest.approx(-0.17), "回撤必须归一为负"
        assert r["n_trades"] == 10
        assert r["period"] == "2022 ~ 2026"

    def test_missing_file_is_empty_state(self, tmp_path):
        from webui import strategies as strat
        assert strat._generic_loader(self._spec(), base=tmp_path)() == {
            "has_result": False}

    def test_non_numeric_metric_becomes_none(self, tmp_path):
        """metrics 里只能是数字或 None —— 字符串会让页面显示成「—」。"""
        from webui import strategies as strat
        out = tmp_path / "out"
        out.mkdir()
        (out / "r.json").write_text(json.dumps({"s": {"sharpe": "很高"}}),
                                    encoding="utf-8")
        r = strat._generic_loader(self._spec(), base=tmp_path)()
        assert r["metrics"]["sharpe"] is None


# --------------------------------------------------------------- 命令拼装

class TestBuildCommand:
    def test_unknown_task_raises(self):
        from webui import registry
        with pytest.raises(ValueError):
            registry.build_command("definitely_not_a_task", {})

    def test_flag_style_core_task(self):
        from webui import registry
        cmd = registry.build_command("train_alstm", {"holdings": 30})
        assert cmd[2] == "strategies/alstm_ppo_csi1000/train_alstm.py"
        assert "--holdings" in cmd and "30" in cmd

    def test_unknown_param_ignored(self):
        from webui import registry
        assert (registry.build_command("train_ppo", {"nope": 1})
                == registry.build_command("train_ppo", {}))

    def test_set_style_json_and_extra_args(self, monkeypatch):
        from webui import registry
        t = registry.Task(
            id="z.run_strategy", name="z", script="scripts/run_strategy.py",
            desc="", param_style="set",
            extra_args=["--strategy", "pkg.mod.Cls"],
            params=[registry.Param("seeds", "种子", "int", 8),
                    registry.Param("flag", "开关", "bool", False)])
        monkeypatch.setitem(registry.TASK_BY_ID, "z.run_strategy", t)
        cmd = registry.build_command("z.run_strategy", {"seeds": 8, "flag": True})
        assert cmd[2:5] == ["scripts/run_strategy.py", "--strategy", "pkg.mod.Cls"]
        assert "--set" in cmd
        assert "seeds=8" in cmd and "flag=true" in cmd

    def test_set_style_coerces_form_strings(self, monkeypatch):
        """表单提交的数字是字符串。原样 json.dumps 会变成 ``n="20"``，
        run_strategy 解析出 str，而 update_setting 不做类型转换 ——
        策略里 ``bar_count % "20"`` 直接崩。"""
        from webui import registry
        t = registry.Task(
            id="z.coerce", name="z", script="scripts/run_strategy.py",
            desc="", param_style="set",
            params=[registry.Param("n", "n", "int", 20),
                    registry.Param("z", "z", "float", 1.5),
                    registry.Param("mode", "m", "choice", "a", choices=["a", "10"])])
        monkeypatch.setitem(registry.TASK_BY_ID, "z.coerce", t)
        cmd = registry.build_command("z.coerce", {"n": "30", "z": "0.25", "mode": "10"})
        assert "n=30" in cmd and "z=0.25" in cmd
        assert 'mode="10"' in cmd, "choice/str 必须保持字符串，不能被 JSON 当成数字"

    def test_set_style_rejects_non_numeric(self, monkeypatch):
        from webui import registry
        t = registry.Task(id="z.bad", name="z", script="scripts/run_strategy.py",
                          desc="", param_style="set",
                          params=[registry.Param("n", "n", "int", 20)])
        monkeypatch.setitem(registry.TASK_BY_ID, "z.bad", t)
        with pytest.raises(ValueError):
            registry.build_command("z.bad", {"n": "abc"})
