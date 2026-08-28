"""策略加载与参数解析测试。

check_params 存在的理由值得单独说：StrategyBase.update_setting
只接受 parameters 里声明过的字段，**未声明的会被静默丢弃** ——
敲错一个参数名不报错，策略照常用默认值跑，而你以为改生效了。
"""
import pytest

from qmtquant.research.loader import (
    BUILTIN,
    check_params,
    describe,
    load_strategy,
    parse_params,
)
from qmtquant.strategy.base import StrategyBase
from qmtquant.strategy.index_timing import IndexTimingStrategy


class TestLoadStrategy:
    def test_builtin_short_name(self):
        assert load_strategy("index_timing") is IndexTimingStrategy

    def test_all_builtins_resolve(self):
        """短名表里的每一项都必须真能加载 —— 改名后忘了同步会在这里挂"""
        for name in BUILTIN:
            cls = load_strategy(name)
            assert issubclass(cls, StrategyBase), name

    def test_full_dotted_path(self):
        cls = load_strategy("qmtquant.strategy.index_timing.IndexTimingStrategy")
        assert cls is IndexTimingStrategy

    def test_unknown_short_name_lists_options(self):
        with pytest.raises(SystemExit, match="可用短名"):
            load_strategy("nonexistent")

    def test_missing_module(self):
        with pytest.raises(SystemExit, match="无法导入模块"):
            load_strategy("no.such.Module")

    def test_missing_class_hints_candidates(self):
        """类名打错时应提示该模块里有哪些策略类"""
        with pytest.raises(SystemExit, match="IndexTimingStrategy"):
            load_strategy("qmtquant.strategy.index_timing.Typo")

    def test_rejects_non_strategy(self):
        with pytest.raises(SystemExit, match="不是 StrategyBase 的子类"):
            load_strategy("qmtquant.config.AppConfig")


class TestParseParams:
    def test_none(self):
        assert parse_params(None) == {}

    def test_int_and_float(self):
        p = parse_params(["ma_window=60", "band=0.02"])
        assert p == {"ma_window": 60, "band": 0.02}
        assert isinstance(p["ma_window"], int)

    def test_bool_and_null(self):
        assert parse_params(["reverse=true", "x=null"]) == {
            "reverse": True, "x": None}

    def test_negative_number(self):
        assert parse_params(["entry_z=-2.0"]) == {"entry_z": -2.0}

    def test_list_value(self):
        assert parse_params(["ws=[5, 20]"]) == {"ws": [5, 20]}

    def test_bare_string_falls_back(self):
        """mode=trend 不是合法 JSON，应原样当字符串"""
        assert parse_params(["mode=trend"]) == {"mode": "trend"}

    def test_strips_key_whitespace(self):
        assert parse_params([" ma_window = 60"]) == {"ma_window": 60}

    def test_missing_equals(self):
        with pytest.raises(SystemExit, match="key=value"):
            parse_params(["ma_window"])

    def test_value_containing_equals(self):
        assert parse_params(["expr=a=b"]) == {"expr": "a=b"}


class TestCheckParams:
    def test_accepts_declared(self):
        check_params(IndexTimingStrategy, {"ma_window": 60, "band": 0.01})

    def test_rejects_typo(self):
        """静默忽略是最坏的行为 —— 必须直接报错"""
        with pytest.raises(SystemExit, match="ma_windwo"):
            check_params(IndexTimingStrategy, {"ma_windwo": 60})

    def test_error_lists_valid_params(self):
        with pytest.raises(SystemExit, match="ma_window"):
            check_params(IndexTimingStrategy, {"bogus": 1})

    def test_empty_ok(self):
        check_params(IndexTimingStrategy, {})


class TestDescribe:
    def test_lists_params_and_defaults(self):
        text = describe(IndexTimingStrategy)
        assert "IndexTimingStrategy" in text
        for name in IndexTimingStrategy.parameters:
            assert name in text
        assert "60" in text, "应显示 ma_window 的默认值"
