"""配置拼写错误不再静默吃掉。

## 起因

_fill() 对未知 key 直接 drop，拼写错误的配置项会静默使用默认值。
比如把 commission_rate 写成 commision_rate，系统会用万 0.854 的默认
佣金率而不是 yaml 里写的 —— 而这个差异在回测报告里看不出来（不会
报错，只是收益率不同），只有算到月底对不上账才发现。

同时顺带修了 config.example.yaml 的印花税率：模板里还是减半之前的
千 1 (0.001)，复制出来的 config.yaml 会让所有卖出成本翻倍。
"""
import logging

import pytest

from qmtquant.config import CostConfig, RiskConfig, _fill


class TestUnknownKeysWarned:
    def test_typo_in_cost_config(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = _fill(CostConfig, {"commision_rate": 0.0003})
        assert cfg.commission_rate == CostConfig.commission_rate, \
            "拼写错误的 key 不应影响正确 key 的默认值"
        assert any("commision_rate" in r.message for r in caplog.records), \
            "拼写错误必须告警，否则和「正确加载了」长得一模一样"

    def test_valid_key_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            _fill(CostConfig, {"commission_rate": 0.0003})
        assert not any("无法识别" in r.message for r in caplog.records)

    def test_empty_data_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            _fill(CostConfig, None)
        assert not any("无法识别" in r.message for r in caplog.records)

    def test_risk_config_typo(self, caplog):
        with caplog.at_level(logging.WARNING):
            _fill(RiskConfig, {"forbit_st": True})
        assert any("forbit_st" in r.message for r in caplog.records)


class TestExampleYamlStampTax:
    """config.example.yaml 的印花税率必须是减半后的万 5。"""

    def test_stamp_tax_matches_code_default(self):
        from pathlib import Path
        import yaml

        example = Path(__file__).resolve().parents[1] / "config" / "config.example.yaml"
        if not example.exists():
            pytest.skip("config.example.yaml 不存在")
        with open(example, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        yaml_rate = raw.get("cost", {}).get("stamp_tax_rate")
        assert yaml_rate is not None, "example.yaml 里没有 stamp_tax_rate"
        assert yaml_rate == CostConfig.stamp_tax_rate, (
            f"example.yaml 印花税 {yaml_rate} != 代码默认 {CostConfig.stamp_tax_rate}。"
            f"复制 example 出来的 config.yaml 会用错的税率")
