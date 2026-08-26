"""miniQMT 网关常量映射测试。

这些映射曾经出过事：深市限价类型硬编码成 101，实测返回 -1，
导致所有深市股票（000/002/300 开头）都下不了单，且失败原因不可见。
现在改为从 xtconstant 读取，本测试确保映射与 SDK 保持一致。

没装 xtquant 时自动跳过 SDK 相关用例。
"""
import pytest

from qmtquant.core.constants import Direction, Status
from qmtquant.gateway.miniqmt_gateway import (
    DIRECTION_VT2XT,
    DIRECTION_XT2VT,
    PRICE_TYPE_LIMIT_VALUE,
    PRICE_TYPE_MARKET_VALUE,
    STATUS_XT2VT,
)

xtconstant = pytest.importorskip(
    "xtquant.xtconstant", reason="未安装 xtquant，跳过 SDK 常量校验"
)


class TestPriceType:
    def test_limit_uses_fix_price(self):
        """限价必须用 FIX_PRICE。旧代码按交易所分 11/101，深市 101 会被拒单。"""
        assert PRICE_TYPE_LIMIT_VALUE == xtconstant.FIX_PRICE

    def test_limit_is_not_legacy_101(self):
        assert PRICE_TYPE_LIMIT_VALUE != 101

    def test_market_type_defined(self):
        assert isinstance(PRICE_TYPE_MARKET_VALUE, int)
        assert PRICE_TYPE_MARKET_VALUE > 0


class TestDirection:
    def test_matches_sdk(self):
        assert DIRECTION_VT2XT[Direction.LONG] == xtconstant.STOCK_BUY
        assert DIRECTION_VT2XT[Direction.SHORT] == xtconstant.STOCK_SELL

    def test_reverse_map_roundtrip(self):
        for direction, code in DIRECTION_VT2XT.items():
            assert DIRECTION_XT2VT[code] is direction


class TestStatusMapping:
    @pytest.mark.parametrize("const_name,expected", [
        ("ORDER_UNREPORTED", Status.SUBMITTING),
        ("ORDER_WAIT_REPORTING", Status.SUBMITTING),
        ("ORDER_REPORTED", Status.NOTTRADED),
        ("ORDER_REPORTED_CANCEL", Status.NOTTRADED),
        ("ORDER_PARTSUCC_CANCEL", Status.PARTTRADED),
        ("ORDER_PART_CANCEL", Status.CANCELLED),
        ("ORDER_CANCELED", Status.CANCELLED),
        ("ORDER_PART_SUCC", Status.PARTTRADED),
        ("ORDER_SUCCEEDED", Status.ALLTRADED),
        ("ORDER_JUNK", Status.REJECTED),
    ])
    def test_each_status(self, const_name, expected):
        code = getattr(xtconstant, const_name)
        assert STATUS_XT2VT[code] is expected, (
            f"{const_name}({code}) 应映射为 {expected}"
        )

    def test_terminal_states_not_active(self):
        """已撤/已成/废单必须是终态，否则订单状态机会卡住：
        明明结束的单系统仍以为活着，导致重复撤单或不敢下新单。"""
        from qmtquant.core.objects import OrderData

        for name in ("ORDER_CANCELED", "ORDER_PART_CANCEL",
                     "ORDER_SUCCEEDED", "ORDER_JUNK"):
            status = STATUS_XT2VT[getattr(xtconstant, name)]
            assert not OrderData(status=status).is_active(), f"{name} 不应为活动状态"

    def test_active_states_are_active(self):
        from qmtquant.core.objects import OrderData

        for name in ("ORDER_UNREPORTED", "ORDER_WAIT_REPORTING",
                     "ORDER_REPORTED", "ORDER_PART_SUCC"):
            status = STATUS_XT2VT[getattr(xtconstant, name)]
            assert OrderData(status=status).is_active(), f"{name} 应为活动状态"

    def test_covers_all_sdk_statuses(self):
        """SDK 新增状态时应能发现，避免出现映射不到的委托状态"""
        missing = []
        for name in dir(xtconstant):
            if not name.startswith("ORDER_"):
                continue
            v = getattr(xtconstant, name)
            # ORDER_UNKNOWN(255) 是哨兵值，不参与映射
            if isinstance(v, int) and v not in STATUS_XT2VT and name != "ORDER_UNKNOWN":
                missing.append(f"{name}={v}")
        assert not missing, f"以下 SDK 状态未映射: {missing}"
