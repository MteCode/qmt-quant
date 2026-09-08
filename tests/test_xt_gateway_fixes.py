"""XtGateway 三处会让实盘静默出错的缺陷。

这个网关是 paper_trade.py 实盘路径实际使用的那个，而它有三处问题
在联调时不会报错、只在真实交易里出错：

1. 回调方法名与 SDK 约定不符 → 下单后零回报
2. 委托状态码映射整体错位、废单缺失 → 废单被当成「还活着」
3. order_stock 参数错位 → 必抛 TypeError

前两条最阴险：不抛异常，只是系统对成交一无所知。
"""
import inspect

import pytest

from qmtquant.core.constants import Status


# ------------------------------------------------------------ 回调方法名

def test_回调方法名必须与SDK约定一致():
    """SDK 在 xttrader.py 里调的是 on_stock_order / on_stock_trade。

    写成 on_order_callback / on_trade_callback 不会报错 —— 因为 Callback
    继承了 XtQuantTraderCallback，SDK 命中基类的空实现，自定义回调成为
    永不执行的死代码。下完单没有任何委托或成交回流。
    """
    src = inspect.getsource(
        __import__("qmtquant.gateway.xt_gateway", fromlist=["x"]))
    assert "def on_stock_order(" in src, "缺 on_stock_order，SDK 不会回调"
    assert "def on_stock_trade(" in src, "缺 on_stock_trade，SDK 不会回调"
    assert "def on_order_callback(" not in src, "on_order_callback 是死代码"
    assert "def on_trade_callback(" not in src, "on_trade_callback 是死代码"


def test_SDK确实调用这两个方法名():
    """锚定 SDK 侧的约定，SDK 升级改名时这条会先失败。"""
    xttrader = pytest.importorskip("xtquant.xttrader")
    src = inspect.getsource(xttrader)
    assert "self.callback.on_stock_order(" in src
    assert "self.callback.on_stock_trade(" in src


# ------------------------------------------------------------ 状态码映射

def _status_map():
    """取出 _on_order 里的状态映射表。"""
    from qmtquant.gateway import xt_gateway
    src = inspect.getsource(xt_gateway.XtGateway._on_order)
    ns = {}
    exec("from xtquant import xtconstant as c\n"
         "from qmtquant.core.constants import Status\n"
         + "\n".join(
             line[8:] for line in src.splitlines()
             if line.strip().startswith(("status_map", "c.ORDER", "}"))
             or "Status." in line and ":" in line),
         ns)
    return ns["status_map"]


def test_状态码映射与SDK语义一致():
    """原表整体错位约 2：50(已报)当成部成、54(已撤)当成废单、
    55(部成)当成全成 —— 每一条都会让系统对订单状态判断错误。"""
    c = pytest.importorskip("xtquant.xtconstant")
    m = _status_map()
    expected = {
        c.ORDER_UNREPORTED: Status.SUBMITTING,       # 48 未报
        c.ORDER_WAIT_REPORTING: Status.SUBMITTING,   # 49 待报
        c.ORDER_REPORTED: Status.NOTTRADED,          # 50 已报
        c.ORDER_REPORTED_CANCEL: Status.NOTTRADED,   # 51 已报待撤
        c.ORDER_PARTSUCC_CANCEL: Status.PARTTRADED,  # 52 部成待撤
        c.ORDER_PART_CANCEL: Status.CANCELLED,       # 53 部撤
        c.ORDER_CANCELED: Status.CANCELLED,          # 54 已撤
        c.ORDER_PART_SUCC: Status.PARTTRADED,        # 55 部成
        c.ORDER_SUCCEEDED: Status.ALLTRADED,         # 56 已成
        c.ORDER_JUNK: Status.REJECTED,               # 57 废单
    }
    assert m == expected


def test_废单必须映射为拒绝():
    """废单缺失时会 fallback 成 SUBMITTING —— 一笔被交易所拒掉的单，
    系统认为它还活着，撤单会去撤不存在的单，风控也不知道钱没花出去。"""
    c = pytest.importorskip("xtquant.xtconstant")
    assert _status_map().get(c.ORDER_JUNK) == Status.REJECTED


def test_两个网关的状态映射口径一致():
    """miniqmt_gateway 的表是对的。两个网关口径相反会让同一笔委托
    在不同路径上被判成不同状态。"""
    c = pytest.importorskip("xtquant.xtconstant")
    from qmtquant.gateway.miniqmt_gateway import _XT
    mini = _XT["status"] if isinstance(_XT, dict) else None
    if mini is None:
        pytest.skip("miniqmt_gateway 映射表结构已变")
    assert _status_map() == mini


# ------------------------------------------------------------ 下单参数

def test_下单参数与SDK签名对齐():
    """SDK: order_stock(account, stock_code, order_type, order_volume,
    price_type, price, strategy_name, order_remark)

    漏传 price_type 会把 price 顶到 price_type 位置，price 无默认值 →
    必抛 TypeError。之所以线上没炸，是因为生产路径直接调
    gateway.trader.order_stock，绕过了整层网关抽象。
    """
    xttrader = pytest.importorskip("xtquant.xttrader")
    sig = inspect.signature(xttrader.XtQuantTrader.order_stock)
    names = [p for p in sig.parameters if p != "self"]
    assert names[:6] == ["account", "stock_code", "order_type",
                         "order_volume", "price_type", "price"]

    from qmtquant.gateway import xt_gateway
    src = inspect.getsource(xt_gateway.XtGateway.send_order)
    assert "price_type" in src, "send_order 未传 price_type，调用必失败"
    # price_type 必须排在 req.price 之前
    assert src.index("price_type, req.price") > 0, "price_type 与 price 顺序错误"


def test_报单失败返回空串而非伪造委托号():
    """order_stock 失败返回 -1。不判断就会拼出 'XT.-1' 这种假委托号，
    之后撤单、对账都会跟着错。"""
    from qmtquant.gateway import xt_gateway
    src = inspect.getsource(xt_gateway.XtGateway.send_order)
    assert "order_id < 0" in src or "order_id is None" in src, \
        "未检查报单返回码"
