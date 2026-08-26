"""标的代码转换。

系统内部统一使用 vt_symbol（如 `000001.SZSE`），
xtquant 使用 `000001.SZ`，只在网关/数据层边界做转换。
"""
from ..core.constants import Exchange

_VT_TO_XT = {
    Exchange.SSE: "SH",
    Exchange.SZSE: "SZ",
    Exchange.BSE: "BJ",
}
_XT_TO_VT = {v: k for k, v in _VT_TO_XT.items()}


def split_vt_symbol(vt_symbol: str) -> tuple[str, Exchange]:
    """`000001.SZSE` -> ('000001', Exchange.SZSE)"""
    symbol, _, ex = vt_symbol.rpartition(".")
    return symbol, Exchange(ex)


def to_xt_symbol(vt_symbol: str) -> str:
    """`000001.SZSE` -> `000001.SZ`"""
    symbol, exchange = split_vt_symbol(vt_symbol)
    return f"{symbol}.{_VT_TO_XT[exchange]}"


def from_xt_symbol(xt_symbol: str) -> str:
    """`000001.SZ` -> `000001.SZSE`"""
    symbol, _, suffix = xt_symbol.rpartition(".")
    return f"{symbol}.{_XT_TO_VT[suffix.upper()].value}"


def guess_exchange(symbol: str) -> Exchange:
    """按代码前缀推断交易所，用于用户只给 6 位代码的场景"""
    if symbol.startswith(("60", "68", "51", "58", "56", "50", "11")):
        return Exchange.SSE
    if symbol.startswith(("4", "8", "92")):
        return Exchange.BSE
    return Exchange.SZSE


def normalize(symbol: str) -> str:
    """把 `000001` / `000001.SZ` / `000001.SZSE` 统一成 vt_symbol"""
    if "." not in symbol:
        return f"{symbol}.{guess_exchange(symbol).value}"
    code, _, suffix = symbol.rpartition(".")
    suffix = suffix.upper()
    if suffix in _XT_TO_VT:
        return f"{code}.{_XT_TO_VT[suffix].value}"
    return f"{code}.{Exchange(suffix).value}"
