"""按名字加载策略类与解析参数。

有了这一层，回测与验证脚本就不必为每个策略各写一份 ——
新策略写完直接用 ``--strategy 模块路径.类名`` 跑，不用改任何脚本。
"""
import importlib
import json
import logging

from ..strategy.base import StrategyBase

logger = logging.getLogger(__name__)

#: 内置策略的短名，省得每次敲完整路径
BUILTIN = {
    "mean_reversion": "qmtquant.strategy.mean_reversion.MeanReversionStrategy",
    "index_timing": "qmtquant.strategy.index_timing.IndexTimingStrategy",
    "momentum": "qmtquant.strategy.momentum.MomentumRotationStrategy",
    "trend_ma": "qmtquant.strategy.trend_ma.TrendMaStrategy",
    "ma_cross": "qmtquant.strategy.ma_cross.MaCrossStrategy",
    "intraday_vwap": "qmtquant.strategy.intraday_vwap.IntradayVwapStrategy",
    "breakout": "qmtquant.strategy.breakout.BreakoutStrategy",
    "low_turnover": "qmtquant.strategy.low_turnover.LowTurnoverStrategy",
}


def load_strategy(name: str) -> type[StrategyBase]:
    """按短名或完整路径加载策略类。

    :param name: ``mean_reversion`` 或 ``my_pkg.my_mod.MyStrategy``
    :raises SystemExit: 加载失败时给出可操作的提示而非堆栈
    """
    path = BUILTIN.get(name, name)
    if "." not in path:
        raise SystemExit(
            f"无法识别的策略 {name!r}。\n"
            f"可用短名: {', '.join(sorted(BUILTIN))}\n"
            f"或给出完整路径，如 my_pkg.my_mod.MyStrategy")

    module_path, _, class_name = path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise SystemExit(f"无法导入模块 {module_path}：{e}") from e

    cls = getattr(module, class_name, None)
    if cls is None:
        # 列出该模块里的策略类，通常是类名打错了
        candidates = [n for n, v in vars(module).items()
                      if isinstance(v, type) and issubclass(v, StrategyBase)
                      and v is not StrategyBase]
        hint = f"，该模块内的策略类有: {', '.join(candidates)}" if candidates else ""
        raise SystemExit(f"{module_path} 中没有 {class_name}{hint}")

    if not (isinstance(cls, type) and issubclass(cls, StrategyBase)):
        raise SystemExit(f"{path} 不是 StrategyBase 的子类，无法作为策略运行")
    return cls


def parse_params(items: list[str] | None) -> dict:
    """解析 ``key=value`` 形式的参数。

    值按 JSON 解析，解析失败则当作字符串 ——
    这样 ``ma_window=60`` 得到 int、``mode=trend`` 得到 str，
    而不需要为每个参数声明类型。
    """
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"参数格式应为 key=value，收到 {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def check_params(cls: type[StrategyBase], params: dict) -> None:
    """校验参数名是否被策略声明过。

    这个检查存在的理由：``StrategyBase.update_setting`` 只接受
    ``parameters`` 里声明过的字段，**未声明的会被静默丢弃** ——
    敲错一个参数名不会报错，只会让策略用默认值跑，
    而你还以为改生效了。
    """
    known = set(cls.parameters)
    unknown = sorted(set(params) - known)
    if not unknown:
        return
    raise SystemExit(
        f"{cls.__name__} 不接受这些参数: {', '.join(unknown)}\n"
        f"（未声明的参数会被静默忽略，所以这里直接报错）\n"
        f"可用参数: {', '.join(sorted(known))}")


def describe(cls: type[StrategyBase]) -> str:
    """打印策略的参数与默认值"""
    lines = [f"{cls.__module__}.{cls.__name__}"]
    doc = (cls.__doc__ or "").strip().splitlines()
    if doc:
        lines.append(f"  {doc[0]}")
    lines.append("  参数:")
    for name in cls.parameters:
        lines.append(f"    {name:<20} = {getattr(cls, name, None)!r}")
    return "\n".join(lines)
