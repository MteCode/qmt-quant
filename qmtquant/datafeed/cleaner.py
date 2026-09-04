"""数据清洗 —— 把校验发现的问题真正修掉。

`validator.py` 的原则是「只报告，不静默修改」，因为自动修复比脏数据更危险：
你不知道它改了什么。本模块是那条原则的另一半 —— **显式调用、逐条记录**。

每次清洗都返回一份动作清单，说明改了哪些行、依据什么规则。
不留记录的清洗等于把问题从「看得见的脏数据」变成「看不见的假数据」。

## 清洗规则的取舍

只做**有唯一正确答案**的修复：

- 重复时间戳 -> 保留最后一条（后到的数据覆盖先到的，符合增量下载语义）
- 时间未升序 -> 排序（顺序本身不含信息）
- 停牌日成交量为 0 而价格仍在变 -> 保留，仅标记（可能是复权因子调整）

**不做**需要猜测的修复：

- 价格为 0 或负 -> 不填补，直接剔除该行并记录。填补等于凭空造价格
- 涨跌幅超限 -> 不修改，仅标记。可能是真实极端行情，也可能是复权跳变，
  分不清就不该动
- 缺失交易日 -> 不插值。停牌与数据缺失在结果上无法区分，
  插值会把停牌期伪造成正常交易
"""
import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CleanAction:
    """一次清洗动作的记录。"""
    rule: str
    n_rows: int
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.rule}: {self.n_rows} 行" + (f"（{self.detail}）"
                                                   if self.detail else "")


@dataclass
class CleanResult:
    df: pd.DataFrame
    actions: list = field(default_factory=list)
    #: 只标记未修改的问题，供下游决定是否采用
    flags: list = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.actions)

    def summary(self) -> str:
        if not self.actions and not self.flags:
            return "无需清洗"
        parts = [str(a) for a in self.actions]
        if self.flags:
            parts.append("仅标记未改：" + "; ".join(str(f) for f in self.flags))
        return " | ".join(parts)


def clean_bars(df: pd.DataFrame, vt_symbol: str,
               drop_nonpositive: bool = True) -> CleanResult:
    """清洗 K 线。

    :param drop_nonpositive: 价格非正的行是否剔除。默认剔除 ——
        这类行无法用于任何计算，留着只会让下游各自处理、各处理各的
    """
    actions, flags = [], []
    if df is None or df.empty:
        return CleanResult(df=df if df is not None else pd.DataFrame())

    out = df.copy()
    n0 = len(out)

    # --- 重复时间戳：保留最后一条。增量下载时后到的数据更新，应覆盖先到的
    dup = out.index.duplicated(keep="last")
    if dup.any():
        out = out[~dup]
        actions.append(CleanAction("去重", int(dup.sum()), "保留最后一条"))

    # --- 排序。顺序本身不含信息，可以放心修
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
        actions.append(CleanAction("按时间排序", len(out)))

    price_cols = [c for c in ("open", "high", "low", "close")
                  if c in out.columns]

    # --- 价格非正：剔除而不填补。填补等于凭空造价格
    if drop_nonpositive and price_cols:
        bad = pd.Series(False, index=out.index)
        for c in price_cols:
            bad |= out[c].notna() & (out[c] <= 0)
        if bad.any():
            out = out[~bad]
            actions.append(CleanAction("剔除非正价格", int(bad.sum()),
                                       "无法用于计算，不填补"))

    # --- 成交量为负：置零并记录。负成交量一定是错误，0 是安全的下界
    if "volume" in out.columns:
        neg = out["volume"].notna() & (out["volume"] < 0)
        if neg.any():
            out.loc[neg, "volume"] = 0
            actions.append(CleanAction("负成交量置零", int(neg.sum())))

    # --- OHLC 逻辑矛盾：只标记不修。改哪一个都是猜
    if {"high", "low"} <= set(out.columns):
        bad = out["high"].notna() & out["low"].notna() & (out["high"] < out["low"])
        if bad.any():
            flags.append(CleanAction("最高价低于最低价", int(bad.sum()),
                                     "改哪一边都是猜，交由下游判断"))
    if {"high", "open", "close"} <= set(out.columns):
        bad = (out["high"].notna()
               & ((out["high"] < out["open"]) | (out["high"] < out["close"])))
        if bad.any():
            flags.append(CleanAction("最高价低于开盘或收盘", int(bad.sum())))

    # --- 涨跌幅异常：只标记。真实极端行情与复权跳变分不清，不该动
    if "close" in out.columns and len(out) > 1:
        from ..core.constants import get_price_limit
        limit = get_price_limit(vt_symbol)
        chg = out["close"].pct_change()
        bad = chg.abs() > limit * 1.5
        if bad.any():
            flags.append(CleanAction(
                f"涨跌幅超 {limit:.0%} 的 1.5 倍", int(bad.sum()),
                f"最大 {chg.abs().max():.1%}，多为复权跳变"))

    if len(out) != n0:
        logger.debug("%s 清洗后 %d -> %d 行", vt_symbol, n0, len(out))
    return CleanResult(df=out, actions=actions, flags=flags)


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """把 QMT 原始日线整理成统一 schema。

    QMT 的原始表用 yyyymmdd 整数作索引、另有一列毫秒时间戳，
    还带着股票用不到的期货字段（结算价、持仓量）。统一成
    `date` 索引 + 标准列，下游不必各自适配。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # 索引可能是 yyyymmdd 整数或字符串，统一成日期
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        out.index = pd.to_datetime(idx.astype(str), format="%Y%m%d",
                                   errors="coerce")
    out = out[out.index.notna()]
    out.index.name = "date"

    # time 列与索引重复，期货字段对股票无意义
    drop = [c for c in ("time", "settelementPrice", "openInterest")
            if c in out.columns]
    if drop:
        out = out.drop(columns=drop)

    return out
