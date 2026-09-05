"""历史 ST 判定 —— 回测、训练、选股共用一份。

判定逻辑此前在 ``backtest_engine`` 和 ``scripts/download_st_history.py``
里各写了一份。训练侧要是再抄第三份，三处迟早走样：回测按 5% 撮合的标的，
训练时却当常规股喂进去，两边的「ST」不是同一个集合，
问题还很难被发现 —— 结果只是回测和实盘悄悄对不上。

数据由 ``scripts/download_st_history.py`` 从 Tushare namechange 生成。

## 两种排除口径

``span``（区间精确）
    只排除每只股票**真正处于 ST 的那段时间**。2023 年 ST、2025 年摘帽的
    股票，2025 年的数据照常使用。这是无前视的正确做法，保留最多样本。

``symbol``（整只排除）
    只要历史上当过 ST，整只股票所有时间段全部排除。更保守 ——
    ST 摘帽往往伴随重组、借壳，其后的价格行为带着一次性事件的印记，
    模型学到的可能是「猜哪只会被重组」而不是可持续的规律。
    代价是丢掉约 15% 的股票池。

默认 ``span``：它在统计上更干净（无前视），且不损失样本。
需要更保守时显式传 ``symbol``。
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: 加载一次就缓存 —— 回测里 is_st 会被调用几十万次
_CACHE: dict[str, object] = {}


def st_history_path() -> Path:
    from ..config import get_config
    return Path(get_config().data.store_dir) / "universe" / "st_history.parquet"


def load_spans() -> pd.DataFrame | None:
    """读 ST 区间表，无数据返回 None。"""
    if "df" in _CACHE:
        return _CACHE["df"]  # type: ignore[return-value]
    p = st_history_path()
    if not p.exists():
        _CACHE["df"] = None
        return None
    try:
        df = pd.read_parquet(p)
    except (OSError, ValueError):
        logger.exception("读取 ST 历史失败: %s", p)
        df = None
    _CACHE["df"] = df
    return df


def spans_by_symbol() -> dict[str, list[tuple]]:
    """{vt_symbol: [(start, end), ...]}。end 为 NaT 表示至今仍是 ST。"""
    if "by_sym" in _CACHE:
        return _CACHE["by_sym"]  # type: ignore[return-value]
    df = load_spans()
    by_sym: dict[str, list[tuple]] = {}
    if df is not None:
        for r in df.itertuples():
            by_sym.setdefault(r.vt_symbol, []).append(
                (r.start_date, r.end_date))
    _CACHE["by_sym"] = by_sym
    return by_sym


def ever_st() -> set[str]:
    """历史上当过 ST 的全部标的。"""
    return set(spans_by_symbol())


def get_checker():
    """返回 ``is_st(vt_symbol, date) -> bool``，无数据时返回 None。

    调用方拿到 None 应退回按板块判定涨跌停（会把 ST 当 10% 处理，
    高估可成交性），并把这件事说出来而不是静默降级。
    """
    by_sym = spans_by_symbol()
    if not by_sym:
        return None

    def is_st(vt_symbol: str, date) -> bool:
        spans = by_sym.get(vt_symbol)
        if not spans:
            return False
        d = pd.Timestamp(date)
        for s, e in spans:
            if pd.isna(s):
                continue
            if d >= s and (pd.isna(e) or d <= e):
                return True
        return False

    return is_st


def st_mask(index: pd.MultiIndex, mode: str = "span") -> pd.Series:
    """给 (datetime, instrument) 多级索引打 ST 标记。

    ``index`` 的 instrument 层可以是 qlib 代码（sh600000）或 vt 代码
    （600000.SSE），自动识别。返回与 index 等长的布尔 Series，
    True 表示该 (日期, 标的) 处于需排除状态。

    向量化实现：逐行调用 is_st 在千万行量级上要跑十几分钟，
    这里按标的分组、用区间做整段赋值。
    """
    by_sym = spans_by_symbol()
    mask = pd.Series(False, index=index)
    if not by_sym or mode == "none":
        return mask

    from .qlib_export import from_qlib_code

    insts = index.get_level_values("instrument")
    dts = index.get_level_values("datetime")

    # 索引里的代码可能是 qlib 格式，转成 vt 才能对上 ST 表
    uniq = pd.Index(insts.unique())
    vt_of: dict[str, str] = {}
    for c in uniq:
        c = str(c)
        if "." in c:
            vt_of[c] = c
        else:
            try:
                vt_of[c] = from_qlib_code(c)
            except (ValueError, KeyError):
                vt_of[c] = c

    vt_series = pd.Series([vt_of[str(c)] for c in insts], index=range(len(insts)))

    if mode == "symbol":
        hit = vt_series.isin(by_sym).values
        return pd.Series(hit, index=index)

    # span：按标的分组，每组内对每个 ST 区间做整段布尔运算
    arr = mask.values.copy()
    dt_vals = pd.DatetimeIndex(dts)
    for vt, positions in vt_series.groupby(vt_series).groups.items():
        spans = by_sym.get(vt)
        if not spans:
            continue
        pos = list(positions)
        sub = dt_vals[pos]
        in_st = pd.Series(False, index=range(len(pos)))
        for s, e in spans:
            if pd.isna(s):
                continue
            hit = sub >= s
            if not pd.isna(e):
                hit = hit & (sub <= e)
            in_st |= pd.Series(hit, index=range(len(pos)))
        arr[pos] = in_st.values

    return pd.Series(arr, index=index)
