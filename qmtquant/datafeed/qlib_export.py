"""把本地行情导出成 Qlib 的二进制数据格式。

## 为什么要自己写转换器

Qlib 官方的 ``dump_bin.py`` 只存在于 GitHub 源码的 ``scripts/`` 下，
**不随 pip 包发布**（``pip install pyqlib`` 装完找不到它）。
与其去抓一个版本可能对不上的外部脚本，不如按格式规范自己写 ——
格式本身很简单，且我们完全清楚本地数据的口径。

## Qlib 的数据目录长什么样

::

    qlib_data/
      calendars/day.txt              全部交易日，一行一个 YYYY-MM-DD
      instruments/all.txt            代码 \\t 起始日 \\t 结束日
      instruments/csi300.txt         指数成分，**同一代码可有多行**（多个在册区间）
      features/sh600000/close.day.bin
      features/sh600000/volume.day.bin
      ...

## .bin 文件格式

第一个 float32 是**该标的首个数据点在全局日历中的下标**，
后面依次是每个交易日的值。读取时按这个偏移量对齐日历。

这个设计意味着：**日历必须是全局统一的**，且每只股票的数据必须
在日历上连续。停牌日要填 NaN 占位，不能跳过 ——
跳过会让后续所有日期错位，而且不会报错，只会静默算出错误的收益率。

## 代码格式

Qlib 用 ``sh600000`` / ``sz000001``（交易所小写前缀 + 代码），
与本项目的 ``600000.SSE`` 不同，边界处转换。
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: 本项目交易所 -> Qlib 前缀
_EX_PREFIX = {"SSE": "sh", "SZSE": "sz", "BSE": "bj"}

#: 默认导出的字段。Qlib 的表达式引擎默认用这几个名字
DEFAULT_FIELDS = ("open", "high", "low", "close", "volume", "amount", "factor")


def to_qlib_code(vt_symbol: str) -> str:
    """``600000.SSE`` -> ``sh600000``"""
    code, _, ex = vt_symbol.rpartition(".")
    prefix = _EX_PREFIX.get(ex)
    if prefix is None:
        raise ValueError(f"未知交易所 {ex}（来自 {vt_symbol}）")
    return f"{prefix}{code}"


def from_qlib_code(qlib_code: str) -> str:
    """``sh600000`` -> ``600000.SSE``"""
    prefix, code = qlib_code[:2].lower(), qlib_code[2:]
    for ex, p in _EX_PREFIX.items():
        if p == prefix:
            return f"{code}.{ex}"
    raise ValueError(f"未知 Qlib 代码前缀: {qlib_code}")


def write_bin(path: Path, values: np.ndarray, start_index: int) -> None:
    """写单个 .bin 文件。

    :param start_index: 首个数据点在全局日历中的下标。
        Qlib 把它当作第一个 float32 存在文件头 —— 这是它能用
        「一个全局日历 + 每股一个偏移量」表达稀疏时间序列的原因。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype="<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<f", float(start_index)))
        f.write(arr.tobytes())


def read_bin(path: Path, calendar: pd.DatetimeIndex) -> pd.Series:
    """读回 .bin，主要用于校验导出是否正确"""
    raw = np.fromfile(path, dtype="<f4")
    if raw.size < 1:
        return pd.Series(dtype=float)
    start = int(raw[0])
    values = raw[1:]
    idx = calendar[start:start + len(values)]
    return pd.Series(values.astype(float), index=idx)


class QlibExporter:
    """把 ``data/1d/`` 的日线导出成 Qlib 数据目录"""

    def __init__(self, store_dir: str, out_dir: str) -> None:
        self.store_dir = Path(store_dir)
        self.out = Path(out_dir)
        self.calendar: pd.DatetimeIndex | None = None

    # ------------------------------------------------------------ 读取

    def _load_one(self, vt_symbol: str) -> pd.DataFrame | None:
        code, _, ex = vt_symbol.rpartition(".")
        path = self.store_dir / "1d" / ex / f"{code}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        idx = pd.to_datetime(df.index.astype(str), format="%Y%m%d",
                             errors="coerce")
        df = df.set_axis(idx).sort_index()
        return df[~df.index.isna()]

    # ------------------------------------------------------------ 导出

    def build_calendar(self, symbols: list[str],
                       start: str, end: str) -> pd.DatetimeIndex:
        """全局交易日历 = 所有标的交易日的并集。

        必须取并集而非某一只的日历：单只股票停牌期间没有数据，
        用它当日历会漏掉那些日子，导致其他股票的数据整体错位。
        """
        all_dates: set = set()
        for vt in symbols:
            df = self._load_one(vt)
            if df is None or df.empty:
                continue
            all_dates |= set(df.index)
        cal = pd.DatetimeIndex(sorted(all_dates))
        cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
        if cal.empty:
            raise SystemExit(f"{start} ~ {end} 区间内没有交易日")
        self.calendar = cal
        return cal

    def export_features(self, symbols: list[str],
                        fields: tuple[str, ...] = DEFAULT_FIELDS,
                        progress=None) -> dict:
        """导出行情。返回 {vt_symbol: (首日, 末日)}，供写 instruments 用。"""
        if self.calendar is None:
            raise RuntimeError("请先调用 build_calendar")
        cal = self.calendar
        pos = pd.Series(range(len(cal)), index=cal)

        spans: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for i, vt in enumerate(symbols, 1):
            if progress:
                progress(i, len(symbols), vt)
            df = self._load_one(vt)
            if df is None or df.empty:
                continue
            df = df[(df.index >= cal[0]) & (df.index <= cal[-1])]
            if df.empty:
                continue

            # 对齐到全局日历的连续区间。区间内缺失的日子填 NaN ——
            # 跳过会让后续日期整体错位，且不报错，只会静默算错收益率
            lo, hi = int(pos[df.index[0]]), int(pos[df.index[-1]])
            window = cal[lo:hi + 1]
            df = df.reindex(window)

            out_dir = self.out / "features" / to_qlib_code(vt)
            for field in fields:
                series = self._field_series(df, field)
                if series is None:
                    continue
                write_bin(out_dir / f"{field}.day.bin", series.to_numpy(), lo)
            spans[vt] = (window[0], window[-1])
        return spans

    @staticmethod
    def _field_series(df: pd.DataFrame, field: str) -> pd.Series | None:
        """把本项目列名映射到 Qlib 字段名"""
        if field == "amount":
            # 本项目叫 amount，Qlib 也叫 amount
            return df["amount"] if "amount" in df.columns else None
        if field == "factor":
            # 复权因子。本地存的已经是**后复权价**，因子恒为 1 ——
            # 写出来是为了让 Qlib 的 $close/$factor 之类表达式不报错
            return pd.Series(1.0, index=df.index)
        return df[field] if field in df.columns else None

    def write_calendar(self) -> Path:
        path = self.out / "calendars" / "day.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(d.strftime("%Y-%m-%d") for d in self.calendar) + "\n",
            encoding="utf-8")
        return path

    def write_instruments(self, spans: dict, name: str = "all") -> Path:
        """写全市场标的清单：代码 \\t 起始日 \\t 结束日"""
        path = self.out / "instruments" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{to_qlib_code(vt)}\t{lo:%Y-%m-%d}\t{hi:%Y-%m-%d}"
                 for vt, (lo, hi) in sorted(spans.items())]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_index_members(self, weight_csv: str, name: str,
                            spans: dict) -> Path:
        """按历史成分名单写指数 instruments 文件。

        Qlib 的格式允许**同一代码出现多行**，每行是一段在册区间。
        这正是消除成分股前视所需要的 —— 某只股票 2018 年调入、
        2021 年调出、2024 年又调入，就写三行。
        """
        w = pd.read_csv(weight_csv, parse_dates=["date"])
        periods = sorted(w["date"].unique())
        members = {d: set(g["symbol"]) for d, g in w.groupby("date")}

        # 把「每期名单」转成「每只股票的在册区间」。
        #
        # ⚠ 调出时区间末端要取**上一期**的日期，不是它消失的那一期。
        # 取消失当期会让 Qlib 在调样日同时把「已调出」和「新调入」
        # 两批都算作成分 —— 实测 2017-06-30 查出 330 只而非 300 只。
        # 那等于让策略在个股被剔除当天仍持有它。
        spans_by_code: dict[str, list[list]] = {}
        prev: set = set()
        prev_date = None
        for d in periods:
            cur = members[d]
            for code in cur - prev:                 # 新调入
                spans_by_code.setdefault(code, []).append([d, None])
            for code in prev - cur:                 # 调出
                if spans_by_code.get(code) and prev_date is not None:
                    spans_by_code[code][-1][1] = prev_date
            prev, prev_date = cur, d

        last = periods[-1]
        lines = []
        for code, ranges in sorted(spans_by_code.items()):
            if code not in spans:
                continue                            # 本地无行情，跳过
            for lo, hi in ranges:
                hi = hi if hi is not None else last
                lines.append(f"{to_qlib_code(code)}\t"
                             f"{pd.Timestamp(lo):%Y-%m-%d}\t"
                             f"{pd.Timestamp(hi):%Y-%m-%d}")

        path = self.out / "instruments" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("%s: %d 段在册区间，涉及 %d 只标的",
                    name, len(lines), len(spans_by_code))
        return path
