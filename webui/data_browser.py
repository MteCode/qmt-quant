"""数据浏览与校验 —— 把各类存储读成表格，并自动查错。

## 为什么不转存成数据库

全库 2.2 GB、七千多个文件。转一份 CSV 或 SQLite 意味着数据翻倍，
而且下一次下载后副本就过期 —— 用过期副本核对数据正确性，
比不核对更危险。因此这里**按需读取原始文件**，看到的永远是真实状态。

## 重点是 qlib_data

`data/1d/` 之类是 parquet，真要看还能用 pandas 打开；而 `data/qlib_data/`
是 Qlib 的二进制 bin，肉眼完全无法检查。之前那次「765 个标的落后于日历」
的数据撕裂，就藏在这里，直到训练报出 KeyError 才暴露。

bin 格式：4 字节头（起始索引，float32）+ N 个 float32 数值。
起始索引指向全局交易日历，据此可还原每个值对应的日期。

## 自动校验

读到数据顺手查几类常见错误：日历缺口、非正价格、OHLC 逻辑矛盾、
超出涨跌停幅度的异动。涨跌停幅度复用 core.constants 的按板块判定 ——
主板 10%、创业板/科创板 20%、北交所 30%，一刀切会误报大量创业板标的。
"""
import json
from datetime import datetime
from pathlib import Path

from .registry import ROOT

DATA = ROOT / "data"

#: 可浏览的数据源。key -> (显示名, 相对路径, 说明)
SOURCES = {
    "qlib": ("Qlib 特征库", "qlib_data/features",
             "训练直接读取的二进制数据。肉眼无法检查，最需要核对"),
    "1d": ("日线", "1d", "从 QMT 下载的原始日线"),
    "1w": ("周线", "1w", "周线"),
    "1m": ("分钟线", "1m", "分钟线，仅部分标的"),
    "financial": ("财务报表", "financial", "利润表/资产负债表/现金流等"),
    "money_flow": ("资金流", "money_flow", "龙虎榜、两融、北向"),
    "index": ("指数", "index", "指数行情与全收益"),
    "universe": ("标的池", "universe", "成分股名单、行业、上市退市"),
}

#: Qlib bin 里的字段
QLIB_FIELDS = ["open", "high", "low", "close", "volume", "amount", "factor"]


def list_symbols(source: str, limit: int = 400, query: str = "") -> list:
    """某数据源下有哪些标的。"""
    cfg = SOURCES.get(source)
    if not cfg:
        return []
    base = DATA / cfg[1]
    if not base.exists():
        return []

    if source == "qlib":
        names = sorted(d.name for d in base.iterdir() if d.is_dir())
    elif source == "financial":
        # financial/<报表类型>/<交易所>/<代码>.parquet
        names = sorted({p.stem for p in base.rglob("*.parquet")})
    else:
        names = sorted({p.stem for p in base.rglob("*.parquet")})

    if query:
        q = query.lower()
        names = [n for n in names if q in n.lower()]
    return names[:limit]


def _qlib_calendar() -> list:
    p = DATA / "qlib_data" / "calendars" / "day.txt"
    if not p.exists():
        return []
    return p.read_text(encoding="utf-8").split()


def read_qlib(symbol: str, tail: int = 120) -> dict:
    """读 Qlib 二进制特征，还原成带日期的表格。"""
    import numpy as np
    import pandas as pd

    d = DATA / "qlib_data" / "features" / symbol
    if not d.exists():
        return {"error": f"标的不存在: {symbol}"}

    cal = _qlib_calendar()
    cols, starts, lengths = {}, {}, {}
    for f in QLIB_FIELDS:
        p = d / f"{f}.day.bin"
        if not p.exists():
            continue
        raw = np.fromfile(p, dtype=np.float32)
        if len(raw) < 2:
            continue
        # 4 字节头是起始索引，指向全局交易日历
        starts[f] = int(raw[0])
        cols[f] = raw[1:]
        lengths[f] = len(raw) - 1

    if not cols:
        return {"error": f"{symbol} 没有可读的 bin 文件"}

    # 各字段的起止应当一致，不一致本身就是数据损坏
    inconsistent = (len(set(starts.values())) > 1
                    or len(set(lengths.values())) > 1)

    s0 = min(starts.values())
    n = max(lengths.values())
    idx = cal[s0:s0 + n] if cal else [str(i) for i in range(n)]

    data = {}
    for f, arr in cols.items():
        pad = starts[f] - s0
        v = np.full(n, np.nan, dtype=np.float64)
        v[pad:pad + len(arr)] = arr
        data[f] = v

    df = pd.DataFrame(data, index=pd.Index(idx[:n], name="date"))
    return {
        "df": df.tail(tail),
        "full": df,
        "start_index": s0,
        "n_rows": n,
        "calendar_len": len(cal),
        "covers_last": (s0 + n - 1) >= (len(cal) - 1) if cal else None,
        "inconsistent_fields": inconsistent,
        "field_ranges": {f: (starts[f], starts[f] + lengths[f] - 1)
                         for f in cols},
    }


def read_parquet_source(source: str, symbol: str, tail: int = 120) -> dict:
    """读 parquet 类数据源。"""
    import pandas as pd

    cfg = SOURCES.get(source)
    if not cfg:
        return {"error": f"未知数据源: {source}"}
    base = DATA / cfg[1]

    matches = list(base.rglob(f"{symbol}.parquet"))
    if not matches:
        return {"error": f"未找到 {symbol}"}

    # financial 下同一标的在多张报表里都有，全部读出来分组展示
    if source == "financial" and len(matches) > 1:
        groups = {}
        for p in matches:
            try:
                df = pd.read_parquet(p)
            except (OSError, ValueError) as e:
                groups[p.parent.parent.name] = {"error": str(e)}
                continue
            groups[p.parent.parent.name] = {
                "df": df.tail(tail), "n_rows": len(df),
                "path": str(p.relative_to(ROOT)),
            }
        return {"groups": groups}

    p = matches[0]
    try:
        df = pd.read_parquet(p)
    except (OSError, ValueError) as e:
        return {"error": f"读取失败: {e}"}
    return {"df": df.tail(tail), "full": df, "n_rows": len(df),
            "path": str(p.relative_to(ROOT))}


def check_quality(df, source: str, symbol: str) -> list:
    """常见数据错误检查。返回问题列表，空列表表示未发现问题。"""
    import numpy as np
    import pandas as pd

    issues = []
    if df is None or df.empty:
        return [{"level": "error", "msg": "数据为空"}]

    cols = {c.lower(): c for c in df.columns}

    # --- 价格非正
    for f in ("open", "high", "low", "close"):
        c = cols.get(f)
        if c is None:
            continue
        bad = df[c].notna() & (df[c] <= 0)
        if bad.any():
            issues.append({"level": "error",
                           "msg": f"{f} 有 {int(bad.sum())} 个非正值"})

    # --- OHLC 逻辑
    o, h, low, c = (cols.get(x) for x in ("open", "high", "low", "close"))
    if h and low:
        bad = df[h].notna() & df[low].notna() & (df[h] < df[low])
        if bad.any():
            issues.append({"level": "error",
                           "msg": f"最高价低于最低价：{int(bad.sum())} 行"})
    if h and o and c:
        bad = (df[h].notna() & df[o].notna() & df[c].notna()
               & ((df[h] < df[o]) | (df[h] < df[c])))
        if bad.any():
            issues.append({"level": "error",
                           "msg": f"最高价低于开盘/收盘：{int(bad.sum())} 行"})

    # --- 成交量为负
    v = cols.get("volume")
    if v is not None:
        bad = df[v].notna() & (df[v] < 0)
        if bad.any():
            issues.append({"level": "error",
                           "msg": f"成交量为负：{int(bad.sum())} 行"})
        zero = df[v].notna() & (df[v] == 0)
        if zero.sum() > len(df) * 0.1:
            issues.append({"level": "warn",
                           "msg": f"成交量为 0 的交易日占 "
                                  f"{zero.sum() / len(df):.0%}（可能长期停牌）"})

    # --- 涨跌幅异常。按板块取涨跌停幅度，一刀切会误报大量创业板标的
    if c and len(df) > 1:
        try:
            from qmtquant.core.constants import get_price_limit
            limit = get_price_limit(symbol.replace("sh", "").replace("sz", ""))
        except Exception:
            limit = 0.10
        chg = df[c].pct_change()
        # 留 5% 余量：复权因子跳变会造成合理的超限
        bad = chg.abs() > limit * 1.5
        if bad.any():
            n = int(bad.sum())
            worst = float(chg.abs().max())
            issues.append({
                "level": "warn",
                "msg": f"{n} 个交易日涨跌幅超过 {limit:.0%} 的 1.5 倍"
                       f"（最大 {worst:.1%}）—— 多为复权因子跳变或数据错误"})

    # --- 重复日期
    if df.index.duplicated().any():
        issues.append({"level": "error",
                       "msg": f"存在重复日期：{int(df.index.duplicated().sum())} 处"})

    # --- 全空列
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        issues.append({"level": "warn",
                       "msg": f"整列为空：{', '.join(map(str, empty[:6]))}"})

    return issues


def overview() -> dict:
    """各数据源的规模与最后更新时间。"""
    out = []
    for key, (label, rel, desc) in SOURCES.items():
        base = DATA / rel
        if not base.exists():
            out.append({"key": key, "label": label, "desc": desc,
                        "exists": False})
            continue
        if key == "qlib":
            n = sum(1 for d in base.iterdir() if d.is_dir())
            files = list(base.rglob("*.bin"))
        else:
            files = list(base.rglob("*.parquet"))
            n = len({p.stem for p in files})
        size = sum(p.stat().st_size for p in files[:6000])
        mtime = max((p.stat().st_mtime for p in files[:6000]), default=0)
        out.append({
            "key": key, "label": label, "desc": desc, "exists": True,
            "n_symbols": n, "n_files": len(files),
            "size_mb": size / 1024 / 1024,
            "mtime": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            if mtime else "—",
        })
    return {"sources": out, "calendar_len": len(_qlib_calendar())}


def _delist_map() -> dict:
    """qlib 代码 -> 退市日期（未退市为 None）。"""
    import pandas as pd

    p = DATA / "universe" / "universe_full.parquet"
    if not p.exists():
        return {}
    try:
        u = pd.read_parquet(p)
    except (OSError, ValueError):
        return {}
    out = {}
    for r in u.itertuples():
        vt = str(getattr(r, "vt_symbol", ""))
        if "." not in vt:
            continue
        code, ex = vt.split(".")
        q = ("sh" if ex == "SSE" else "sz" if ex == "SZSE" else "") + code
        if not q.startswith(("sh", "sz")):
            continue
        d = getattr(r, "delist_date", None)
        out[q] = None if pd.isna(d) else str(d)[:10]
    return out


def qlib_consistency(sample: int = 0) -> dict:
    """全库一致性扫描：多少标的落后于全局日历。

    这正是之前那次数据撕裂的检测方法 —— `export_qlib --index` 只重写
    该指数的成分股，日历却按本次导出的标的重建，未被重写的标的就会落后。

    **必须区分「退市」与「未更新」**：落后最多的那批多半是退市股，
    它们本就不该有新数据（武钢股份 2017 年退市，数据止于 2017-01-23
    完全正确）。把两者混为一谈会让人去追不存在的问题，
    也会把真正的数据撕裂淹没在几百条噪声里。
    """
    import numpy as np

    cal = _qlib_calendar()
    if not cal:
        return {"error": "无日历文件"}
    base = DATA / "qlib_data" / "features"
    if not base.exists():
        return {"error": "无特征目录"}

    n_cal = len(cal)
    delist = _delist_map()
    fresh = stale = delisted = 0
    ends = {}
    stale_list = []
    dirs = sorted(d for d in base.iterdir() if d.is_dir())
    if sample:
        dirs = dirs[:sample]
    for d in dirs:
        p = d / "close.day.bin"
        if not p.exists():
            continue
        n = (p.stat().st_size - 4) // 4
        try:
            s0 = int(np.fromfile(p, dtype=np.float32, count=1)[0])
        except (OSError, ValueError):
            continue
        end = s0 + n - 1
        ends[end] = ends.get(end, 0) + 1
        if end >= n_cal - 1:
            fresh += 1
            continue

        last_date = cal[end] if 0 <= end < n_cal else "?"
        dd = delist.get(d.name)
        # 已退市的标的本就不该有新数据。数据可以在退市日之前很久就停 ——
        # 退市前普遍先长期停牌，武钢股份退市日 2017-02-14、数据止于
        # 2017-01-23，中间隔 22 天属正常。因此只要没超过退市日就算正常，
        # 不设下限容差
        if dd and last_date != "?" and last_date <= dd[:10]:
            delisted += 1
            continue

        stale += 1
        if len(stale_list) < 40:
            stale_list.append({
                "symbol": d.name,
                "last_date": last_date,
                "behind": n_cal - 1 - end,
                "delist_date": dd,
            })
    stale_list.sort(key=lambda x: -x["behind"])
    return {
        "calendar_len": n_cal,
        "last_date": cal[-1] if cal else "?",
        "fresh": fresh, "stale": stale, "delisted": delisted,
        "total": fresh + stale + delisted,
        "end_dist": sorted(ends.items(), key=lambda x: -x[1])[:6],
        "stale_sample": stale_list,
        "has_delist_data": bool(delist),
    }
