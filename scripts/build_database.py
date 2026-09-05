"""把下载的数据清洗后写入 SQLite —— 可查询、可核对、可用外部工具打开。

## 为什么值得建

原始存储是**每个标的一个 parquet**（日线 3274 个文件）。这个布局适合
按标的读取，但做不了截面查询：「2026-08-28 涨幅前 20 的股票」这种问题
要打开三千多个文件。数据库把这类查询变成一条 SQL。

同时这是数据清洗真正落地的地方。此前 `validator.py` 只在手动脚本里被调用，
下载流程完全没接，文档里提到的 `clean()` 甚至没有实现 ——
也就是说下载至今的数据从未被清洗过。

## 清洗与「修复」的界限

只修有唯一正确答案的（去重、排序、负成交量置零、剔除非正价格）；
需要猜测的一律只标记不修改（OHLC 矛盾、涨跌幅异动、缺失交易日）。
详见 `qmtquant/datafeed/cleaner.py`。

所有清洗动作与标记都写进 `data_issue` 表，可随时回查「这份数据被改过什么」。

## 增量重建

按源文件 mtime 判断是否需要重新处理，未变的标的直接跳过。
全量首次约需几分钟，之后每日增量只处理当天下载过的标的。

用法::

    python scripts/build_database.py                 # 增量
    python scripts/build_database.py --rebuild       # 全量重建
    python scripts/build_database.py --tables bars   # 只建某几张表
"""
import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "market.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bar (
    vt_symbol   TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    open        REAL, high REAL, low REAL, close REAL,
    volume      INTEGER,
    amount      REAL,
    pre_close   REAL,
    suspended   INTEGER DEFAULT 0,
    PRIMARY KEY (vt_symbol, date)
);
CREATE INDEX IF NOT EXISTS ix_bar_date ON daily_bar(date);

CREATE TABLE IF NOT EXISTS instrument (
    vt_symbol   TEXT PRIMARY KEY,
    symbol      TEXT,
    exchange    TEXT,
    name        TEXT,
    industry    TEXT,
    area        TEXT,
    list_date   TEXT,
    delist_date TEXT,
    status      TEXT
);

CREATE TABLE IF NOT EXISTS index_weight (
    index_code  TEXT NOT NULL,
    date        TEXT NOT NULL,
    vt_symbol   TEXT NOT NULL,
    weight      REAL,
    PRIMARY KEY (index_code, date, vt_symbol)
);
CREATE INDEX IF NOT EXISTS ix_weight_sym ON index_weight(vt_symbol);

-- 稀疏存储：只存非零值。dragon_count_20 有 6451 只 x 2590 日 = 1670 万行，
-- 其中 84% 是 0（当日无标的上榜）。密集存储会让库膨胀到 2.3 GB 且 99%
-- 是零。缺失的语义记在 factor_meta.fill_value 里，读取时按该值补齐即可，
-- 信息无损。
CREATE TABLE IF NOT EXISTS money_factor (
    factor      TEXT NOT NULL,
    date        TEXT NOT NULL,
    vt_symbol   TEXT NOT NULL,
    value       REAL,
    PRIMARY KEY (factor, date, vt_symbol)
);
CREATE INDEX IF NOT EXISTS ix_factor_date ON money_factor(date);

CREATE TABLE IF NOT EXISTS factor_meta (
    factor      TEXT PRIMARY KEY,
    fill_value  REAL,        -- 表中缺失的行应补成这个值
    n_stored    INTEGER,
    n_dense     INTEGER,     -- 密集存储本应有多少行
    first_date  TEXT,
    last_date   TEXT,
    avail_lag   INTEGER,     -- 可获得滞后（交易日），用于避免前视偏差
    note        TEXT
);

-- 清洗动作与仅标记的问题，全部留痕
CREATE TABLE IF NOT EXISTS data_issue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT,
    vt_symbol   TEXT,
    rule        TEXT,
    n_rows      INTEGER,
    modified    INTEGER,     -- 1=已修改数据  0=仅标记
    detail      TEXT,
    detected_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_issue_sym ON data_issue(vt_symbol);

-- 清洗动作汇总，从 data/clean/clean_report.json 导入。
-- 这是「数据到底洗掉了什么」的可查询记录 —— 只看日志无法交叉核对
CREATE TABLE IF NOT EXISTS clean_action (
    dataset     TEXT NOT NULL,     -- 1d / financial / money_flow ...
    rule        TEXT NOT NULL,
    modified    INTEGER,           -- 1=已修改数据  0=仅标记
    n_rows      INTEGER,
    cleaned_at  TEXT,
    PRIMARY KEY (dataset, rule, modified)
);

-- 逐标的的清洗明细，便于定位「哪只股票被洗掉最多」
CREATE TABLE IF NOT EXISTS clean_detail (
    dataset     TEXT NOT NULL,
    vt_symbol   TEXT NOT NULL,
    rows_in     INTEGER,
    rows_out    INTEGER,
    rule        TEXT,
    n_rows      INTEGER,
    PRIMARY KEY (dataset, vt_symbol, rule)
);

-- 增量重建用：记录每个标的的源文件 mtime
CREATE TABLE IF NOT EXISTS build_state (
    source      TEXT NOT NULL,
    key         TEXT NOT NULL,
    src_mtime   REAL,
    n_rows      INTEGER,
    built_at    TEXT,
    PRIMARY KEY (source, key)
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    # WAL 让写入期间仍可并发读；synchronous=NORMAL 在批量写入时快得多，
    # 崩溃最多丢最后一个事务 —— 数据可从 parquet 重建，代价可接受
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def load_build_state(conn, source: str) -> dict:
    cur = conn.execute(
        "SELECT key, src_mtime FROM build_state WHERE source = ?", (source,))
    return dict(cur.fetchall())


def record_issues(conn, source: str, vt_symbol: str, result) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    rows = [(source, vt_symbol, a.rule, a.n_rows, 1, a.detail, now)
            for a in result.actions]
    rows += [(source, vt_symbol, f.rule, f.n_rows, 0, f.detail, now)
             for f in result.flags]
    if rows:
        conn.executemany(
            "INSERT INTO data_issue"
            "(source,vt_symbol,rule,n_rows,modified,detail,detected_at)"
            " VALUES (?,?,?,?,?,?,?)", rows)


def build_bars(conn, store: Path, rebuild: bool, limit: int = 0) -> dict:
    """日线：清洗后写入 daily_bar。"""
    import pandas as pd

    from qmtquant.datafeed.cleaner import clean_bars, normalize_bars

    # 优先读清洗层 —— 与 qlib 导出共用同一份数据，避免两条链路各清各的
    clean_root = store / "clean" / "1d"
    use_clean = clean_root.exists() and any(clean_root.rglob("*.parquet"))
    src_root = clean_root if use_clean else store / "1d"
    print(f"  数据源: {'清洗层' if use_clean else '原始层（将就地清洗）'}"
          f"  {src_root}")

    files = sorted(src_root.rglob("*.parquet"))
    if limit:
        files = files[:limit]
    state = {} if rebuild else load_build_state(conn, "1d")

    n_sym = n_rows = n_skip = 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        vt = f"{p.stem}.{p.parent.name}"
        mtime = p.stat().st_mtime
        if not rebuild and abs(state.get(vt, -1) - mtime) < 1e-6:
            n_skip += 1
            continue

        try:
            raw = pd.read_parquet(p)
        except (OSError, ValueError):
            continue

        if use_clean:
            # 清洗层已是成品，不再重复清洗
            df = raw
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()].sort_index()
        else:
            df = normalize_bars(raw)
            if df.empty:
                continue
            res = clean_bars(df, vt)
            df = res.df
            record_issues(conn, "1d", vt, res)
        if df.empty:
            continue

        # 旧记录先删再插，避免清洗后行数变少时残留脏数据
        conn.execute("DELETE FROM daily_bar WHERE vt_symbol = ?", (vt,))
        cols = ["open", "high", "low", "close", "volume", "amount",
                "preClose", "suspendFlag"]
        have = [c for c in cols if c in df.columns]
        recs = []
        for d, r in df[have].iterrows():
            recs.append((
                vt, d.strftime("%Y-%m-%d"),
                _f(r.get("open")), _f(r.get("high")),
                _f(r.get("low")), _f(r.get("close")),
                _i(r.get("volume")), _f(r.get("amount")),
                _f(r.get("preClose")), _i(r.get("suspendFlag")) or 0,
            ))
        conn.executemany(
            "INSERT OR REPLACE INTO daily_bar"
            "(vt_symbol,date,open,high,low,close,volume,amount,"
            "pre_close,suspended) VALUES (?,?,?,?,?,?,?,?,?,?)", recs)
        conn.execute(
            "INSERT OR REPLACE INTO build_state"
            "(source,key,src_mtime,n_rows,built_at) VALUES (?,?,?,?,?)",
            ("1d", vt, mtime, len(recs),
             datetime.now().isoformat(timespec="seconds")))

        n_sym += 1
        n_rows += len(recs)
        if i % 200 == 0:
            conn.commit()
            print(f"  {i}/{len(files)}  已写 {n_sym} 只 / {n_rows:,} 行"
                  f"  跳过 {n_skip}  ({time.time() - t0:.0f}s)")
    conn.commit()
    return {"symbols": n_sym, "rows": n_rows, "skipped": n_skip}


def _f(v):
    try:
        import math
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _i(v):
    f = _f(v)
    return None if f is None else int(f)


def build_instruments(conn, store: Path) -> dict:
    import pandas as pd

    p = store / "universe" / "universe_full.parquet"
    if not p.exists():
        return {"rows": 0}
    u = pd.read_parquet(p)

    ind = {}
    pi = store / "universe" / "industry.parquet"
    if pi.exists():
        d = pd.read_parquet(pi)
        ind = {r.vt_symbol: (getattr(r, "industry", ""), getattr(r, "area", ""))
               for r in d.itertuples()}

    recs = []
    for r in u.itertuples():
        vt = str(getattr(r, "vt_symbol", ""))
        if "." not in vt:
            continue
        code, ex = vt.split(".")
        i, a = ind.get(vt, ("", ""))
        recs.append((
            vt, code, ex, getattr(r, "name", ""), i, a,
            _d(getattr(r, "listing_date", None)),
            _d(getattr(r, "delist_date", None)),
            getattr(r, "status", ""),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO instrument"
        "(vt_symbol,symbol,exchange,name,industry,area,"
        "list_date,delist_date,status) VALUES (?,?,?,?,?,?,?,?,?)", recs)
    conn.commit()
    return {"rows": len(recs)}


def _d(v):
    import pandas as pd
    if v is None or (hasattr(pd, "isna") and pd.isna(v)):
        return None
    return str(v)[:10]


def build_index_weight(conn, store: Path) -> dict:
    import pandas as pd

    n = 0
    for p in sorted((store / "universe").glob("index_weight_*.csv")):
        code = p.stem.replace("index_weight_", "")
        try:
            df = pd.read_csv(p, parse_dates=["date"])
        except (OSError, ValueError):
            continue
        recs = [(code, r.date.strftime("%Y-%m-%d"), str(r.symbol),
                 _f(getattr(r, "weight", None)))
                for r in df.itertuples()]
        conn.executemany(
            "INSERT OR REPLACE INTO index_weight"
            "(index_code,date,vt_symbol,weight) VALUES (?,?,?,?)", recs)
        n += len(recs)
        print(f"  {code}: {len(recs):,} 行")
    conn.commit()
    return {"rows": n}


#: 各因子的可获得滞后（交易日）。与 eval_factors.FACTOR_LAG 保持一致 ——
#: 因子按 trade_date 索引，那是数据「所属」日期而非「拿得到」日期，
#: 不滞后使用即为前视偏差。写进 factor_meta 供任何读取方参照
FACTOR_LAG = {
    "dragon_count_20": 1, "dragon_inst_net": 1,
    "margin_buy_ratio": 2, "margin_bal_chg": 2, "north_net_pct": 1,
}

#: 缺失值应补成什么。计数型因子缺失即 0 次，其余因子缺失是真的未知
FACTOR_FILL = {
    "dragon_count_20": 0.0,
    "dragon_inst_net": 0.0,
}


def build_money_factors(conn, store: Path) -> dict:
    import pandas as pd

    from qmtquant.datafeed.qlib_export import from_qlib_code

    d = store / "qlib_data" / "money_factors"
    if not d.exists():
        return {"rows": 0}
    total = 0
    for p in sorted(d.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except (OSError, ValueError):
            continue
        s = df.iloc[:, 0].dropna()
        n_dense = len(s)

        # 稀疏化：等于填充值的行不存，读取时补回即可，信息无损
        fill = FACTOR_FILL.get(p.stem)
        if fill is not None:
            s = s[s != fill]

        conn.execute("DELETE FROM money_factor WHERE factor = ?", (p.stem,))
        recs, n = [], 0
        dates = []
        for (dt, inst), v in s.items():
            try:
                vt = from_qlib_code(str(inst))
            except ValueError:
                continue
            ds = pd.Timestamp(dt).strftime("%Y-%m-%d")
            dates.append(ds)
            recs.append((p.stem, ds, vt, float(v)))
            if len(recs) >= 200_000:
                conn.executemany(
                    "INSERT OR REPLACE INTO money_factor"
                    "(factor,date,vt_symbol,value) VALUES (?,?,?,?)", recs)
                n += len(recs)
                recs = []
        if recs:
            conn.executemany(
                "INSERT OR REPLACE INTO money_factor"
                "(factor,date,vt_symbol,value) VALUES (?,?,?,?)", recs)
            n += len(recs)

        conn.execute(
            "INSERT OR REPLACE INTO factor_meta"
            "(factor,fill_value,n_stored,n_dense,first_date,last_date,"
            "avail_lag,note) VALUES (?,?,?,?,?,?,?,?)",
            (p.stem, fill, n, n_dense,
             min(dates) if dates else None, max(dates) if dates else None,
             FACTOR_LAG.get(p.stem, 1),
             "稀疏存储，缺失补 fill_value" if fill is not None else "全量存储"))
        conn.commit()
        total += n
        pct = f"（密集 {n_dense:,}，省 {1 - n / n_dense:.0%}）" if fill is not None and n_dense else ""
        print(f"  {p.stem:<20s} {n:>10,} 行 {pct}")
    return {"rows": total}


def build_clean_report(conn, store: Path) -> dict:
    """把清洗报告导入数据库，让「洗掉了什么」变成可查询的。

    只看日志无法交叉核对 —— 比如「某只股票被洗掉 2569 行」这条，
    要能和 daily_bar 里该标的的实际行数对上，才谈得上验证清洗是否正确。
    """
    import json as _json

    p = store / "clean" / "clean_report.json"
    if not p.exists():
        return {"error": "无清洗报告，请先运行 scripts/clean_data.py"}
    try:
        rep = _json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"error": f"报告读取失败: {e}"}

    conn.execute("DELETE FROM clean_action")
    conn.execute("DELETE FROM clean_detail")
    n_act = n_det = 0
    for dataset, r in rep.items():
        ts = r.get("cleaned_at", "")
        for rule, n in (r.get("rules") or {}).items():
            conn.execute(
                "INSERT OR REPLACE INTO clean_action"
                "(dataset,rule,modified,n_rows,cleaned_at) VALUES (?,?,?,?,?)",
                (dataset, rule, 1, int(n), ts))
            n_act += 1
        for rule, n in (r.get("flags") or {}).items():
            conn.execute(
                "INSERT OR REPLACE INTO clean_action"
                "(dataset,rule,modified,n_rows,cleaned_at) VALUES (?,?,?,?,?)",
                (dataset, rule, 0, int(n), ts))
            n_act += 1
        for s in r.get("worst") or []:
            for a in s.get("actions") or []:
                conn.execute(
                    "INSERT OR REPLACE INTO clean_detail"
                    "(dataset,vt_symbol,rows_in,rows_out,rule,n_rows)"
                    " VALUES (?,?,?,?,?,?)",
                    (dataset, s["vt_symbol"], s.get("rows_in"),
                     s.get("rows_out"), a["rule"], a["n"]))
                n_det += 1
    conn.commit()
    return {"actions": n_act, "details": n_det, "datasets": list(rep)}


def build_financial(conn, store: Path, limit: int = 0) -> dict:
    """财报入库。各报表列数不同（Income 有 86 列），用 to_sql 自动建表。"""
    import pandas as pd

    root = store / "clean" / "financial"
    if not root.exists():
        return {"error": "清洗层无财报，请先运行 clean_data.py"}

    by_table = {}
    for p in sorted(root.rglob("*.parquet")):
        by_table.setdefault(p.parent.parent.name, []).append(p)

    total = 0
    for table, files in by_table.items():
        if limit:
            files = files[:limit]
        tname = f"fin_{table.lower()}"
        conn.execute(f"DROP TABLE IF EXISTS {tname}")
        n = 0
        buf = []
        for p in files:
            try:
                df = pd.read_parquet(p)
            except (OSError, ValueError):
                continue
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "vt_symbol", f"{p.stem}.{p.parent.name}")
            buf.append(df)
            if len(buf) >= 300:
                out = pd.concat(buf, ignore_index=True)
                out.to_sql(tname, conn, if_exists="append", index=False)
                n += len(out)
                buf = []
        if buf:
            out = pd.concat(buf, ignore_index=True)
            out.to_sql(tname, conn, if_exists="append", index=False)
            n += len(out)
        if n:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS ix_{tname}_sym "
                f"ON {tname}(vt_symbol)")
        conn.commit()
        total += n
        print(f"  {tname:<20s} {n:>9,} 行")
    return {"rows": total, "tables": list(by_table)}


def build_money_flow(conn, store: Path) -> dict:
    """资金流原始数据入库（龙虎榜、两融明细）。"""
    import pandas as pd

    root = store / "clean" / "money_flow"
    if not root.exists():
        return {"error": "清洗层无资金流"}

    total = 0
    for p in sorted(root.glob("*.parquet")):
        tname = f"flow_{p.stem.lower()}"
        try:
            df = pd.read_parquet(p)
        except (OSError, ValueError):
            continue
        conn.execute(f"DROP TABLE IF EXISTS {tname}")
        # 单表最大 218 万行，分块写避免内存峰值
        step = 200_000
        for i in range(0, len(df), step):
            df.iloc[i:i + step].to_sql(tname, conn, if_exists="append",
                                       index=False)
        if "trade_date" in df.columns:
            conn.execute(f"CREATE INDEX IF NOT EXISTS ix_{tname}_date "
                         f"ON {tname}(trade_date)")
        conn.commit()
        total += len(df)
        print(f"  {tname:<24s} {len(df):>9,} 行")
    return {"rows": total}


def verify(conn, store: Path) -> None:
    """清洗效果核对 —— 用 SQL 直接验证脏数据是否真的没了。"""
    print("\n" + "=" * 60)
    print("清洗效果核对")
    print("=" * 60)

    checks = [
        ("日线非正价格", """
            SELECT COUNT(*) FROM daily_bar
            WHERE open<=0 OR high<=0 OR low<=0 OR close<=0"""),
        ("日线最高<最低", """
            SELECT COUNT(*) FROM daily_bar WHERE high < low"""),
        ("日线成交量为负", """
            SELECT COUNT(*) FROM daily_bar WHERE volume < 0"""),
        ("日线重复(标的+日期)", """
            SELECT COUNT(*)-COUNT(DISTINCT vt_symbol||date) FROM daily_bar"""),
    ]
    for name, sql in checks:
        try:
            n = conn.execute(sql).fetchone()[0]
        except Exception as e:
            print(f"  {name:<24s} 查询失败: {e}")
            continue
        mark = "通过" if n == 0 else f"**残留 {n:,} 行**"
        print(f"  {name:<24s} {mark}")

    # 财报时间逻辑
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM fin_income "
            "WHERE m_anntime < m_timetag").fetchone()[0]
        print(f"  {'财报公告日早于报告期':<24s} "
              f"{'通过' if n == 0 else f'**残留 {n:,} 行**'}")
    except Exception:
        pass

    # 日期对齐检查。分批下载时新老标的的最后交易日会不一致 ——
    # 实测全市场扩容后 2039 只到 09-04、2489 只停在 08-31、625 只更早。
    # 截面策略在这种数据上会静默漏掉大半标的池，必须查出来
    try:
        mx = conn.execute("SELECT MAX(date) FROM daily_bar").fetchone()[0]
        rows = conn.execute(
            "SELECT COUNT(*) FROM (SELECT vt_symbol, MAX(date) d "
            "FROM daily_bar GROUP BY vt_symbol) WHERE d < ?", (mx,)).fetchall()
        stale = rows[0][0] if rows else 0
        total = conn.execute(
            "SELECT COUNT(DISTINCT vt_symbol) FROM daily_bar").fetchone()[0]
        print(f"\n数据新鲜度: 最新 {mx}，"
              f"{total - stale}/{total} 只已跟上")
        if stale:
            print(f"  !! {stale} 只落后于最新交易日")
            print(f"     退市股属正常；在市股票落后说明需补下载")
    except Exception:
        pass

    print("\n清洗动作汇总（可用 SQL 交叉核对）:")
    rows = conn.execute(
        "SELECT dataset, rule, modified, n_rows FROM clean_action "
        "ORDER BY modified DESC, n_rows DESC LIMIT 12").fetchall()
    for ds, rule, mod, n in rows:
        tag = "已修改" if mod else "仅标记"
        print(f"  [{tag}] {ds:<12s} {rule:<26s} {n:>10,}")


def main() -> int:
    p = argparse.ArgumentParser(description="构建市场数据库")
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--rebuild", action="store_true", help="全量重建")
    p.add_argument("--tables", nargs="*",
                    default=["bars", "instrument", "weight", "factor",
                             "financial", "flow", "clean"],
                    help="要构建的表")
    p.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 个标的，用于快速验证")
    args = p.parse_args()

    from qmtquant.config import get_config
    store = Path(get_config().data.store_dir)
    db = Path(args.db)

    print("=" * 60)
    print("构建市场数据库")
    print(f"  源目录 : {store}")
    print(f"  数据库 : {db}")
    print(f"  模式   : {'全量重建' if args.rebuild else '增量'}")
    print("=" * 60)

    conn = connect(db)
    if args.rebuild:
        print("\n清空旧数据...")
        for t in ("daily_bar", "instrument", "index_weight",
                  "money_factor", "data_issue", "build_state"):
            conn.execute(f"DELETE FROM {t}")
        conn.commit()

    t0 = time.time()
    if "instrument" in args.tables:
        print("\n[标的基础信息]")
        print(f"  {build_instruments(conn, store)['rows']:,} 只")
    if "weight" in args.tables:
        print("\n[指数成分]")
        build_index_weight(conn, store)
    if "factor" in args.tables:
        print("\n[资金流因子]")
        build_money_factors(conn, store)
    if "clean" in args.tables:
        print("\n[清洗记录]")
        r = build_clean_report(conn, store)
        if "error" in r:
            print(f"  {r['error']}")
        else:
            print(f"  动作 {r['actions']} 条，明细 {r['details']} 条，"
                  f"覆盖 {', '.join(r['datasets'])}")
    if "financial" in args.tables:
        print("\n[财务报表]")
        r = build_financial(conn, store, args.limit)
        if "error" in r:
            print(f"  {r['error']}")
    if "flow" in args.tables:
        print("\n[资金流原始数据]")
        r = build_money_flow(conn, store)
        if "error" in r:
            print(f"  {r['error']}")
    if "bars" in args.tables:
        print("\n[日线]（清洗后写入）")
        r = build_bars(conn, store, args.rebuild, args.limit)
        print(f"  写入 {r['symbols']:,} 只 / {r['rows']:,} 行，"
              f"跳过未变更 {r['skipped']:,} 只")

    # ---- 汇总
    print("\n" + "=" * 60)
    print("数据库概况")
    print("=" * 60)
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "ORDER BY name").fetchall()]
    for t in names:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<22s} {n:>12,} 行")

    verify(conn, store)

    print("\n清洗动作统计:")
    for rule, n, mod in conn.execute(
            "SELECT rule, SUM(n_rows), modified FROM data_issue "
            "GROUP BY rule, modified ORDER BY SUM(n_rows) DESC LIMIT 12"):
        tag = "已修改" if mod else "仅标记"
        print(f"  [{tag}] {rule:<28s} {n:>10,} 行")

    conn.execute("PRAGMA optimize")
    conn.close()
    size = db.stat().st_size / 1024 / 1024 if db.exists() else 0
    print(f"\n数据库大小 {size:.0f} MB   耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"路径: {db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
