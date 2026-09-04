"""数据清洗层 —— 原始数据的唯一净化出口。

## 为什么要有独立的一层

此前清洗散落在各个消费方：`qlib_export` 清一次、`build_database` 再清一次。
虽然调的是同一个函数，但有三个问题：

1. **结果不落盘**，每次消费都要重算
2. **谁忘了调就读到脏数据** —— 训练链路曾因此吃了 25000 多行非正价格
3. 新增消费方时容易漏掉这一步

改为：原始层保持不动，清洗产出独立的一层，**所有下游只读清洗层**。

    data/1d/          原始下载，不可变，可随时回溯
            │
            ▼  clean_data.py
    data/clean/1d/    清洗层，唯一真相源
            │
            ├──> qlib_data/   训练与回测（快）
            └──> market.db    查询与核对（灵活）

原始层保留的意义：清洗规则会演进，改了规则要能重新清洗；
若直接覆盖原始数据，规则一改就无从追溯。

## 增量

按源文件 mtime 判断，未变的标的直接跳过。首次全量约几分钟，
之后每日只处理当天下载过的标的。

用法::

    python scripts/clean_data.py              # 增量
    python scripts/clean_data.py --rebuild    # 全量重清
    python scripts/clean_data.py --report     # 只看上次清洗报告
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: 清洗层目录名。与原始层同级，互不覆盖
CLEAN_DIRNAME = "clean"
STATE_FILE = "clean_state.json"
REPORT_FILE = "clean_report.json"


def clean_dir(store: Path) -> Path:
    return store / CLEAN_DIRNAME


def load_state(store: Path) -> dict:
    p = clean_dir(store) / STATE_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(store: Path, state: dict) -> None:
    d = clean_dir(store)
    d.mkdir(parents=True, exist_ok=True)
    (d / STATE_FILE).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_bars_layer(store: Path, rebuild: bool, limit: int = 0,
                     interval: str = "1d") -> dict:
    """把 data/<interval>/ 清洗到 data/clean/<interval>/。"""
    import pandas as pd

    from qmtquant.datafeed.cleaner import clean_bars, normalize_bars

    src_root = store / interval
    dst_root = clean_dir(store) / interval
    if not src_root.exists():
        return {"error": f"源目录不存在: {src_root}"}

    files = sorted(src_root.rglob("*.parquet"))
    if limit:
        files = files[:limit]

    state = {} if rebuild else load_state(store).get(interval, {})
    new_state = dict(state)

    stats = {"symbols": 0, "skipped": 0, "rows_in": 0, "rows_out": 0}
    rules = {}
    flags = {}
    per_symbol = []

    t0 = time.time()
    for i, p in enumerate(files, 1):
        vt = f"{p.stem}.{p.parent.name}"
        mtime = p.stat().st_mtime
        if not rebuild and abs(state.get(vt, {}).get("mtime", -1) - mtime) < 1e-6:
            stats["skipped"] += 1
            continue

        try:
            raw = pd.read_parquet(p)
        except (OSError, ValueError) as e:
            per_symbol.append({"vt_symbol": vt, "error": str(e)})
            continue

        df = normalize_bars(raw)
        n_in = len(df)
        if df.empty:
            continue

        res = clean_bars(df, vt)
        out = res.df

        dst = dst_root / p.parent.name
        dst.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dst / f"{p.stem}.parquet")

        stats["symbols"] += 1
        stats["rows_in"] += n_in
        stats["rows_out"] += len(out)
        for a in res.actions:
            rules[a.rule] = rules.get(a.rule, 0) + a.n_rows
        for f in res.flags:
            flags[f.rule] = flags.get(f.rule, 0) + f.n_rows
        if res.actions or res.flags:
            per_symbol.append({
                "vt_symbol": vt,
                "rows_in": n_in, "rows_out": len(out),
                "actions": [{"rule": a.rule, "n": a.n_rows} for a in res.actions],
                "flags": [{"rule": f.rule, "n": f.n_rows} for f in res.flags],
            })

        new_state[vt] = {"mtime": mtime, "rows": len(out),
                         "cleaned_at": datetime.now().isoformat(timespec="seconds")}

        if i % 400 == 0:
            print(f"  {i}/{len(files)}  已清洗 {stats['symbols']} 只，"
                  f"跳过 {stats['skipped']}  ({time.time() - t0:.0f}s)")

    full = load_state(store)
    full[interval] = new_state
    save_state(store, full)

    return {**stats, "rules": rules, "flags": flags,
            "per_symbol": per_symbol, "elapsed": time.time() - t0}


def write_report(store: Path, result: dict, interval: str) -> Path:
    d = clean_dir(store)
    d.mkdir(parents=True, exist_ok=True)
    p = d / REPORT_FILE
    old = {}
    if p.exists():
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            old = {}
    old[interval] = {
        "cleaned_at": datetime.now().isoformat(timespec="seconds"),
        "symbols": result.get("symbols", 0),
        "skipped": result.get("skipped", 0),
        "rows_in": result.get("rows_in", 0),
        "rows_out": result.get("rows_out", 0),
        "rules": result.get("rules", {}),
        "flags": result.get("flags", {}),
        # 明细只留问题最多的，避免报告文件无限膨胀
        "worst": sorted(
            [s for s in result.get("per_symbol", []) if s.get("actions")],
            key=lambda s: -sum(a["n"] for a in s["actions"]))[:50],
    }
    p.write_text(json.dumps(old, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def show_report(store: Path) -> int:
    p = clean_dir(store) / REPORT_FILE
    if not p.exists():
        print("尚无清洗报告，请先运行 python scripts/clean_data.py")
        return 1
    rep = json.loads(p.read_text(encoding="utf-8"))
    for interval, r in rep.items():
        print(f"\n=== {interval} ===")
        print(f"  清洗于 {r['cleaned_at']}")
        print(f"  标的 {r['symbols']:,} 只（跳过未变更 {r['skipped']:,}）")
        print(f"  行数 {r['rows_in']:,} -> {r['rows_out']:,}"
              f"（剔除 {r['rows_in'] - r['rows_out']:,}）")
        if r["rules"]:
            print("  已修改:")
            for k, v in sorted(r["rules"].items(), key=lambda x: -x[1]):
                print(f"    {k:<28s} {v:>10,} 行")
        if r["flags"]:
            print("  仅标记（未修改，交由下游判断）:")
            for k, v in sorted(r["flags"].items(), key=lambda x: -x[1]):
                print(f"    {k:<28s} {v:>10,} 行")
        if r.get("worst"):
            print(f"  问题最多的标的:")
            for s in r["worst"][:8]:
                acts = ", ".join(f"{a['rule']} {a['n']}" for a in s["actions"])
                print(f"    {s['vt_symbol']:<14s} {s['rows_in']:>6d} -> "
                      f"{s['rows_out']:<6d}  {acts}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="数据清洗层")
    p.add_argument("--rebuild", action="store_true", help="全量重清")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 只，用于验证")
    p.add_argument("--intervals", nargs="*", default=["1d"])
    p.add_argument("--report", action="store_true", help="只显示上次报告")
    args = p.parse_args()

    from qmtquant.config import get_config
    store = Path(get_config().data.store_dir)

    if args.report:
        return show_report(store)

    print("=" * 62)
    print("数据清洗")
    print(f"  原始层 : {store}")
    print(f"  清洗层 : {clean_dir(store)}")
    print(f"  模式   : {'全量重清' if args.rebuild else '增量'}")
    print("=" * 62)

    for interval in args.intervals:
        print(f"\n[{interval}]")
        r = clean_bars_layer(store, args.rebuild, args.limit, interval)
        if "error" in r:
            print(f"  {r['error']}")
            continue
        print(f"  清洗 {r['symbols']:,} 只，跳过未变更 {r['skipped']:,} 只")
        print(f"  行数 {r['rows_in']:,} -> {r['rows_out']:,}"
              f"（剔除 {r['rows_in'] - r['rows_out']:,}）")
        if r["rules"]:
            print("  已修改:")
            for k, v in sorted(r["rules"].items(), key=lambda x: -x[1]):
                print(f"    {k:<28s} {v:>10,} 行")
        if r["flags"]:
            print("  仅标记:")
            for k, v in sorted(r["flags"].items(), key=lambda x: -x[1]):
                print(f"    {k:<28s} {v:>10,} 行")
        rp = write_report(store, r, interval)
        print(f"  耗时 {r['elapsed'] / 60:.1f} 分钟")
        print(f"  报告: {rp}")

    print("\n下游现在应读取清洗层：")
    print(f"  {clean_dir(store) / '1d'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
