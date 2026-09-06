"""全市场行情下载 —— 沪深 A 股，排除 ST。

## 为什么排除 ST

ST 股（风险警示）与普通股的交易规则不同：

- **涨跌停 5%** 而非 10%，撮合行为完全不同
- 流动性差，滑点远高于常规估计
- 退市风险，可能在持仓期间被强制终止交易

留在标的池里会让回测系统性失真 —— 引擎按板块判定涨跌停时
拿不到历史 ST 状态（那是随时间变化的），会用 10% 去撮合本该 5% 的标的。

## ST 的识别

两种途径，本脚本用**股票名称**：

1. QMT 板块「GNST股」—— 但那是**当前**状态，用于历史回测有前视问题
2. 名称含 "ST" 或 "退" —— 同样是当前状态

两者都只能反映当下。历史上某只股票何时被 ST、何时摘帽，QMT 拿不到。
因此这里的排除是「当前是 ST 的一律不下载」，属于**保守处理**：
可能漏掉一些曾经 ST 但现已摘帽的正常标的。

真正无前视的做法需要历史 ST 名单（Tushare 的 namechange 接口可查），
那是后续要补的。当前先保证不把明确的 ST 放进来。

## 规模

沪深 A 股约 5217 只，扣除 ST 约 5000 只。相比中证 1000（1000 只成分、
历史并集 2839 只）大幅增加：

- 磁盘：日线约每只 130 KB，全量约 650 MB
- 下载：约 40~60 分钟
- **训练内存会显著上升** —— 实测中证 1000 峰值 13.76 GB，
  全市场按标的数线性外推约 25~40 GB，31.7 GB 内存可能不够。
  下载不受影响，但训练时要留意

用法::

    python scripts/download_full_market.py
    python scripts/download_full_market.py --intervals 5m  # 全市场 5 分钟线
    python scripts/download_full_market.py --intervals 1d,5m  # 同时下载日线和 5 分钟线
    python scripts/download_full_market.py --include-st   # 不排除 ST
    python scripts/download_full_market.py --list-only    # 只看名单不下载
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def classify(codes: list) -> dict:
    """按名称把标的分成正常 / ST / 退市。"""
    from xtquant import xtdata

    normal, st, delisted, unknown = [], [], [], []
    for c in codes:
        try:
            d = xtdata.get_instrument_detail(c) or {}
        except Exception:
            unknown.append((c, ""))
            continue
        name = str(d.get("InstrumentName", "")).strip()
        if not name:
            unknown.append((c, ""))
        elif "退" in name:
            delisted.append((c, name))
        elif "ST" in name.upper():
            st.append((c, name))
        else:
            normal.append((c, name))
    return {"normal": normal, "st": st,
            "delisted": delisted, "unknown": unknown}


def to_vt(code: str) -> str:
    c, _, ex = code.rpartition(".")
    return f"{c}.{'SSE' if ex == 'SH' else 'SZSE'}"


def local_listed_codes() -> list[str]:
    """从本地 market.db 取在册沪深股票，作为 QMT 板块缓存的兜底。

    新安装或 QMT 板块数据尚未同步时，``get_stock_list_in_sector`` 可能返回
    空列表。行情下载本身仍可直接按代码调用，因此使用本地已建好的股票主表
    继续执行全市场下载；名称分类仍交给 QMT ``get_instrument_detail``。
    """
    db = ROOT / "data" / "market.db"
    if not db.exists():
        return []
    try:
        with sqlite3.connect(db) as con:
            rows = con.execute(
                "SELECT symbol, exchange FROM instrument "
                "WHERE status = 'listed' AND exchange IN ('SSE', 'SZSE') "
                "ORDER BY exchange, symbol"
            ).fetchall()
    except sqlite3.Error:
        return []
    suffix = {"SSE": "SH", "SZSE": "SZ"}
    return [f"{symbol}.{suffix[exchange]}" for symbol, exchange in rows]


def main() -> int:
    p = argparse.ArgumentParser(description="全市场行情下载")
    p.add_argument("--sector", default="沪深A股")
    p.add_argument(
        "--intervals", default="1d",
        help="逗号分隔的周期：1m,5m,15m,30m,1h,1d（默认 1d）",
    )
    p.add_argument("--start", default="20160101")
    p.add_argument("--end", default="20301231")
    p.add_argument("--include-st", action="store_true",
                    help="不排除 ST。ST 涨跌停 5%%，回测撮合会失真")
    p.add_argument("--list-only", action="store_true", help="只统计名单")
    p.add_argument("--rebuild", action="store_true",
                    help="重新下载已有的标的")
    args = p.parse_args()

    from qmtquant.core.constants import Interval

    try:
        intervals = [
            Interval(s.strip())
            for s in args.intervals.split(",")
            if s.strip()
        ]
    except ValueError as e:
        print(f"无效的周期: {e}")
        return 1
    if not intervals:
        print("至少指定一个周期，例如 --intervals 5m")
        return 1

    from xtquant import xtdata

    from qmtquant.config import get_config
    from qmtquant.datafeed.xt_feed import XtDataFeed

    cfg = get_config()
    store = Path(cfg.data.store_dir)

    print("=" * 60)
    print(f"全市场行情下载  板块 {args.sector}")
    print("=" * 60)

    codes = xtdata.get_stock_list_in_sector(args.sector) or []
    if not codes:
        # QMT 板块缓存为空并不代表无法下载。项目的日线数据库保存了
        # 当前在册沪深股票主表，可作为全市场下载的稳定兜底来源。
        codes = local_listed_codes()
        if codes:
            print(f"QMT 板块缓存为空，改用本地 instrument 表：{len(codes)} 只")
        else:
            print("取不到标的名单 —— 请确认 miniQMT 已启动，或检查 data/market.db")
            return 1
    print(f"\n板块内 {len(codes)} 只，正在分类...")

    cls = classify(codes)
    print(f"  正常   {len(cls['normal']):>5d} 只")
    print(f"  ST     {len(cls['st']):>5d} 只")
    print(f"  已退市 {len(cls['delisted']):>5d} 只")
    if cls["unknown"]:
        print(f"  未知   {len(cls['unknown']):>5d} 只（取不到名称）")

    if args.include_st:
        picked = cls["normal"] + cls["st"]
        print(f"\n包含 ST：共 {len(picked)} 只")
        print("  [注意] ST 涨跌停 5%%，回测按 10%% 撮合会系统性失真")
    else:
        picked = cls["normal"]
        print(f"\n排除 ST 与已退市：共 {len(picked)} 只")

    if cls["st"][:5]:
        print(f"  排除的 ST 样例: "
              f"{', '.join(n for _, n in cls['st'][:5])}")

    symbols = [to_vt(c) for c, _ in picked]

    # 每个周期独立判断本地库存。XtDataFeed._path 会自动创建
    # data/{周期}/{交易所}/ 目录，因此 5m 首次运行无需手工建目录。
    plans = []
    print()
    for interval in intervals:
        interval_dir = store / interval.value
        have = {
            f"{p.stem}.{p.parent.name}"
            for p in interval_dir.rglob("*.parquet")
        } if interval_dir.exists() else set()
        todo = symbols if args.rebuild else [s for s in symbols if s not in have]
        plans.append((interval, todo))
        print(f"[{interval.value}] 本地已有 {len(have)} 只，本次需下载 {len(todo)} 只")

    if args.list_only:
        return 0
    if not any(todo for _, todo in plans):
        print("无需下载")
        return 0

    # 这里只给出数量级提示；实际耗时主要取决于 QMT 客户端和网络。
    minute_intervals = {"1m": 1.85, "5m": 0.37, "15m": 0.13,
                        "30m": 0.07, "1h": 0.04}
    est_mb = sum(
        len(todo) * minute_intervals.get(interval.value, 0.13)
        for interval, todo in plans
    )
    if est_mb >= 1024:
        print(f"预估磁盘占用：约 {est_mb / 1024:.1f} GB\n")
    else:
        print(f"预估磁盘占用：约 {est_mb:.0f} MB\n")

    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    t0 = time.time()

    def on_progress(done: int, total: int, cur: str) -> None:
        if done % 100 and done != total:
            return
        el = time.time() - t0
        eta = el / max(done, 1) * (total - done) / 60
        print(f"  {done}/{total}  {cur:<14s} 剩约 {eta:.0f} 分钟")

    # download_history 自带批量、断点续传与进度回调。
    # 逐个调用 download()（那是 IndexFeed 的方法）会全部失败。
    all_failed = {}
    for interval, todo in plans:
        if not todo:
            continue
        print(f"\n>>> {interval.value}")
        r = feed.download_history(
            todo, args.start, args.end, interval=interval,
            skip_existing=not args.rebuild, progress=on_progress,
        )
        ok, fail = len(r.get("ok", [])), len(r.get("failed", []))
        skipped = len(r.get("skipped", []))
        print(f"\n完成 {interval.value}：成功 {ok}，失败 {fail}，跳过 {skipped}")
        if fail:
            all_failed[interval.value] = r["failed"]

    print(f"\n全部完成，耗时 {(time.time() - t0) / 60:.1f} 分钟")
    for interval, _ in plans:
        total = len(list((store / interval.value).rglob("*.parquet")))
        print(f"[{interval.value}] 本地现有 {total} 只")
    if all_failed:
        print("\n以下周期有下载失败：")
        for period, failed in all_failed.items():
            print(f"  [{period}] {', '.join(failed[:5])}"
                  + (f" ... 等 {len(failed)} 只" if len(failed) > 5 else ""))
    print("\n下一步：")
    print("  python scripts/clean_data.py                    清洗")
    print("  python scripts/export_qlib.py --all             导出（全量）")
    print("  python scripts/build_database.py                入库")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
