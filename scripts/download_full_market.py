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
    python scripts/download_full_market.py --include-st   # 不排除 ST
    python scripts/download_full_market.py --list-only    # 只看名单不下载
"""
import argparse
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


def main() -> int:
    p = argparse.ArgumentParser(description="全市场行情下载")
    p.add_argument("--sector", default="沪深A股")
    p.add_argument("--start", default="20160101")
    p.add_argument("--end", default="20301231")
    p.add_argument("--include-st", action="store_true",
                    help="不排除 ST。ST 涨跌停 5%%，回测撮合会失真")
    p.add_argument("--list-only", action="store_true", help="只统计名单")
    p.add_argument("--rebuild", action="store_true",
                    help="重新下载已有的标的")
    args = p.parse_args()

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
        print("取不到标的名单 —— 请确认 miniQMT 已启动")
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

    # 已有的跳过
    have = {f"{p.stem}.{p.parent.name}"
            for p in (store / "1d").rglob("*.parquet")}
    todo = symbols if args.rebuild else [s for s in symbols if s not in have]
    print(f"\n本地已有 {len(have)} 只，本次需下载 {len(todo)} 只")

    if args.list_only:
        return 0
    if not todo:
        print("无需下载")
        return 0

    est = len(todo) * 0.6 / 60
    print(f"预计耗时 {est:.0f} ~ {est * 2:.0f} 分钟\n")

    feed = XtDataFeed(cfg.data.store_dir, cfg.data.dividend_type)
    t0 = time.time()
    ok = fail = 0
    for i, vt in enumerate(todo, 1):
        try:
            r = feed.download([vt], args.start, args.end)
            if r.get("ok"):
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  成功 {ok} 失败 {fail}  "
                  f"剩约 {el / i * (len(todo) - i) / 60:.0f} 分钟")

    print(f"\n完成：成功 {ok}，失败 {fail}，"
          f"耗时 {(time.time() - t0) / 60:.1f} 分钟")
    total = len(list((store / "1d").rglob("*.parquet")))
    print(f"本地现有 {total} 只")
    print("\n下一步：")
    print("  python scripts/clean_data.py                    清洗")
    print("  python scripts/export_qlib.py --all             导出（全量）")
    print("  python scripts/build_database.py                入库")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
