"""把本地行情导出成 Qlib 数据格式。

## 为什么值得接 Qlib

本项目自研的回测框架能跑通全流程，但在两件事上不如 Qlib：

1. **机器学习模型库** —— LightGBM / LSTM / GATs / Transformer 等一整套
   已经调好接口的量价模型，我们从零写要很久
2. **成熟的因子表达式引擎** —— ``Ref($close, -1)/$close - 1`` 这种写法，
   以及 Alpha158 / Alpha360 两套现成的因子集（158 / 360 个因子）

接进来之后，Qlib 负责「找信号」，本项目负责「按 A 股规则执行」——
T+1、涨跌停、整手、回撤控制这些 Qlib 处理得比较粗。

## ⚠ 导出的是后复权价，factor 恒为 1

本地存的已经是后复权价格，所以 Qlib 里的 ``$factor`` 全是 1。
Qlib 官方数据的约定是「$close 为前复权、$factor 为复权因子」，
用官方示例配置时要注意这个差别 —— 我们的 ``$close`` 直接就是可比价格。

用法::

    # 导出沪深300 历史成分（含指数在册区间，可消除成分股前视）
    python scripts/export_qlib.py --index 000300.SH --start 2016-01-01

    # 导出本地全部标的
    python scripts/export_qlib.py --all
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.datafeed.qlib_export import QlibExporter, read_bin  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402

#: 指数代码 -> Qlib instruments 文件名（沿用 Qlib 官方命名）
INDEX_NAME = {
    "000300.SH": "csi300",
    "000905.SH": "csi500",
    "000852.SH": "csi1000",
    "000016.SH": "sse50",
}


def progress(done: int, total: int, info: str) -> None:
    filled = int(28 * done / total)
    sys.stdout.write(f"\r  [{'#' * filled}{'-' * (28 - filled)}] "
                     f"{done}/{total} {info:<16}")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def main() -> int:
    p = argparse.ArgumentParser(description="导出 Qlib 数据")
    p.add_argument("--index", default="000300.SH")
    p.add_argument("--all", action="store_true",
                   help="导出 data/1d 下的全部标的，而非某个指数的历史成分")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--out", default=None,
                   help="输出目录，默认 data/qlib_data")
    p.add_argument("--no-clean", action="store_true",
                   help="跳过清洗直接导出。脏数据会进训练集，仅排查问题时用")
    args = p.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    store = cfg.data.store_dir
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    out = Path(args.out or Path(store) / "qlib_data")

    # ---- 确定要导出的标的
    weight_csv = Path(store) / "universe" / f"index_weight_{args.index}.csv"
    if args.all:
        symbols = sorted(
            f"{p.stem}.{p.parent.name}"
            for p in (Path(store) / "1d").rglob("*.parquet"))
        print(f"导出本地全部标的：{len(symbols)} 只")
    else:
        if not weight_csv.exists():
            print(f"缺少历史成分 {weight_csv}\n"
                  f"请先运行 scripts/download_index_weight.py --index {args.index}")
            return 1
        w = pd.read_csv(weight_csv, parse_dates=["date"])
        w = w[(w["date"] >= args.start) & (w["date"] <= end)]
        symbols = sorted(w["symbol"].unique())
        print(f"{args.index} 历史成分并集：{len(symbols)} 只")

    exporter = QlibExporter(store, out, clean=not args.no_clean)

    t0 = time.time()
    print("构建全局交易日历（所有标的交易日的并集）...")
    cal = exporter.build_calendar(symbols, args.start, end)
    print(f"  {len(cal)} 个交易日  {cal[0].date()} ~ {cal[-1].date()}")

    print("导出行情...")
    spans = exporter.export_features(symbols, progress=progress)
    print(f"  {len(spans)} 只标的有数据"
          f"（{len(symbols) - len(spans)} 只本地无行情，已跳过）")

    # 清洗结果必须打出来。清了什么无人知晓，等于把「看得见的脏数据」
    # 变成「看不见的改动」
    if exporter.clean_stats:
        st = dict(exporter.clean_stats)
        from_clean = st.pop("来自清洗层", 0)
        fallback = st.pop("回退原始层", 0)
        print("\n数据来源:")
        print(f"  清洗层  {from_clean:>6d} 只")
        if fallback:
            print(f"  原始层  {fallback:>6d} 只  <- 清洗层缺失，已就地清洗")
            print(f"    建议运行 python scripts/clean_data.py 补齐清洗层，"
                  f"否则每次导出都要重清")
        if st:
            print("  就地清洗明细:")
            for rule, n in sorted(st.items(), key=lambda x: -x[1]):
                tag = "" if rule.startswith("[标记]") else "[已修改] "
                print(f"    {tag}{rule}: {n:,} 行")
    elif exporter.clean:
        print("\n数据清洗: 未发现需处理的问题")

    exporter.write_calendar()
    exporter.write_instruments(spans, "all")

    if not args.all and weight_csv.exists():
        name = INDEX_NAME.get(args.index, args.index.replace(".", "_").lower())
        path = exporter.write_index_members(str(weight_csv), name, spans)
        n_lines = len(path.read_text(encoding="utf-8").strip().splitlines())
        print(f"  指数成分 {name}.txt：{n_lines} 段在册区间")
        print("    （同一代码可有多行 —— 调入调出各算一段，"
              "这是消除成分股前视的关键）")

    # ---- 回读校验：导出的值必须与源数据一致
    sample = next(iter(spans))
    from qmtquant.datafeed.qlib_export import to_qlib_code
    got = read_bin(out / "features" / to_qlib_code(sample) / "close.day.bin", cal)
    src = exporter._load_one(sample)["close"]
    src = src[(src.index >= cal[0]) & (src.index <= cal[-1])]
    common = got.index.intersection(src.index)
    diff = (got[common] - src[common]).abs().max()
    print(f"\n回读校验 {sample}：{len(common)} 个交易日，最大偏差 {diff:.6f}")
    if diff > 1e-2:
        print("  ⚠ 偏差过大，导出可能有问题")
        return 1
    print("  ✓ 与源数据一致（float32 精度内）")

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"\n完成，耗时 {(time.time() - t0) / 60:.1f} 分钟，"
          f"{size_mb:.0f} MB")
    print(f"输出目录: {out.resolve()}")
    print("\n在 Qlib 里这样用：")
    print("    import qlib")
    print(f'    qlib.init(provider_uri=r"{out.resolve()}", region="cn")')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
