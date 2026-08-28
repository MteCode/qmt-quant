"""估值因子 IC 分析。

在写策略之前先回答一个问题：**这些因子在沪深300 上到底有没有预测力。**

一次 IC 分析几秒钟，一次完整回测几分钟，而且回测结果掺杂了仓位、
成本、调仓频率等一堆与因子本身无关的因素 —— 因子没有预测力的话，
回测出来的任何盈亏都只是噪声的形状。

## 标的池

用 Tushare 拉到的**历史成分股**，逐日取当期真实名单 ——
不这样做的话，IC 会被「后来才被纳入指数的股票」污染。

## 因子方向

PE/PB/PS 是「越小越便宜」，所以取**倒数**（EP/BP/SP）后再算 IC。
这样所有因子都统一成「越大越好」，IC 为正即符合价值投资直觉。
不取倒数也能算，但负 PE（亏损股）会让排序失去意义。

用法::

    python scripts/analyze_factors.py
    python scripts/analyze_factors.py --index 000905.SH --start 2016-01-01
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from qmtquant.config import LOG_DIR, get_config  # noqa: E402
from qmtquant.research.factor_ic import analyze, compare_table  # noqa: E402
from qmtquant.utils.logger import setup_logging  # noqa: E402

#: 待检验的因子。值为 (daily_basic 列名, 是否取倒数, 中文名)
#:
#: 取倒数的理由：PE=10 比 PE=20 便宜，但 PE=-5（亏损）不是"最便宜"。
#: 取倒数后 EP=0.1 > EP=0.05 > EP=-0.2，排序才有经济含义
FACTORS = {
    # 价格派生因子用 __price__ 标记，由 close 矩阵现算，不来自 daily_basic
    "低波动": ("__vol20__", False, "20 日已实现波动率的负值"),
    "低波动60": ("__vol60__", False, "60 日已实现波动率的负值"),
    "EP": ("pe_ttm", True, "盈利收益率（PE_TTM 倒数）"),
    "BP": ("pb", True, "账面市值比（PB 倒数）"),
    "SP": ("ps_ttm", True, "销售收益率（PS_TTM 倒数）"),
    "股息率": ("dv_ttm", False, "股息率 TTM"),
    "换手率": ("turnover_rate_f", False, "自由流通换手率"),
    "对数市值": ("total_mv", False, "总市值（取对数）"),
}

#: 预测周期（交易日）。多周期一起看是为了识别「只在某个特定周期有效」
#: 的假信号 —— 真实因子通常在相邻周期上表现连续
HORIZONS = [5, 20, 60]


def load_factor_panel(cfg, start: str, end: str) -> pd.DataFrame:
    """读取 daily_basic 分年 parquet，拼成长表"""
    root = Path(cfg.data.store_dir) / "factor" / "daily_basic"
    if not root.exists():
        raise SystemExit(
            f"缺少估值因子数据: {root}\n"
            "请先运行: python scripts/download_daily_basic.py --start 2016-01-01")

    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    frames = []
    for y in years:
        p = root / f"{y}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit(f"{start} ~ {end} 区间内没有因子数据")

    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]


def load_universe_mask(cfg, index_code: str, dates) -> pd.DataFrame:
    """逐日的成分股掩码：True 表示该日该股票在指数内。

    这是消除成分股前视的关键 —— 不加这层过滤，IC 会被
    「后来才被纳入指数」的股票污染，而它们能进指数恰恰因为之前涨得好。
    """
    path = (Path(cfg.data.store_dir) / "universe"
            / f"index_weight_{index_code}.csv")
    if not path.exists():
        raise SystemExit(
            f"缺少历史成分股: {path}\n"
            f"请先运行: python scripts/download_index_weight.py "
            f"--index {index_code}")

    w = pd.read_csv(path, parse_dates=["date"])
    periods = sorted(w["date"].unique())
    members = {d: set(g["symbol"]) for d, g in w.groupby("date")}

    # asof：每个交易日取「不晚于它的最近一期名单」，绝不用未来的调整结果
    idx = np.searchsorted(periods, dates, side="right") - 1
    rows = []
    for d, i in zip(dates, idx):
        rows.append(members[periods[i]] if i >= 0 else set())
    return rows


def build_matrix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """长表转 日期 × 标的 的宽表"""
    return df.pivot_table(index="trade_date", columns="symbol",
                          values=col, aggfunc="last")


def main() -> int:
    p = argparse.ArgumentParser(description="估值因子 IC 分析")
    p.add_argument("--index", default="000300.SH")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--report", default="reports")
    args = p.parse_args()

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"读取估值因子 {args.start} ~ {end} ...")
    df = load_factor_panel(cfg, args.start, end)
    print(f"  {len(df):,} 行，{df['symbol'].nunique():,} 只标的，"
          f"{df['trade_date'].nunique():,} 个交易日")

    dates = sorted(df["trade_date"].unique())
    print(f"读取 {args.index} 历史成分股 ...")
    member_sets = load_universe_mask(cfg, args.index, dates)
    sizes = [len(s) for s in member_sets]
    print(f"  每日成分数 中位数 {int(np.median(sizes))}")

    # 只保留当日在指数内的记录
    date_to_members = dict(zip(dates, member_sets))
    df = df[[s in date_to_members[d]
             for d, s in zip(df["trade_date"], df["symbol"])]]
    print(f"  过滤后 {len(df):,} 行")

    prices = build_matrix(df, "close")

    all_reports = []
    for name, (col, invert, desc) in FACTORS.items():
        if col.startswith("__vol"):
            # 已实现波动率 = 日收益率的滚动标准差。**取负值**使「越大越好」，
            # 与其他因子方向统一 —— IC 为正即支持「低波动股后续表现更好」
            window = int(col.strip("_").replace("vol", ""))
            m = -prices.pct_change().rolling(window).std()
        elif col not in df.columns:
            print(f"\n[!] 缺少列 {col}，跳过 {name}")
            continue
        else:
            m = build_matrix(df, col)

        if col.startswith("__vol"):
            pass    # 已取负，不再变换
        elif invert:
            # 取倒数前先剔除 <=0：负 PE 是亏损股，倒数后会排到最前面，
            # 变成「专挑亏损最惨的买」，与因子本意完全相反
            m = m.where(m > 0)
            m = 1.0 / m
        elif name == "对数市值":
            m = np.log(m.where(m > 0))

        reports = analyze(m, prices, name, HORIZONS)
        all_reports += reports
        print()
        print(f"########## {name} · {desc} ##########")
        for r in reports:
            print(r.summary())
            print()

    if not all_reports:
        return 1

    table = compare_table(all_reports)
    print("\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    usable = table[table["结论"].str.contains("有效")]
    print(f"\n可用因子：{len(usable)}/{len(table)} 个组合")
    if usable.empty:
        print("⚠ 没有任何因子在该标的池上显著有效 —— "
              "在此基础上写选股策略，回测出来的盈亏只是噪声的形状。")

    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"factor_ic_{args.index}.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n完整结果: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
