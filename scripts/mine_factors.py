"""逐个评估 Alpha158 因子在中证1000 上的 IC，找出最强因子。

对每个因子：
1. 计算日频横截面 Spearman IC（因子值 vs 次日收益率）
2. 前半段 / 后半段分别算 IC，看稳定性
3. 按 |t| 降序排列，标记两段都显著的

用法::

    python scripts/mine_factors.py
    python scripts/mine_factors.py --market csi300 --top 30
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

PERIOD = ("2016-01-01", "2026-08-27")


def get_all_factors():
    """提取 Alpha158 的全部因子名和表达式"""
    from qlib.contrib.data.handler import Alpha158DL

    conf = {
        "kbar": {},
        "price": {
            "windows": [0],
            "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
        },
        "rolling": {},
    }
    fields, names = Alpha158DL.get_feature_config(conf)
    return list(zip(names, fields))


def compute_ic(factor_values, returns):
    """横截面 Spearman IC 序列"""
    import numpy as np
    import pandas as pd

    df = pd.concat([factor_values.rename("factor"),
                     returns.rename("ret")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    grouped = df.groupby(level="datetime")
    ic_dict = {}
    for dt, g in grouped:
        if len(g) >= 30:
            v = g["factor"].corr(g["ret"], method="spearman")
            if not np.isnan(v):
                ic_dict[dt] = v
    return pd.Series(ic_dict).sort_index()


def ic_stats(ic_series):
    """IC 统计"""
    import numpy as np
    if len(ic_series) < 10:
        return {"mean": 0, "std": 0, "ir": 0, "t": 0, "pos_pct": 0, "n": 0}
    mean = float(ic_series.mean())
    std = float(ic_series.std(ddof=1))
    ir = mean / std if std > 0 else 0
    t = ir * np.sqrt(len(ic_series))
    return {
        "mean": mean,
        "std": std,
        "ir": ir,
        "t": t,
        "pos_pct": float((ic_series > 0).mean()),
        "n": len(ic_series),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Alpha158 因子挖掘")
    p.add_argument("--uri", default=None)
    p.add_argument("--market", default="csi1000")
    p.add_argument("--top", type=int, default=50, help="显示前 N 个因子")
    p.add_argument("--out", default="reports/factor_mining")
    args = p.parse_args()

    import numpy as np
    import pandas as pd
    import qlib
    from qlib.data import D

    from qmtquant.config import get_config

    cfg = get_config()
    uri = args.uri or str(Path(cfg.data.store_dir) / "qlib_data")
    qlib.init(provider_uri=uri, region="cn", joblib_backend="threading")

    print("=" * 70)
    print(f"Alpha158 因子挖掘 — {args.market}")
    print("=" * 70)

    factors = get_all_factors()
    print(f"因子数量: {len(factors)}")
    print(f"评估区间: {PERIOD[0]} ~ {PERIOD[1]}")

    # 获取次日收益率作为标签
    print("\n计算收益率标签...")
    instruments = D.instruments(args.market)
    label_df = D.features(
        instruments, ["Ref($close,-1)/$close-1"],
        start_time=PERIOD[0], end_time=PERIOD[1],
    )
    label = label_df.iloc[:, 0]
    print(f"  标签: {len(label):,} 条")

    # 确定前后半段分界点
    dates = label.index.get_level_values("datetime").unique().sort_values()
    mid = dates[len(dates) // 2]
    mid_str = str(mid.date())
    print(f"  前半段: {dates[0].date()} ~ {mid.date()}")
    print(f"  后半段: {mid.date()} ~ {dates[-1].date()}")

    # 逐个因子评估
    print(f"\n评估 {len(factors)} 个因子...")
    results = []
    t0 = time.time()

    for i, (name, expr) in enumerate(factors):
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(factors) - i - 1) if i > 0 else 0
            sys.stdout.write(
                f"\r  [{i+1}/{len(factors)}] {name:<16} "
                f"ETA {eta/60:.1f}min")
            sys.stdout.flush()

        try:
            feat_df = D.features(
                instruments, [expr],
                start_time=PERIOD[0], end_time=PERIOD[1],
            )
            feat = feat_df.iloc[:, 0]
        except Exception as e:
            results.append({
                "name": name, "expr": expr,
                "ic_mean": 0, "t": 0, "t_h1": 0, "t_h2": 0,
                "error": str(e),
            })
            continue

        ic_all = compute_ic(feat, label)
        s_all = ic_stats(ic_all)

        # 前后半段 — 按位置切分，避免 index 类型问题
        n_half = len(ic_all) // 2
        ic_h1 = ic_all.iloc[:n_half]
        ic_h2 = ic_all.iloc[n_half:]
        s_h1 = ic_stats(ic_h1)
        s_h2 = ic_stats(ic_h2)

        results.append({
            "name": name,
            "expr": expr[:60],
            "ic_mean": s_all["mean"],
            "ic_std": s_all["std"],
            "icir": s_all["ir"],
            "t": s_all["t"],
            "ic_pos": s_all["pos_pct"],
            "t_h1": s_h1["t"],
            "t_h2": s_h2["t"],
            "n": s_all["n"],
        })

    print(f"\r  完成，耗时 {(time.time() - t0) / 60:.1f} 分钟" + " " * 30)

    df = pd.DataFrame(results)
    df["abs_t"] = df["t"].abs()
    df["both_sig"] = (df["t_h1"].abs() >= 2) & (df["t_h2"].abs() >= 2)
    df["same_sign"] = np.sign(df["t_h1"]) == np.sign(df["t_h2"])
    df["stable"] = df["both_sig"] & df["same_sign"]
    df = df.sort_values("abs_t", ascending=False)

    # 输出
    print("\n" + "=" * 70)
    print(f"Top {args.top} 因子（按 |t| 排序）")
    print("=" * 70)
    print(f"{'#':>3} {'因子':<16} {'IC均值':>8} {'ICIR':>7} {'t值':>7} "
          f"{'t前半':>7} {'t后半':>7} {'稳定':>4}")
    print("-" * 70)

    for rank, (_, row) in enumerate(df.head(args.top).iterrows(), 1):
        stable_mark = "Y" if row["stable"] else ""
        print(f"{rank:3d} {row['name']:<16} {row['ic_mean']:>8.4f} "
              f"{row.get('icir', 0):>7.3f} {row['t']:>7.2f} "
              f"{row['t_h1']:>7.2f} {row['t_h2']:>7.2f} "
              f"{'  '+stable_mark:>4}")

    n_stable = df["stable"].sum()
    print(f"\n两段都显著且同向: {n_stable} / {len(df)} 个因子")

    if n_stable > 0:
        print("\n稳定因子列表:")
        stable = df[df["stable"]].copy()
        for _, row in stable.iterrows():
            print(f"  {row['name']:<16} IC={row['ic_mean']:+.4f}  "
                  f"t={row['t']:.2f}  表达式: {row['expr']}")

    # 保存
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "factor_ic_all.csv", index=False, encoding="utf-8-sig")
    if n_stable > 0:
        stable.to_csv(out / "stable_factors.csv", index=False,
                      encoding="utf-8-sig")
    print(f"\n明细已保存: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
