"""候选因子评估 —— 决定哪些值得进多因子模型。

## 为什么先评估、不直接建模

ALSTM 用的 Alpha360 是纯价量特征，8 种子实验已经证明它的信号弱到无法
支撑实盘。加因子的意义在于**拓宽信息面**，而不是把同一批信息换个模型再挖。

因此一个候选因子要过三关，缺一不可：

1. **有预测力** —— IC 显著（|t| >= 2）。没有预测力的因子加进去只是噪声。
2. **跨期稳定** —— 分段后 IC 符号不翻转。翻转说明是样本内挖出来的，
   换个时段就失效，与之前参数扫描踩的坑同源。
3. **与现有信号正交** —— 与 ALSTM 分数的相关性低。相关性高意味着
   讲的是同一件事，模型已经知道了，加进来不会带来新信息。

第三关最容易被忽略，却是多因子成败的关键：五个高度相关的因子，
其信息量约等于一个因子。

## 覆盖率也是硬门槛

截面因子要求同一天有足够多的标的有值，否则排名失去意义。
实测 north_net_pct 日均只有 19 只有值、dragon_inst_net 只有 58 只 ——
这类因子无法参与截面排序，评估阶段就该筛掉。

用法::

    python strategies/alstm_ppo_csi1000/eval_factors.py
    python strategies/alstm_ppo_csi1000/eval_factors.py --min-coverage 300
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import paths  # noqa: E402
from train_alstm import TEST  # noqa: E402

RESULT_JSON = paths.BACKTEST_DIR / "factor_eval.json"

#: 截面排序所需的最少标的数。低于这个数，当天的排名不具代表性
MIN_COVERAGE = 200

#: 预测周期（交易日）。与策略 20 日调仓对齐
FWD_DAYS = 20


def load_money_factors(store_dir: str) -> dict:
    """加载已建好的资金流因子。"""
    import pandas as pd

    out = {}
    d = Path(store_dir) / "qlib_data" / "money_factors"
    if not d.exists():
        return out
    for p in sorted(d.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except (OSError, ValueError):
            continue
        s = df.iloc[:, 0]
        # 因子面板是 (datetime, instrument)，转成 日期 x 标的 的宽表
        try:
            out[p.stem] = s.unstack(level=1).sort_index()
        except (ValueError, KeyError):
            continue
    return out


def forward_returns(market: str, start: str, end: str, days: int):
    """未来 N 日收益率，作为 IC 的比较基准。"""
    import pandas as pd
    from qlib.data import D

    inst = D.instruments(market)
    px = D.features(inst, ["$close"], start_time=start, end_time=end)
    px.columns = ["close"]
    wide = px["close"].unstack(level=0).sort_index()
    if wide.index.name != "datetime":
        wide = px["close"].unstack(level="instrument").sort_index()
    return wide.shift(-days) / wide - 1


def newey_west_t(series, lag: int) -> float:
    """自相关稳健的 t 值（Newey-West）。

    IC 序列是**每日**用 20 日前瞻收益算出来的，相邻两天共享 19 天的收益数据，
    重叠极重。直接用 ``mean/std*sqrt(n)`` 会把有效样本量当成 n，
    而实际独立观测约只有 n/20 —— t 值因此被放大约 sqrt(20) ≈ 4.5 倍。

    不修正的话会得出「t = -25，极其显著」这种误导性结论。
    Newey-West 通过纳入前 lag 阶自相关来还原真实的标准误。
    """
    import numpy as np

    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 3:
        return 0.0
    mu = x.mean()
    e = x - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        # Bartlett 核：高阶自相关权重递减，保证方差估计非负
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if var <= 0:
        return 0.0
    return float(mu / np.sqrt(var / n))


def evaluate(name: str, panel, fwd, min_cov: int, split_date: str,
             fwd_days: int = FWD_DAYS) -> dict:
    """单因子评估：IC、显著性、分段稳定性、覆盖率。"""
    import numpy as np
    import pandas as pd

    # 对齐到共同的日期与标的
    dates = panel.index.intersection(fwd.index)
    cols = panel.columns.intersection(fwd.columns)
    if len(dates) < 60 or len(cols) < min_cov:
        return {"name": name, "usable": False,
                "reason": f"对齐后仅 {len(dates)} 期 x {len(cols)} 只"}

    f = panel.loc[dates, cols]
    r = fwd.loc[dates, cols]

    ics, cov = [], []
    for d in dates:
        fr, rr = f.loc[d], r.loc[d]
        m = fr.notna() & rr.notna()
        n = int(m.sum())
        cov.append(n)
        if n < min_cov:
            continue
        ic = fr[m].corr(rr[m], method="spearman")
        if pd.notna(ic):
            ics.append((d, ic))

    mean_cov = float(np.mean(cov)) if cov else 0.0
    if len(ics) < 30:
        return {"name": name, "usable": False,
                "mean_coverage": round(mean_cov, 1),
                "reason": f"有效截面仅 {len(ics)} 期"
                          f"（日均 {mean_cov:.0f} 只有值，需 >= {min_cov}）"}

    idx = pd.DatetimeIndex([d for d, _ in ics])
    vals = pd.Series([v for _, v in ics], index=idx)
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    t_raw = mean / sd * np.sqrt(len(vals)) if sd else 0.0
    t = newey_west_t(vals, lag=fwd_days - 1)

    # 分段：同一因子在前后两段是否讲同一个故事
    h1 = vals[vals.index < split_date]
    h2 = vals[vals.index >= split_date]
    ic1 = float(h1.mean()) if len(h1) >= 15 else None
    ic2 = float(h2.mean()) if len(h2) >= 15 else None
    flip = bool(ic1 is not None and ic2 is not None
                and (ic1 > 0) != (ic2 > 0))

    return {
        "name": name, "usable": True,
        "ic_mean": round(mean, 5),
        "ic_std": round(sd, 5),
        "icir": round(mean / sd, 4) if sd else 0.0,
        "t": round(float(t), 3),              # Newey-West 修正后
        "t_raw": round(float(t_raw), 3),      # 未修正，仅供对照
        # numpy 的 bool_/float64 不能直接 json 序列化，统一转成 Python 原生类型
        "significant": bool(abs(t) >= 2),
        "ic_positive_ratio": round(float((vals > 0).mean()), 4),
        "n_periods": int(len(vals)),
        "mean_coverage": round(mean_cov, 1),
        "ic_h1": round(ic1, 5) if ic1 is not None else None,
        "ic_h2": round(ic2, 5) if ic2 is not None else None,
        "sign_flip": flip,
    }


def correlation_with_alstm(panels: dict, alstm) -> dict:
    """各因子与 ALSTM 分数的截面相关性。

    这是多因子成败的关键：与现有信号高度相关的因子讲的是同一件事，
    加进来不带来新信息。五个高度相关的因子，信息量约等于一个。
    """
    import numpy as np
    import pandas as pd

    out = {}
    for name, panel in panels.items():
        dates = panel.index.intersection(alstm.index)
        cols = panel.columns.intersection(alstm.columns)
        if len(dates) < 30 or len(cols) < 50:
            out[name] = None
            continue
        cs = []
        for d in dates[::5]:  # 每 5 天取一次，够用且快得多
            a, b = panel.loc[d, cols], alstm.loc[d, cols]
            m = a.notna() & b.notna()
            if m.sum() < 50:
                continue
            c = a[m].corr(b[m], method="spearman")
            if pd.notna(c):
                cs.append(c)
        out[name] = round(float(np.mean(cs)), 4) if cs else None
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="候选因子评估")
    p.add_argument("--market", default="csi1000")
    p.add_argument("--min-coverage", type=int, default=MIN_COVERAGE)
    p.add_argument("--fwd-days", type=int, default=FWD_DAYS)
    p.add_argument("--split", default="2024-03-01",
                    help="分段检验的切分日")
    args = p.parse_args()

    import pandas as pd

    from qmtquant.config import LOG_DIR, get_config
    from qmtquant.datafeed.qlib_init import init_qlib
    from qmtquant.datafeed.qlib_export import to_qlib_code
    from qmtquant.utils.logger import setup_logging

    cfg = get_config()
    setup_logging(LOG_DIR, cfg.log_level)
    uri = str(Path(cfg.data.store_dir) / "qlib_data")
    init_qlib(uri, n_expressions=8)

    print("=" * 72)
    print("候选因子评估")
    print("=" * 72)
    print(f"区间      : {TEST[0]} ~ {TEST[1]}（样本外）")
    print(f"预测周期  : {args.fwd_days} 交易日（与调仓周期对齐）")
    print(f"最少覆盖  : {args.min_coverage} 只/日")
    print(f"分段切分  : {args.split}")

    print("\n加载候选因子...")
    panels = load_money_factors(cfg.data.store_dir)
    if not panels:
        print("未找到因子。请先运行 scripts/build_money_factors.py")
        return 1
    for k, v in panels.items():
        print(f"  {k:<22s} {v.shape[0]} 期 x {v.shape[1]} 只")

    print("\n计算未来收益...")
    fwd = forward_returns(args.market, TEST[0], TEST[1], args.fwd_days)
    print(f"  {fwd.shape[0]} 期 x {fwd.shape[1]} 只")

    print("\n" + "-" * 72)
    print(f"{'因子':<22s}{'IC':>9s}{'t(NW)':>8s}{'t(未修正)':>11s}"
          f"{'前段':>9s}{'后段':>9s}{'覆盖':>7s}")
    print("-" * 72)

    rows = []
    for name, panel in panels.items():
        r = evaluate(name, panel, fwd, args.min_coverage, args.split,
                     args.fwd_days)
        rows.append(r)
        if not r.get("usable"):
            print(f"{name:<22s}  {'不可用':<10s} {r.get('reason', '')}")
            continue
        flag = " !" if r["sign_flip"] else ""
        print(f"{name:<22s}{r['ic_mean']:>+9.4f}{r['t']:>8.2f}"
              f"{r['t_raw']:>11.2f}"
              f"{(r['ic_h1'] if r['ic_h1'] is not None else float('nan')):>+9.4f}"
              f"{(r['ic_h2'] if r['ic_h2'] is not None else float('nan')):>+9.4f}"
              f"{r['mean_coverage']:>7.0f}{flag}")

    # ---- 与 ALSTM 的正交性
    corr = {}
    if paths.ALSTM_SCORES.exists():
        print("\n计算与 ALSTM 信号的相关性...")
        a = pd.read_parquet(paths.ALSTM_SCORES).sort_index()
        # ALSTM 分数列是 vt 格式，因子面板是 qlib 格式，统一到 qlib
        ren = {}
        for c in a.columns:
            try:
                ren[c] = to_qlib_code(str(c))
            except ValueError:
                pass
        a = a.rename(columns=ren)
        corr = correlation_with_alstm(
            {k: v for k, v in panels.items()
             if next((r for r in rows if r["name"] == k), {}).get("usable")},
            a)
        for k, v in corr.items():
            print(f"  {k:<22s} {v:+.4f}" if v is not None
                  else f"  {k:<22s} 无法计算")

    # ---- 结论
    print("\n" + "-" * 72)
    print("结论")
    print("-" * 72)
    keep, drop = [], []
    for r in rows:
        if not r.get("usable"):
            drop.append((r["name"], r.get("reason", "不可用")))
            continue
        c = corr.get(r["name"])
        if not r["significant"]:
            drop.append((r["name"], f"IC 不显著（|t|={abs(r['t']):.2f} < 2）"))
        elif r["sign_flip"]:
            drop.append((r["name"], "分段 IC 符号翻转，不具跨期稳定性"))
        elif c is not None and abs(c) > 0.5:
            drop.append((r["name"], f"与 ALSTM 相关性 {c:+.2f} 过高，信息重复"))
        else:
            keep.append((r["name"], r, c))

    if keep:
        print(f"\n  可用于多因子（{len(keep)} 个）:")
        for name, r, c in sorted(keep, key=lambda x: -abs(x[1]["t"])):
            cs = f"与ALSTM相关 {c:+.2f}" if c is not None else "相关性未知"
            print(f"    {name:<22s} IC {r['ic_mean']:+.4f} "
                  f"(t={r['t']:+.2f})  {cs}")
    else:
        print("\n  没有因子通过全部三关。")

    if drop:
        print(f"\n  剔除（{len(drop)} 个）:")
        for name, why in drop:
            print(f"    {name:<22s} {why}")

    paths.ensure_dirs()
    RESULT_JSON.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {"market": args.market, "fwd_days": args.fwd_days,
                   "min_coverage": args.min_coverage, "split": args.split,
                   "period": list(TEST)},
        "factors": rows,
        "corr_with_alstm": corr,
        "keep": [k for k, _, _ in keep],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细: {RESULT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
