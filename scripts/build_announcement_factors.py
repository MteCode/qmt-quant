"""把公告转成截面因子。

## 事件型数据怎么变成因子

公告是**稀疏事件**：某只股票某天有公告，大部分股票大部分时候没有。
直接当因子用不行 —— 截面上绝大多数是缺失，排名没有意义。

做法是**滚动累计**：过去 N 个交易日内该类公告出现几次。
这样每只股票每天都有值（多数为 0），可以做截面排序。
与 dragon_count_20 是同一思路。

## 可获得时点

公告的 `date` 是**披露日**。交易所要求在开市前或收市后披露，
盘中披露的属少数。保守起见统一滞后 1 个交易日 ——
即当日公告最早次日才可交易。

不滞后就是前视：晚上 8 点披露的减持公告，当天收盘价里不可能反映它。

## 产出的因子

    ann_reduce_20      20 日减持公告次数（利空）
    ann_risk_20        20 日风险类公告次数（诉讼/处罚/问询/质押）
    ann_buyback_20     20 日回购公告次数（利好）
    ann_dilution_20    20 日增发公告次数（利空）
    ann_unlock_20      20 日限售解禁次数（利空）
    ann_attention_20   20 日调研活动次数（关注度，方向未定）
    ann_net_score_20   20 日公告净分（利好数 - 利空数）

方向标注仅供参考 —— **实际方向由 IC 决定**。评估框架会告诉我们
哪些因子真的有预测力、哪些只是常识错觉。

用法::

    python scripts/build_announcement_factors.py
    python scripts/build_announcement_factors.py --window 60
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC_DIR = ROOT / "data" / "announcement"

#: 要建成因子的类别。方向仅作命名参考，实际由 IC 判定
CATEGORIES = ["reduce", "risk", "buyback", "dilution",
              "unlock", "attention", "incentive", "guarantee"]

#: 公告可获得滞后（交易日）。披露日当天不可交易，见模块说明
LAG = 1


def load_all():
    """读全部公告，只保留已归类的。"""
    import pandas as pd

    files = sorted(SRC_DIR.glob("*.parquet"))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except (OSError, ValueError):
            continue
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    return df[df["date"].notna()]


def build(df, calendar, window: int) -> dict:
    """按类别构建滚动计数因子。"""
    import pandas as pd

    out = {}
    cal = pd.DatetimeIndex(sorted(calendar))

    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        # 同一标的同一天可能有多条同类公告，计数而非去重 ——
        # 一天发三份减持公告比发一份信息量更大
        daily = (sub.groupby(["date", "vt_symbol"]).size()
                 .unstack(fill_value=0))
        # 对齐到完整交易日历，缺失日补 0（那天确实没公告）
        daily = daily.reindex(cal, fill_value=0)
        rolled = daily.rolling(window, min_periods=1).sum()
        # 滞后：披露日当天不可交易
        out[f"ann_{cat}_{window}"] = rolled.shift(LAG)

    # 净分：利好减利空
    pos = df[df["direction"] > 0]
    neg = df[df["direction"] < 0]
    if not pos.empty or not neg.empty:
        def counts(x):
            if x.empty:
                return None
            return (x.groupby(["date", "vt_symbol"]).size()
                    .unstack(fill_value=0)
                    .reindex(cal, fill_value=0)
                    .rolling(window, min_periods=1).sum())
        p, n = counts(pos), counts(neg)
        if p is not None and n is not None:
            cols = p.columns.union(n.columns)
            net = (p.reindex(columns=cols, fill_value=0)
                   - n.reindex(columns=cols, fill_value=0))
            out[f"ann_net_score_{window}"] = net.shift(LAG)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="构建公告因子")
    p.add_argument("--window", type=int, default=20,
                    help="滚动窗口（交易日），与调仓周期对齐")
    args = p.parse_args()

    import pandas as pd

    from qmtquant.config import get_config
    from qmtquant.datafeed.qlib_export import to_qlib_code

    cfg = get_config()
    store = Path(cfg.data.store_dir)

    print("=" * 58)
    print("构建公告因子")
    print(f"  窗口 {args.window} 交易日   可获得滞后 {LAG} 日")
    print("=" * 58)

    df = load_all()
    if df is None or df.empty:
        print("无公告数据，请先运行 scripts/download_announcements.py")
        return 1
    n_cat = (df["category"] != "neutral").sum()
    print(f"\n公告 {len(df):,} 条，已归类 {n_cat:,} 条"
          f"（{df['date'].min().date()} ~ {df['date'].max().date()}）")

    cal_file = store / "qlib_data" / "calendars" / "day.txt"
    if not cal_file.exists():
        print("缺少交易日历，请先运行 scripts/export_qlib.py")
        return 1
    cal = pd.to_datetime(cal_file.read_text(encoding="utf-8").split())
    cal = cal[(cal >= df["date"].min()) & (cal <= df["date"].max())]
    print(f"交易日历 {len(cal)} 天")

    factors = build(df, cal, args.window)
    if not factors:
        print("没有可用类别")
        return 1

    out_dir = store / "qlib_data" / "money_factors"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'因子':<24s}{'覆盖标的':>10s}{'日均非零':>10s}")
    print("-" * 46)
    for name, panel in factors.items():
        # 存成与现有因子一致的 (datetime, instrument) 长格式
        ren = {}
        for c in panel.columns:
            try:
                ren[c] = to_qlib_code(str(c))
            except ValueError:
                pass
        panel = panel.rename(columns=ren)
        panel = panel[[c for c in panel.columns if c in set(ren.values())]]

        s = panel.stack()
        s.index.names = ["datetime", "instrument"]
        s.name = name
        s.to_frame().to_parquet(out_dir / f"{name}.parquet")

        nz = (panel > 0).sum(axis=1).mean()
        print(f"{name:<24s}{panel.shape[1]:>10d}{nz:>10.0f}")

    print(f"\n已写入 {out_dir}")
    print("\n下一步：评估这些因子是否真的有预测力")
    print("  python strategies/alstm_ppo_csi1000/eval_factors.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
