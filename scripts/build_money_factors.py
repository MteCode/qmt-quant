"""把大资金原始数据转化为个股日频因子，导出为 Qlib 可读格式。

因子列表：
1. north_net_pct   — 北向十大活跃股净买入占成交比（事件型，多数日为 NaN）
2. dragon_net      — 龙虎榜机构净买入金额（事件型）
3. dragon_count_20 — 过去 20 日上龙虎榜次数（累计成连续信号）
4. margin_bal_chg  — 融资余额日变化率（杠杆资金方向）
5. margin_ratio    — 融资余额 / 成交额（杠杆资金浓度）

用法::

    python scripts/build_money_factors.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from qmtquant.config import get_config


def build_northbound_factor(data_dir: Path) -> pd.DataFrame:
    """北向资金因子 — 基于十大活跃股"""
    path = data_dir / "northbound_top10.parquet"
    if not path.exists():
        print("  跳过：northbound_top10.parquet 不存在")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if "ts_code" in df.columns and "symbol" not in df.columns:
        from qmtquant.utils.symbol import from_xt_symbol
        df["symbol"] = df["ts_code"].map(from_xt_symbol)

    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # net_amount: 净买入金额（万元），amount: 成交金额（万元）
    if "net_amount" in df.columns and "amount" in df.columns:
        df["north_net_pct"] = df["net_amount"] / (df["amount"].abs() + 1e-8)
    elif "net_amount" in df.columns:
        df["north_net_pct"] = df["net_amount"]
    else:
        print("  警告：northbound_top10 缺少 net_amount 字段")
        return pd.DataFrame()

    result = df.pivot_table(
        index="trade_date", columns="symbol",
        values="north_net_pct", aggfunc="sum"
    )
    print(f"  北向因子: {result.shape[0]} 日 x {result.shape[1]} 只")
    return result


def build_dragon_factors(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """龙虎榜因子"""
    # 机构净买入
    inst_path = data_dir / "dragon_tiger_inst.parquet"
    list_path = data_dir / "dragon_tiger_list.parquet"

    dragon_net = pd.DataFrame()
    dragon_count = pd.DataFrame()

    if inst_path.exists():
        df = pd.read_parquet(inst_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if "ts_code" in df.columns and "symbol" not in df.columns:
            from qmtquant.utils.symbol import from_xt_symbol
            df["symbol"] = df["ts_code"].map(from_xt_symbol)

        if "buy" in df.columns and "sell" in df.columns:
            df["inst_net"] = df["buy"].fillna(0) - df["sell"].fillna(0)
        elif "net_buy" in df.columns:
            df["inst_net"] = df["net_buy"]
        else:
            # 看看有啥列
            print(f"  龙虎榜机构列: {list(df.columns)}")
            df["inst_net"] = 0

        dragon_net = df.pivot_table(
            index="trade_date", columns="symbol",
            values="inst_net", aggfunc="sum"
        )
        print(f"  龙虎榜机构净买入: {dragon_net.shape[0]} 日 x {dragon_net.shape[1]} 只")

    if list_path.exists():
        df = pd.read_parquet(list_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if "ts_code" in df.columns and "symbol" not in df.columns:
            from qmtquant.utils.symbol import from_xt_symbol
            df["symbol"] = df["ts_code"].map(from_xt_symbol)

        # 每只股票每天上榜 = 1，过去 20 日累计
        appear = df.groupby(["trade_date", "symbol"]).size().unstack(fill_value=0)
        appear = appear.clip(upper=1)  # 同天多次上榜算1次
        dragon_count = appear.rolling(20, min_periods=1).sum()
        print(f"  龙虎榜20日累计上榜: {dragon_count.shape[0]} 日 x {dragon_count.shape[1]} 只")

    return dragon_net, dragon_count


def build_margin_factors(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """融资融券因子"""
    path = data_dir / "margin_detail.parquet"
    if not path.exists():
        print("  跳过：margin_detail.parquet 不存在")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "ts_code" in df.columns and "symbol" not in df.columns:
        from qmtquant.utils.symbol import from_xt_symbol
        df["symbol"] = df["ts_code"].map(from_xt_symbol)

    # rzye: 融资余额
    if "rzye" not in df.columns:
        print(f"  融资融券列: {list(df.columns)}")
        return pd.DataFrame(), pd.DataFrame()

    bal = df.pivot_table(
        index="trade_date", columns="symbol",
        values="rzye", aggfunc="last"
    ).sort_index()

    # 融资余额变化率
    margin_chg = bal.pct_change()
    print(f"  融资余额变化率: {margin_chg.shape[0]} 日 x {margin_chg.shape[1]} 只")

    # 融资余额 / 成交额（如果有 rzmre 融资买入额）
    margin_ratio = pd.DataFrame()
    if "rzmre" in df.columns:
        buy_amt = df.pivot_table(
            index="trade_date", columns="symbol",
            values="rzmre", aggfunc="sum"
        ).sort_index()
        margin_ratio = buy_amt / (bal + 1e-8)
        print(f"  融资买入/余额: {margin_ratio.shape[0]} 日 x {margin_ratio.shape[1]} 只")

    return margin_chg, margin_ratio


def export_to_qlib(factors: dict[str, pd.DataFrame], qlib_dir: Path):
    """把因子面板导出为 Qlib 能读的 csv/pkl"""
    out_dir = qlib_dir / "money_factors"
    out_dir.mkdir(parents=True, exist_ok=True)

    from qmtquant.datafeed.qlib_export import to_qlib_code

    for name, df in factors.items():
        if df.empty:
            continue
        # 转为 MultiIndex (datetime, instrument) -> value 格式
        stacked = df.stack()
        stacked.index.names = ["datetime", "instrument"]
        stacked.name = name

        # vt_symbol -> qlib code
        new_idx = stacked.index.set_levels(
            stacked.index.levels[1].map(to_qlib_code), level="instrument"
        )
        stacked.index = new_idx

        path = out_dir / f"{name}.parquet"
        stacked.to_frame().to_parquet(path)
        print(f"  -> {path} ({len(stacked):,} 条)")

    print(f"\n因子已保存到 {out_dir.resolve()}")


def compute_factor_ic(factors: dict[str, pd.DataFrame],
                      qlib_dir: Path) -> None:
    """对每个大资金因子计算 IC"""
    import qlib
    from qlib.data import D
    import os
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    qlib.init(provider_uri=str(qlib_dir), region="cn",
              joblib_backend="threading")

    instruments = D.instruments("csi1000")
    label_df = D.features(
        instruments, ["Ref($close,-1)/$close-1"],
        start_time="2016-01-01", end_time="2026-08-27",
    )
    label = label_df.iloc[:, 0]

    print("\n" + "=" * 60)
    print("大资金因子 IC 评估")
    print("=" * 60)
    print(f"{'因子':<20} {'IC均值':>8} {'ICIR':>7} {'t值':>7} {'覆盖率':>7}")
    print("-" * 60)

    from qmtquant.datafeed.qlib_export import to_qlib_code

    for name, df in factors.items():
        if df.empty:
            continue

        stacked = df.stack()
        stacked.index.names = ["datetime", "instrument"]

        # vt_symbol -> qlib code
        new_idx = stacked.index.set_levels(
            stacked.index.levels[1].map(to_qlib_code), level="instrument"
        )
        stacked.index = new_idx

        # label 的 index 顺序是 (instrument, datetime)，因子是 (datetime, instrument)
        stacked = stacked.swaplevel().sort_index()

        # 与 label 对齐
        aligned = pd.concat([stacked.rename("factor"),
                              label.rename("ret")], axis=1).dropna()
        if len(aligned) < 1000:
            print(f"{name:<20} 数据不足 ({len(aligned)} 条)")
            continue

        # 日频 IC
        ic_dict = {}
        for dt, g in aligned.groupby(level="datetime"):
            if len(g) >= 30:
                v = g["factor"].corr(g["ret"], method="spearman")
                if not np.isnan(v):
                    ic_dict[dt] = v
        ic = pd.Series(ic_dict).sort_index()

        if len(ic) < 10:
            print(f"{name:<20} IC 天数不足 ({len(ic)})")
            continue

        mean = ic.mean()
        std = ic.std(ddof=1)
        ir = mean / std if std > 0 else 0
        t = ir * np.sqrt(len(ic))
        coverage = len(aligned) / len(label.dropna())

        sig = "**" if abs(t) >= 2 else ""
        print(f"{name:<20} {mean:>+8.4f} {ir:>7.3f} {t:>7.2f} "
              f"{coverage:>6.1%} {sig}")


def main() -> int:
    cfg = get_config()
    data_dir = Path(cfg.data.store_dir) / "money_flow"
    qlib_dir = Path(cfg.data.store_dir) / "qlib_data"

    print("=" * 60)
    print("构建大资金因子")
    print("=" * 60)
    print(f"数据源: {data_dir.resolve()}")

    factors = {}

    print("\n--- 北向资金 ---")
    north = build_northbound_factor(data_dir)
    if not north.empty:
        factors["north_net_pct"] = north

    print("\n--- 龙虎榜 ---")
    dragon_net, dragon_count = build_dragon_factors(data_dir)
    if not dragon_net.empty:
        factors["dragon_inst_net"] = dragon_net
    if not dragon_count.empty:
        factors["dragon_count_20"] = dragon_count

    print("\n--- 融资融券 ---")
    margin_chg, margin_ratio = build_margin_factors(data_dir)
    if not margin_chg.empty:
        factors["margin_bal_chg"] = margin_chg
    if not margin_ratio.empty:
        factors["margin_buy_ratio"] = margin_ratio

    if not factors:
        print("\n没有构建出任何因子")
        return 1

    print(f"\n共构建 {len(factors)} 个因子")

    # 导出
    export_to_qlib(factors, qlib_dir)

    # IC 评估
    compute_factor_ic(factors, qlib_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
