"""盘中全市场日内预测 —— 用训练好的 GBM 对全市场实时打分。

## 数据流

    data/clean/1m/    （update_intraday.py 实时追加）
        │  读取最新 bar
        ▼
    make_intraday_features()  → 24 维特征
        │
        ▼
    model.predict_proba()  → 每只股票的上涨概率
        │
        ▼
    predictions/{date}/intraday_scores.csv
        │
        ├──> 日内做T策略：筛选 prob > 阈值 且持有底仓的票
        ├──> 打板策略：筛选 prob 最高 + 量能放大的票
        └──> 均值回归：筛选 prob 反转 + 偏离均线的票

## 输出格式

    datetime,symbol,prob_up,close,volume,day_ret,vol_z_30,vwap_gap
    2026-09-06 10:30:00,600519.SSE,0.72,1680.5,1234,...
    2026-09-06 10:30:00,000001.SZSE,0.45,10.2,98765,...

prob_up > 0.5 的为模型看多，< 0.5 为看空。附带关键特征供人工过滤。

## 用法

    # 全市场最新一根 bar 预测
    python scripts/predict_intraday.py

    # 只预测指定标的
    python scripts/predict_intraday.py --symbols 600519.SH 000001.SZ

    # 持续模式：每分钟预测一次
    python scripts/predict_intraday.py --loop 60

    # 只输出 top N
    python scripts/predict_intraday.py --top 50
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "intraday_gbm"
PREDICTIONS_DIR = ROOT / "predictions"


def load_model(model_dir: Path) -> dict:
    """加载训练好的模型。"""
    p = model_dir / "model.joblib"
    if not p.exists():
        raise FileNotFoundError(
            f"模型不存在: {p}\n请先运行: python scripts/train_intraday_gbm.py")
    return joblib.load(p)


def get_latest_features(store: Path, symbols: list[str] | None,
                        features: list[str],
                        lookback: int = 120) -> pd.DataFrame:
    """读取全市场最新 bar 并计算特征。

    :param lookback: 每只票读最后 N 根 bar（特征窗口需要历史）
    """
    from qmtquant.features.intraday import make_intraday_features

    clean_1m = store / "clean" / "1m"
    if not clean_1m.exists():
        clean_1m = store / "1m"

    files = []
    for ex in sorted(clean_1m.iterdir()):
        if not ex.is_dir():
            continue
        for f in sorted(ex.glob("*.parquet")):
            vt = f"{f.stem}.{ex.name}"
            if symbols and vt not in symbols:
                continue
            files.append((f, vt))

    rows = []
    for path, vt in files:
        try:
            df = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        if len(df) < 60:
            continue

        # 只取最后 lookback 根 bar 计算特征（节省内存）
        df = df.iloc[-lookback:]
        feat = make_intraday_features(df)

        # 取最后一行（最新时刻的特征）
        last = feat.iloc[-1:].copy()
        last["symbol"] = vt
        last["close"] = float(df["close"].iloc[-1])
        last["volume"] = float(df["volume"].iloc[-1])
        last["amount"] = float(df["amount"].iloc[-1])
        rows.append(last)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0)


def predict(model_data: dict, df: pd.DataFrame) -> pd.DataFrame:
    """对全市场打分。"""
    model = model_data["model"]
    features = model_data["features"]

    x = df[features].fillna(0)
    prob = model.predict_proba(x)[:, 1]

    result = pd.DataFrame({
        "datetime": df.index,
        "symbol": df["symbol"].values,
        "prob_up": prob,
        "close": df["close"].values,
        "volume": df["volume"].values,
        "day_ret": df["day_ret"].values if "day_ret" in df.columns else 0,
        "vol_z_30": df["vol_z_30"].values if "vol_z_30" in df.columns else 0,
        "vwap_gap": df["vwap_gap"].values if "vwap_gap" in df.columns else 0,
    })
    return result.sort_values("prob_up", ascending=False)


def save_predictions(result: pd.DataFrame, output_dir: Path) -> Path:
    """保存预测结果。"""
    today = datetime.now().strftime("%Y%m%d")
    d = output_dir / today
    d.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%H%M%S")
    path = d / f"intraday_scores_{ts}.csv"
    result.to_csv(path, index=False)

    # 同时维护一个最新快照
    latest = d / "intraday_scores_latest.csv"
    result.to_csv(latest, index=False)

    return path


def is_trading_time() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    return (930 <= t <= 1130) or (1300 <= t <= 1500)


def main() -> int:
    p = argparse.ArgumentParser(description="盘中全市场日内预测")
    p.add_argument("--model-dir", default=str(MODEL_DIR))
    p.add_argument("--symbols", nargs="*", default=[])
    p.add_argument("--top", type=int, default=0,
                   help="只输出 top N 标的（0=全部）")
    p.add_argument("--loop", type=int, default=0,
                   help="持续运行，每 N 秒预测一次（0=单次）")
    p.add_argument("--output", default=str(PREDICTIONS_DIR))
    args = p.parse_args()

    from qmtquant.config import get_config
    from qmtquant.utils.symbol import normalize

    cfg = get_config()
    store = Path(cfg.data.store_dir)

    print("=" * 58)
    print("  全市场日内预测")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 58)

    model_data = load_model(Path(args.model_dir))
    features = model_data["features"]
    horizon = model_data.get("horizon", 10)
    print(f"  模型特征 {len(features)} 维，预测窗口 {horizon} bar")

    symbols = [normalize(s) for s in args.symbols] if args.symbols else None

    from qmtquant.engine.backtest_engine import _load_st_checker
    is_st = _load_st_checker()
    if is_st is None:
        print("  [!] 加载不到 ST 历史，信号里可能含 ST 标的 —— "
              "它们会被实盘风控拒单并占掉 top-N 名额")

    def run_once():
        ts = datetime.now().strftime('%H:%M:%S')
        df = get_latest_features(store, symbols, features)
        if df.empty:
            print(f"  [{ts}] 无数据")
            return

        result = predict(model_data, df)

        # ST 标的剔除 —— 实盘风控 forbid_st 会拒绝买入它们
        # （见 RiskManager._do_check）。不在这里剔，它们会占掉 top-N
        # 的名额，下游拿到一份「50 只信号、其中若干必被拒单」的清单，
        # 而拒单发生在风控层，看起来像是策略选错了标的。
        #
        # 这里按**当前**状态判定，与实盘的时点一致
        # （回测那边要按区间判，见 backtest_intraday_gbm 的说明）。
        if is_st is not None and not result.empty:
            now = datetime.now()
            keep = []
            for sym in result["symbol"]:
                try:
                    keep.append(not is_st(sym, now))
                except Exception:               # noqa: BLE001
                    keep.append(True)           # 判不出就放行，与风控一致
            n_drop = len(keep) - sum(keep)
            if n_drop:
                result = result[keep]
                print(f"  [{ts}] 剔除 ST 标的 {n_drop} 只")

        if args.top:
            result = result.head(args.top)

        if result.empty:
            print(f"  [{ts}] 过滤后无可用信号")
            return

        path = save_predictions(result, Path(args.output))

        n_bullish = (result["prob_up"] > 0.5).sum()
        n_bearish = (result["prob_up"] < 0.5).sum()
        print(f"  [{ts}] {len(result)} 只 | "
              f"看多 {n_bullish} 看空 {n_bearish} | "
              f"top1 {result.iloc[0]['symbol']} "
              f"prob={result.iloc[0]['prob_up']:.3f}")

    if not args.loop:
        run_once()
        return 0

    print(f"\n持续模式，每 {args.loop} 秒预测一次，Ctrl+C 退出")
    round_n = 0
    try:
        while True:
            if is_trading_time():
                round_n += 1
                run_once()
            else:
                now = datetime.now()
                print(f"\r  {now.strftime('%H:%M:%S')} 非交易时段",
                      end="", flush=True)
            time.sleep(args.loop)
    except KeyboardInterrupt:
        print(f"\n已停止，共运行 {round_n} 轮")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
