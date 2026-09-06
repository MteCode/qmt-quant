from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "clean" / "1m" / "BSE" / "920368.parquet"
OUT = Path(__file__).resolve().parent / "models"
FEATURES = ["ret1", "ret3", "ret5", "ret10", "ma_gap", "rsi", "vol_z", "vwap_gap", "tod"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_index()
    close, vol = x["close"].astype(float), x["volume"].astype(float)
    out = pd.DataFrame(index=x.index)
    for n in (1, 3, 5, 10):
        out[f"ret{n}"] = close.pct_change(n)
    out["ma_gap"] = close / close.rolling(20, min_periods=5).mean() - 1
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(14, min_periods=5).mean() / down.rolling(14, min_periods=5).mean().replace(0, np.nan)
    out["rsi"] = (100 - 100 / (1 + rs)).fillna(50) / 100
    out["vol_z"] = (vol - vol.rolling(30, min_periods=5).mean()) / vol.rolling(30, min_periods=5).std().replace(0, np.nan)
    day = x.index.normalize()
    vwap = (x["amount"].groupby(day).cumsum() / x["volume"].replace(0, np.nan).groupby(day).cumsum()).replace([np.inf, -np.inf], np.nan)
    out["vwap_gap"] = (close / vwap - 1).fillna(0)
    out["tod"] = (x.index.hour * 60 + x.index.minute - 570) / 240
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--horizon", type=int, default=3)
    args = ap.parse_args()
    df = pd.read_parquet(args.data).sort_index()
    feat = make_features(df)
    future = df["close"].shift(-args.horizon) / df["close"] - 1
    # 以 3bp 过滤噪声，标签 1=未来上涨，0=下跌/横盘
    y = (future > 0.0003).astype(int)
    valid = future.notna()
    feat, y = feat.loc[valid], y.loc[valid]
    days = pd.Index(feat.index.normalize()).unique()
    cut_day = days[max(1, int(len(days) * 0.7)) - 1]
    train = feat.index.normalize() <= cut_day
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=0.5, max_iter=300, class_weight="balanced"))])
    model.fit(feat.loc[train, FEATURES], y.loc[train])
    pred = model.predict_proba(feat.loc[~train, FEATURES])[:, 1]
    pred_label = (pred >= 0.5).astype(int)
    metrics = {
        "symbol": "920368.BSE", "horizon_bars": args.horizon,
        "features": FEATURES, "train_start": str(feat.index[train][0]),
        "train_end": str(feat.index[train][-1]), "test_start": str(feat.index[~train][0]),
        "test_end": str(feat.index[~train][-1]), "train_samples": int(train.sum()),
        "test_samples": int((~train).sum()), "accuracy": float(accuracy_score(y.loc[~train], pred_label)),
        "auc": float(roc_auc_score(y.loc[~train], pred)) if y.loc[~train].nunique() > 1 else None,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "horizon": args.horizon}, OUT / "logistic_t_model.joblib")
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"datetime": feat.loc[~train].index, "prob_up": pred, "label": y.loc[~train].values}).to_csv(OUT / "predictions.csv", index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
