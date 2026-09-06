from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    from .train_model import DATA, FEATURES, OUT, make_features
except ImportError:
    from train_model import DATA, FEATURES, OUT, make_features

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=str(DATA)); ap.add_argument('--horizon', type=int, default=15); args = ap.parse_args()
    df = pd.read_parquet(args.data).sort_index()
    x = make_features(df)
    future = df.close.shift(-args.horizon) / df.close - 1
    y = (future > 0.0005).astype(int)
    valid = future.notna(); x, y = x.loc[valid], y.loc[valid]
    days = pd.Index(x.index.normalize()).unique(); cut = days[max(1, int(len(days)*0.70))-1]
    train = x.index.normalize() <= cut; xt, xv = x.loc[train, FEATURES], x.loc[~train, FEATURES]; yt, yv = y.loc[train], y.loc[~train]
    model = LGBMClassifier(n_estimators=260, learning_rate=0.035, num_leaves=15, max_depth=5,
                           min_child_samples=80, subsample=0.85, colsample_bytree=0.9,
                           reg_alpha=0.2, reg_lambda=1.0, objective='binary', random_state=42,
                           verbosity=-1, n_jobs=-1)
    model.fit(xt, yt)
    prob = model.predict_proba(xv)[:, 1]; pred = (prob >= 0.5).astype(int)
    metrics = {'symbol':'920368.BSE','model':'LightGBMClassifier','horizon_bars':args.horizon,
      'features':FEATURES,'train_start':str(xt.index[0]),'train_end':str(xt.index[-1]),
      'test_start':str(xv.index[0]),'test_end':str(xv.index[-1]),'train_samples':int(len(xt)),
      'test_samples':int(len(xv)),'accuracy':float(accuracy_score(yv,pred)),
      'auc':float(roc_auc_score(yv,prob)) if yv.nunique()>1 else None}
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model':model,'features':FEATURES,'horizon':args.horizon}, OUT/'gbm_t_model.joblib')
    (OUT/'gbm_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame({'feature':FEATURES,'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(OUT/'gbm_feature_importance.csv',index=False)
    pd.DataFrame({'datetime':xv.index,'prob_up':prob,'label':yv.values}).to_csv(OUT/'gbm_predictions.csv',index=False)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
