"""Stage 1 - Anomaly detection (Isolation Forest) -> anomaly score between 0 and 1."""
from itertools import product

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from ..common import config


def _fit(train_healthy, features, n_estimators, max_samples):
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                            random_state=config.SEED, n_jobs=-1)
    return model.fit(train_healthy[features])


def _raw(model, df, features):
    return -model.score_samples(df[features])       # higher = more anomalous


def fit_anomaly_model(data, features, verbose=True):
    """Learn 'normal' from healthy TRAIN runs, choose settings on VAL.
    `data` needs: features, is_anomaly, split."""
    train = data[data["split"] == "train"]
    val = data[data["split"] == "val"]
    healthy = train[train["is_anomaly"] == 0]

    best = None
    for n_est, max_s in product([100, 200, 400], [64, 128, 256]):
        m = _fit(healthy, features, n_est, max_s)
        auc = roc_auc_score(val["is_anomaly"], _raw(m, val, features))
        if verbose:
            print(f"  n_estimators={n_est:<4} max_samples={max_s:<4} val AUC={auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, n_est, max_s)
    _, n_est, max_s = best
    model = _fit(healthy, features, n_est, max_s)

    tr = _raw(model, train, features)               # 0-1 range learned from train only
    return {"model": model, "features": list(features),
            "lo": float(tr.min()), "hi": float(tr.max()),
            "params": {"n_estimators": n_est, "max_samples": max_s}}


def score(bundle, df):
    """Anomaly score in [0, 1] for each row of `df` (0 = normal, 1 = very strange)."""
    raw = _raw(bundle["model"], df, bundle["features"])
    return np.clip((raw - bundle["lo"]) / (bundle["hi"] - bundle["lo"]), 0, 1)


def save(bundle, path=config.ANOMALY_MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load(path=config.ANOMALY_MODEL_PATH):
    return joblib.load(path)
