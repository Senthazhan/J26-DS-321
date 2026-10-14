"""Stage 2 - Root-cause classifier: one calibrated Gradient Boosted model per cause.
Output: a probability for each of the 5 causes (multi-label)."""
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

from ..common import config


def _make(n_estimators, max_depth, learning_rate=0.05):
    base = GradientBoostingClassifier(n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      max_depth=max_depth,
                                      random_state=config.SEED)
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)   # Platt scaling


def fit_models(train, features, params):
    return {lab: _make(**params).fit(train[features], train[lab])
            for lab in config.LABELS}


def predict_probs_from(models, df, features):
    return np.column_stack([models[lab].predict_proba(df[features])[:, 1]
                            for lab in config.LABELS])


def fit_root_cause_models(train, val, features=None, verbose=True):
    """Choose settings on VAL (macro F1), then fit on TRAIN."""
    features = features or config.ROOT_CAUSE_FEATURES
    best = None
    for n_est in [100, 200]:
        for depth in [2, 3, 4]:
            params = {"n_estimators": n_est, "max_depth": depth}
            models = fit_models(train, features, params)
            probs = predict_probs_from(models, val, features)
            f1 = f1_score(val[config.LABELS].values, (probs >= 0.5).astype(int),
                          average="macro", zero_division=0)
            if verbose:
                print(f"  n_estimators={n_est:<4} max_depth={depth}  val macro F1={f1:.4f}")
            if best is None or f1 > best[0]:
                best = (f1, params)
    params = best[1]
    return {"models": fit_models(train, features, params),
            "features": list(features), "params": params}


def predict_probs(bundle, df):
    return predict_probs_from(bundle["models"], df, bundle["features"])


def save(bundle, path=config.ROOT_CAUSE_MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load(path=config.ROOT_CAUSE_MODEL_PATH):
    return joblib.load(path)
