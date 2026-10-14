"""Rule-based baselines (what a team could write by hand), used for the ablation study.

Baseline A: generic fixed rules on the epoch curves (thresholds chosen up front).
Baseline B: best single-threshold rule per cause, fitted on TRAIN (a harder test).
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ..common import config

RULE_FEATURES = ["final_train_accuracy", "final_val_accuracy", "train_val_gap",
                 "n_epochs_run", "avg_gpu_utilization_pct", "hyperparam_tuning_logged",
                 "commit_gap_days", "notebook_edit_freq_per_day"]
RULE_A_COLS = [f"rule_a_{c}" for c in config.CAUSES]


def rule_flags(g: pd.DataFrame) -> list:
    """Baseline A rules for ONE run. Returns [overfit, stalled, leak, gpu, premature]."""
    g = g.sort_values("epoch")
    tr = g["train_accuracy"].to_numpy()
    va = g["val_accuracy"].to_numpy()
    loss = g["loss"].to_numpy()
    n = len(g)
    best_val, since = -1, 0
    for v in va:                                   # epochs since val accuracy last improved
        if v > best_val + 0.001:
            best_val, since = v, 0
        else:
            since += 1
    over = int(since >= 5 and n > 5 and tr[-1] > tr[-6])
    stalled = int(n > 5 and (loss[-6] - loss[-1]) / max(abs(loss[-6]), 1e-9) < 0.01)
    leak = int(va[-1] > 0.95)
    gpu = int(g["gpu_utilization_pct"].mean() > 90)
    prem = int(n < 10)
    return [over, stalled, leak, gpu, prem]


def baseline_a(traj: pd.DataFrame) -> pd.DataFrame:
    rows = [[run_id] + rule_flags(g) for run_id, g in traj.groupby("run_id")]
    return pd.DataFrame(rows, columns=["run_id"] + RULE_A_COLS)


def baseline_b(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    pred = np.zeros((len(test), len(config.LABELS)), dtype=int)
    for i, lab in enumerate(config.LABELS):
        stump = DecisionTreeClassifier(max_depth=1, random_state=config.SEED)
        stump.fit(train[RULE_FEATURES], train[lab])
        pred[:, i] = stump.predict(test[RULE_FEATURES])
    return pred
