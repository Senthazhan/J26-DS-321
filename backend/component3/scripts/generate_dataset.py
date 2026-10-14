"""
generate_dataset.py
--------------------
Component 3 — Real-Time Collaboration Intelligence & Adaptive Intervention
Synthetic dataset generator.

Produces TWO files:

1. experiment_trajectories.csv   (Dataset A — for Model 1: Anomaly Detection)
   One row per (run_id, epoch). Columns: the 5 raw signals over time.

2. run_level_labels.csv          (Dataset B — for Model 2: Root Cause Classifier)
   One row per run_id. Columns: summary features + context features +
   multi-label root-cause flags.

Run types generated (matches TAF numbers):
    healthy               : 500 runs
    overfitting           : 300 runs
    stalled               : 200 runs
    data_leakage          : 150 runs
    resource_bottleneck   : 150 runs
    premature_evaluation  : 100 runs
    TOTAL                 : 1400 runs  (~28,000 epoch-level rows)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RNG = np.random.default_rng(seed=42)

RUN_COUNTS = {
    "healthy": 500,
    "overfitting": 300,
    "stalled": 200,
    "data_leakage": 150,
    "resource_bottleneck": 150,
    "premature_evaluation": 100,
}

ROOT_CAUSES = [
    "overfitting",
    "stalled",
    "data_leakage",
    "resource_bottleneck",
    "premature_evaluation",
]


def make_healthy_run(n_epochs):
    """Train/val accuracy rise together smoothly; loss decays; GPU normal; steady commits."""
    x = np.arange(n_epochs)
    target = RNG.uniform(0.85, 0.94)
    train_acc = target * (1 - np.exp(-x / 6)) + RNG.normal(0, 0.01, n_epochs)
    val_acc = train_acc - RNG.normal(0.02, 0.01, n_epochs)  # stays close to train
    loss = 2.0 * np.exp(-x / 5) + 0.05 + RNG.normal(0, 0.02, n_epochs)
    gpu = RNG.uniform(55, 85, n_epochs)
    commits = RNG.poisson(2.2, n_epochs)
    return train_acc, val_acc, loss, gpu, commits


def make_overfitting_run(n_epochs):
    """Train accuracy keeps climbing; validation accuracy peaks then diverges/drops."""
    x = np.arange(n_epochs)
    peak_epoch = int(n_epochs * RNG.uniform(0.3, 0.5))
    train_acc = 0.99 * (1 - np.exp(-x / 4)) + RNG.normal(0, 0.008, n_epochs)
    val_acc = np.where(
        x <= peak_epoch,
        train_acc - RNG.normal(0.02, 0.01, n_epochs),
        (train_acc[peak_epoch] - RNG.normal(0.02, 0.01)) - 0.03 * (x - peak_epoch) + RNG.normal(0, 0.015, n_epochs),
    )
    val_acc = np.clip(val_acc, 0.1, 1.0)
    loss = 1.8 * np.exp(-x / 3.5) + 0.02 + RNG.normal(0, 0.02, n_epochs)
    gpu = RNG.uniform(55, 85, n_epochs)
    commits = RNG.poisson(2.0, n_epochs)
    return train_acc, val_acc, loss, gpu, commits


def make_stalled_run(n_epochs):
    """Metrics improve briefly then flatline for the rest of the run."""
    x = np.arange(n_epochs)
    stall_epoch = int(n_epochs * RNG.uniform(0.25, 0.4))
    ramp = 0.55 * (1 - np.exp(-x / 4))
    flat_val = ramp[stall_epoch]
    train_acc = np.where(x <= stall_epoch, ramp, flat_val) + RNG.normal(0, 0.006, n_epochs)
    val_acc = train_acc - RNG.normal(0.02, 0.008, n_epochs)
    loss = np.where(x <= stall_epoch, 2.0 * np.exp(-x / 4), 2.0 * np.exp(-stall_epoch / 4)) + RNG.normal(0, 0.01, n_epochs)
    gpu = RNG.uniform(55, 85, n_epochs)
    commits = np.where(x <= stall_epoch, RNG.poisson(2.0, n_epochs), RNG.poisson(0.3, n_epochs))
    return train_acc, val_acc, loss, gpu, commits


def make_leakage_run(n_epochs):
    """Both accuracies jump unnaturally high from epoch 1 (near-perfect, too early)."""
    x = np.arange(n_epochs)
    train_acc = 0.99 - RNG.normal(0, 0.005, n_epochs)
    val_acc = 0.985 - RNG.normal(0, 0.006, n_epochs)  # suspiciously close to train, from the start
    loss = 0.05 * np.exp(-x / 2) + RNG.normal(0, 0.005, n_epochs)
    gpu = RNG.uniform(55, 85, n_epochs)
    commits = RNG.poisson(1.8, n_epochs)
    return train_acc, val_acc, loss, gpu, commits


def make_bottleneck_run(n_epochs):
    """GPU pinned near 100%, metrics barely move, commits stop partway through."""
    x = np.arange(n_epochs)
    stop_epoch = int(n_epochs * RNG.uniform(0.3, 0.5))
    train_acc = 0.35 + 0.05 * (1 - np.exp(-x / 8)) + RNG.normal(0, 0.008, n_epochs)
    val_acc = train_acc - RNG.normal(0.02, 0.008, n_epochs)
    loss = 1.9 * np.exp(-x / 12) + RNG.normal(0, 0.015, n_epochs)
    gpu = RNG.uniform(96, 100, n_epochs)
    commits = np.where(x <= stop_epoch, RNG.poisson(2.0, n_epochs), 0)
    return train_acc, val_acc, loss, gpu, commits


def make_premature_eval_run(n_epochs_short):
    """Run ends very early, before metrics have properly converged or been tuned."""
    x = np.arange(n_epochs_short)
    train_acc = 0.5 * (1 - np.exp(-x / 6)) + RNG.normal(0, 0.01, n_epochs_short)
    val_acc = train_acc - RNG.normal(0.03, 0.012, n_epochs_short)
    loss = 2.0 * np.exp(-x / 6) + RNG.normal(0, 0.02, n_epochs_short)
    gpu = RNG.uniform(55, 85, n_epochs_short)
    commits = RNG.poisson(1.5, n_epochs_short)
    return train_acc, val_acc, loss, gpu, commits


GENERATORS = {
    "healthy": make_healthy_run,
    "overfitting": make_overfitting_run,
    "stalled": make_stalled_run,
    "data_leakage": make_leakage_run,
    "resource_bottleneck": make_bottleneck_run,
    "premature_evaluation": make_premature_eval_run,
}


def build_datasets():
    trajectory_rows = []
    run_level_rows = []
    run_id_counter = 1

    for run_type, count in RUN_COUNTS.items():
        for _ in range(count):
            run_id = f"run_{run_id_counter:05d}"
            run_id_counter += 1

            if run_type == "premature_evaluation":
                n_epochs = int(RNG.integers(3, 7))  # short run = the anomaly itself
            else:
                n_epochs = int(RNG.integers(15, 31))

            train_acc, val_acc, loss, gpu, commits = GENERATORS[run_type](n_epochs)

            for epoch in range(n_epochs):
                trajectory_rows.append({
                    "run_id": run_id,
                    "epoch": epoch + 1,
                    "train_accuracy": round(float(np.clip(train_acc[epoch], 0, 1)), 4),
                    "val_accuracy": round(float(np.clip(val_acc[epoch], 0, 1)), 4),
                    "loss": round(float(max(loss[epoch], 0)), 4),
                    "gpu_utilization_pct": round(float(np.clip(gpu[epoch], 0, 100)), 2),
                    "commits_that_day": int(max(commits[epoch], 0)),
                })

            # ---- run-level context features for Model 2 ----
            is_anomaly = 0 if run_type == "healthy" else 1
            root_cause_flags = {rc: 0 for rc in ROOT_CAUSES}
            if run_type != "healthy":
                root_cause_flags[run_type] = 1
                # small chance of a co-occurring secondary cause, for genuine multi-label cases
                if RNG.random() < 0.12:
                    other = RNG.choice([rc for rc in ROOT_CAUSES if rc != run_type])
                    root_cause_flags[other] = 1

            hyperparam_tuning_logged = 0 if run_type == "premature_evaluation" else int(RNG.random() < 0.85)
            drift_severity_score = round(float(RNG.uniform(0.5, 0.9) if is_anomaly else RNG.uniform(0.0, 0.3)), 3)
            commit_gap_days = round(float(RNG.uniform(2, 7) if run_type == "resource_bottleneck" else RNG.uniform(0, 1.5)), 2)
            notebook_edit_freq = round(float(RNG.uniform(0, 2) if run_type in ("stalled", "resource_bottleneck") else RNG.uniform(2, 8)), 2)

            run_level_rows.append({
                "run_id": run_id,
                "run_type": run_type,
                "is_anomaly": is_anomaly,
                "final_train_accuracy": round(float(np.clip(train_acc[-1], 0, 1)), 4),
                "final_val_accuracy": round(float(np.clip(val_acc[-1], 0, 1)), 4),
                "train_val_gap": round(float(np.clip(train_acc[-1], 0, 1) - np.clip(val_acc[-1], 0, 1)), 4),
                "n_epochs_run": n_epochs,
                "avg_gpu_utilization_pct": round(float(np.mean(gpu)), 2),
                "hyperparam_tuning_logged": hyperparam_tuning_logged,
                "drift_severity_score": drift_severity_score,
                "commit_gap_days": commit_gap_days,
                "notebook_edit_freq_per_day": notebook_edit_freq,
                **{f"cause_{rc}": v for rc, v in root_cause_flags.items()},
            })

    trajectory_df = pd.DataFrame(trajectory_rows)
    run_level_df = pd.DataFrame(run_level_rows)
    return trajectory_df, run_level_df


if __name__ == "__main__":
    trajectory_df, run_level_df = build_datasets()

    trajectory_df.to_csv(DATA_DIR / "experiment_trajectories.csv", index=False)
    run_level_df.to_csv(DATA_DIR / "run_level_labels.csv", index=False)

    print("Dataset A — experiment_trajectories.csv")
    print(f"  Runs: {run_level_df.shape[0]}, Epoch-level rows: {trajectory_df.shape[0]}")
    print(trajectory_df.head(3).to_string(index=False))
    print()
    print("Dataset B — run_level_labels.csv")
    print(f"  Rows: {run_level_df.shape[0]}, Columns: {run_level_df.shape[1]}")
    print(run_level_df["run_type"].value_counts())
    print()
    print(run_level_df.head(3).to_string(index=False))
