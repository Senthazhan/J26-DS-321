"""
ks_validation.py
Validates the synthetic experiment dataset (Component 3) against a real,
publicly-available Kaggle training log using a Kolmogorov-Smirnov test.

Real reference: Kaggle notebook "detect-fake-text-kerasnlp-tf-torch-jax-train"
(KerasNLP DeBERTa-v3 model, tracked via Weights & Biases)
"""
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

real = pd.read_csv(DATA_DIR / "real_reference_sample.csv")
real_loss = real["val_loss"].tolist()

traj = pd.read_csv(DATA_DIR / "experiment_trajectories.csv")
labels = pd.read_csv(DATA_DIR / "run_level_labels.csv")
healthy_run_ids = labels[labels["run_type"] == "healthy"]["run_id"].tolist()
healthy_traj = traj[traj["run_id"].isin(healthy_run_ids)]

# Compare converged states: final epoch of each synthetic healthy run
# against the real (near-final) validation loss values.
final_epoch_loss = healthy_traj.loc[healthy_traj.groupby("run_id")["epoch"].idxmax()]["loss"].tolist()

stat, p = stats.ks_2samp(real_loss, final_epoch_loss)
print(f"KS statistic: {stat:.4f}")
print(f"p-value: {p:.4f}")
print("Result:", "PASS (p > 0.05, distributions statistically consistent)" if p > 0.05 else "FAIL")
