"""Turn the epoch-by-epoch signals of each run into ONE row of summary features."""
import numpy as np
import pandas as pd

from .config import SIGNALS


def _features_for_run(g: pd.DataFrame) -> dict:
    g = g.sort_values("epoch")
    out = {"n_epochs": len(g)}
    x = g["epoch"].to_numpy(dtype=float)
    for s in SIGNALS:
        v = g[s].to_numpy(dtype=float)
        out[f"{s}_mean"] = v.mean()
        out[f"{s}_std"] = v.std()
        out[f"{s}_first"] = v[0]
        out[f"{s}_last"] = v[-1]
        out[f"{s}_slope"] = np.polyfit(x, v, 1)[0] if len(v) > 1 else 0.0
    gap = g["train_accuracy"].to_numpy() - g["val_accuracy"].to_numpy()
    out["gap_mean"] = gap.mean()
    out["gap_last"] = gap[-1]
    out["val_peak_minus_last"] = g["val_accuracy"].max() - g["val_accuracy"].iloc[-1]
    half = len(g) // 2
    out["loss_late_change"] = (abs(g["loss"].iloc[-1] - g["loss"].iloc[half])
                               if len(g) > 1 else 0.0)
    out["commits_zero_share"] = float((g["commits_that_day"] == 0).mean())
    return out


def build_run_features(traj: pd.DataFrame) -> pd.DataFrame:
    """Epoch-level table -> one row per run (run_id + 31 features)."""
    rows = []
    for run_id, g in traj.groupby("run_id"):
        row = _features_for_run(g)
        row["run_id"] = run_id
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[["run_id"] + [c for c in df.columns if c != "run_id"]]


def feature_names(feats: pd.DataFrame) -> list:
    return [c for c in feats.columns if c != "run_id"]
