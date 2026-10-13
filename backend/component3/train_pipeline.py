"""Train and evaluate the whole Component 3 pipeline in one go.

Run from the REPOSITORY ROOT (the folder that contains `backend/`):

    python -m backend.component3.train_pipeline

It trains Stage 1 and Stage 2, runs the baseline/ablation study and the severity
engine, then saves:
    models/anomaly_model.joblib, models/root_cause_models.joblib
    results/anomaly_scores.csv, root_cause_probs.csv, ablation_results.csv,
            intervention_results.csv, metrics.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, f1_score, hamming_loss,
                             roc_auc_score)

from .common import config, features
from .common.splits import load_or_make_splits
from .evaluation import ablation, baseline
from .stage1_anomaly import anomaly_model
from .stage2_root_cause import root_cause_model as rcm
from .stage3_severity import severity

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def main():
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    traj = pd.read_csv(config.TRAJECTORIES_CSV)
    runs = pd.read_csv(config.RUN_LABELS_CSV)
    split = load_or_make_splits(runs)
    print(f"Runs: {runs['run_id'].nunique()} | epoch rows: {len(traj)} | "
          f"split: {split['split'].value_counts().to_dict()}")

    # ---------------- Stage 1: anomaly detection ----------------
    print("\n=== STAGE 1: anomaly detection (Isolation Forest) ===")
    feats = features.build_run_features(traj)
    names = features.feature_names(feats)
    d1 = feats.merge(runs[["run_id", "run_type", "is_anomaly"]], on="run_id") \
              .merge(split, on="run_id")
    a_bundle = anomaly_model.fit_anomaly_model(d1, names)
    anomaly_model.save(a_bundle)
    d1["anomaly_score"] = anomaly_model.score(a_bundle, d1)
    t1 = d1[d1["split"] == "test"]
    auc = roc_auc_score(t1["is_anomaly"], t1["anomaly_score"])
    print(f"Chosen: {a_bundle['params']}")
    print(f"TEST AUC-ROC: {auc:.4f}   (target > 0.85)")
    d1[["run_id", "run_type", "is_anomaly", "split", "anomaly_score"]] \
        .to_csv(config.RESULTS_DIR / "anomaly_scores.csv", index=False)

    # ---------------- Stage 2: root-cause classifier ----------------
    print("\n=== STAGE 2: root-cause classifier (calibrated Gradient Boosting) ===")
    d2 = runs.merge(split, on="run_id").merge(d1[["run_id", "anomaly_score"]], on="run_id")
    train, val, test = (d2[d2["split"] == s] for s in ("train", "val", "test"))
    rc_bundle = rcm.fit_root_cause_models(train, val)
    rcm.save(rc_bundle)
    probs = rcm.predict_probs(rc_bundle, test)
    y = test[config.LABELS].values
    pred = (probs >= 0.5).astype(int)
    macro = f1_score(y, pred, average="macro", zero_division=0)
    ham = hamming_loss(y, pred)
    brier = float(np.mean([brier_score_loss(y[:, i], probs[:, i]) for i in range(5)]))
    print(f"Chosen: {rc_bundle['params']}")
    print(f"TEST macro F1: {macro:.4f} (target > 0.80) | Hamming loss: {ham:.4f} | "
          f"Brier: {brier:.4f} (target < 0.15)")

    all_probs = rcm.predict_probs(rc_bundle, d2)
    d2[config.PROB_COLS] = all_probs
    d2[["run_id", "run_type", "is_anomaly", "split", "anomaly_score"] + config.PROB_COLS] \
        .to_csv(config.RESULTS_DIR / "root_cause_probs.csv", index=False)

    # ---------------- Baselines + ablation ----------------
    print("\n=== BASELINES + ABLATION (test split) ===")
    table, ci, per_cause = ablation.run_ablation(d2, baseline.baseline_a(traj), rc_bundle)
    print(table.to_string(index=False))
    f1_ml = float(table.loc[table.system == "Trained ML (full)", "macro_F1"].iloc[0])
    gains = {}
    for key, nm in [("vs_baseline_A", "Baseline A (generic rules)"),
                    ("vs_baseline_B", "Baseline B (best single rule per cause, fit on train)")]:
        f1_b = float(table.loc[table.system == nm, "macro_F1"].iloc[0])
        rel = (f1_ml - f1_b) / max(f1_b, 1e-9)
        gains[key] = {"baseline_F1": f1_b, "relative_gain": rel,
                      "target_10pct_met": bool(rel >= 0.10), "bootstrap_95_CI_abs": ci[key]}
        print(f"ML {key}: F1 {f1_b:.4f} -> {f1_ml:.4f} ({100 * rel:+.1f}% relative) "
              f"| 10% target {'MET' if rel >= 0.10 else 'NOT MET'} "
              f"| 95% CI of gain {ci[key][0]:+.3f} .. {ci[key][1]:+.3f}")
    print("\nPer-cause F1:\n" + per_cause.to_string(index=False))
    table.to_csv(config.RESULTS_DIR / "ablation_results.csv", index=False)

    # ---------------- Stage 3: severity + intervention ----------------
    print("\n=== STAGE 3: severity + intervention (test split) ===")
    rows = []
    for _, r in d2.iterrows():
        p = r[config.PROB_COLS].to_numpy(dtype=float)
        s = severity.severity_score(r["anomaly_score"], p)
        lvl = severity.severity_level(s)
        top, second, text = severity.recommend(lvl, p)
        rows.append({"run_id": r["run_id"], "severity": s, "level": lvl,
                     "recommended_cause": top, "second_cause": second,
                     "recommendation": text,
                     "recommended_correct": bool(top is not None and r[f"cause_{top}"] == 1)})
    sev = d2[["run_id", "run_type", "is_anomaly", "split", "anomaly_score"]] \
        .merge(pd.DataFrame(rows), on="run_id")
    sev.to_csv(config.RESULTS_DIR / "intervention_results.csv", index=False)
    st = sev[sev["split"] == "test"]
    print(pd.crosstab(st["level"], st["run_type"]))
    high = st[st["level"] == "HIGH"]
    ok = int(((high["is_anomaly"] == 1) & high["recommended_correct"]).sum())
    rate = ok / len(high) if len(high) else float("nan")
    print(f"HIGH cases: {len(high)} | correct: {ok} ({100 * rate:.1f}%, target > 85%)")
    print(f"Anomalous runs alerted: {int((st[st.is_anomaly == 1]['level'] != 'LOW').sum())}"
          f" / {int((st.is_anomaly == 1).sum())} | false alarms on healthy runs: "
          f"{int((st[st.is_anomaly == 0]['level'] != 'LOW').sum())}"
          f" / {int((st.is_anomaly == 0).sum())}")

    metrics = {
        "stage1_test_AUC_ROC": auc,
        "stage2_test": {"macro_F1": macro, "hamming_loss": ham, "brier": brier},
        "ablation_gains": gains,
        "stage3": {"high_cases": len(high), "high_correct": ok, "correct_rate": rate},
        "chosen_params": {"anomaly": a_bundle["params"], "root_cause": rc_bundle["params"]},
    }
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved models/ and results/ (see metrics.json for the summary).")


if __name__ == "__main__":
    main()
