# Component 3 - Real-Time Collaboration Intelligence & Adaptive Intervention

Watches a data science team's experiment runs and answers three questions:

1. **Is this run healthy or anomalous?** (Stage 1 - Isolation Forest, anomaly score 0-1)
2. **If anomalous, why?** (Stage 2 - calibrated Gradient Boosting, a probability for each of 5 root causes: overfitting, stalled, data leakage, resource bottleneck, premature evaluation)
3. **How urgent is it and what should the team do?** (Stage 3 - `severity = 0.6 * anomaly + 0.4 * top cause probability`, level LOW / MEDIUM / HIGH, template advice)

The trained models are compared with rule-based baselines in an ablation study.

> The data is **synthetic** (1,400 labelled runs made by `scripts/generate_dataset.py`), so the near-perfect scores below show the pipeline works on clean simulated failures, not that it will perform as well on real experiments.

## Folder layout

```
component3/
  train_pipeline.py        runs everything and saves models + results
  common/                  shared parts
    config.py                paths, seed, names of signals and causes
    splits.py                train / val / test split by run (70/15/15)
    features.py              epoch rows -> one row of 31 features per run
  stage1_anomaly/          Stage 1
    anomaly_model.py         Isolation Forest + 0-1 anomaly score
  stage2_root_cause/       Stage 2
    root_cause_model.py      calibrated Gradient Boosting (5 causes)
  stage3_severity/         Stage 3
    severity.py              severity score, level, advice templates
  evaluation/              proving the ML beats simple rules
    baseline.py              rule-based baselines A and B
    ablation.py              baseline vs ML comparison + feature ablations
  data/                    the datasets, splits.csv, real reference sample
  models/                  saved models (created by train_pipeline)
  results/                 metrics and per-run outputs (created by train_pipeline)
  scripts/                 generate_dataset.py, ks_validation.py
```

## How to run

From the **repository root** (the folder that contains `backend/`), with the packages from `backend/requirements.txt` installed:

```
python -m backend.component3.train_pipeline
```

This takes about a minute and prints the results of every stage. It saves the models to `models/` and the outputs to `results/` (`metrics.json` has the summary).

Check the realism of the synthetic data (KS test, expected p = 0.136):

```
python backend/component3/scripts/ks_validation.py
```

Do **not** re-run `scripts/generate_dataset.py` unless you want to rebuild the data: it overwrites the CSVs in `data/`.

## Results (test split, 210 runs)

| Check | Result | Target |
|---|---|---|
| Stage 1 AUC-ROC | 0.9998 | > 0.85 |
| Stage 2 macro F1 | 0.9568 | > 0.80 |
| Stage 2 Brier score | 0.0157 | < 0.15 |
| Stage 2 Hamming loss | 0.0114 | lower than baseline |
| Macro F1 gain vs Baseline A (generic rules) | +21.9% | >= 10% (met) |
| Macro F1 gain vs Baseline B (best single rule per cause) | +5.6% | >= 10% (not met) |
| HIGH-severity cases correctly triggered | 133 / 133 | > 85% |
| KS test (synthetic vs real loss values) | p = 0.136 | > 0.05 |

## Limitations

- Synthetic, cleanly separated data: scores are likely higher than on real experiments.
- The KS test uses only 6 real values from a different task, so it is a basic realism check.
- Removing the anomaly score or the drift score from the Stage 2 inputs did not change its results on this data.
- `drift_severity_score` (from Component 2) is a simulated feature that separates classes very easily.
- Runs use the CSV files; live collection from MLflow / W&B / Git is future work.

## Status

Stages 1-3, the baselines and the ablation study are finished. The live prototype (`predict_live.py`) and the JSON contract for Component 4 are still to be added.
