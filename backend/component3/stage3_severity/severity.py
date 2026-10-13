"""Stage 3 - Severity score, LOW/MEDIUM/HIGH level and template recommendation."""
import numpy as np

from ..common.config import CAUSES

W_ANOMALY, W_CAUSE = 0.6, 0.4          # from the proposal
MEDIUM_AT, HIGH_AT = 0.40, 0.70        # chosen from the validation severity distribution
SECOND_CAUSE_AT = 0.50                 # mention a 2nd cause if its probability >= this

TEMPLATES = {
    "overfitting": "Training accuracy is far ahead of validation accuracy. Add regularisation "
                   "(dropout / weight decay), use early stopping, or get more training data.",
    "stalled": "Loss has stopped improving. Try a different learning rate or optimiser, "
               "check the features, or revisit the model choice.",
    "data_leakage": "Validation accuracy is suspiciously high. Check that no target-related "
                    "columns are in the features and that train/test were split before preprocessing.",
    "resource_bottleneck": "GPU is saturated and progress is slow. Reduce batch size or model size, "
                           "check the data loading pipeline, or move to a larger instance.",
    "premature_evaluation": "The run was evaluated after very few epochs without hyperparameter "
                            "tuning. Train longer and log a tuning step before drawing conclusions.",
}


def severity_score(anomaly_score, cause_probs):
    return W_ANOMALY * anomaly_score + W_CAUSE * max(cause_probs)


def severity_level(sev):
    if sev >= HIGH_AT:
        return "HIGH"
    if sev >= MEDIUM_AT:
        return "MEDIUM"
    return "LOW"


def recommend(level, cause_probs):
    """Template advice, only when severity is MEDIUM or HIGH.
    Returns (top_cause, second_cause, text); all None when LOW."""
    if level == "LOW":
        return None, None, None
    order = np.argsort(cause_probs)[::-1]
    top = CAUSES[order[0]]
    second = CAUSES[order[1]] if cause_probs[order[1]] >= SECOND_CAUSE_AT else None
    text = TEMPLATES[top] + (f" Also possible: {second}." if second else "")
    return top, second, text
