"""Shared classification metrics (binary: 0 = alert, 1 = fatigued)."""
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)


def binary_metrics(y_true, y_pred, y_proba=None):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    out = {
        "n": int(len(y_true)),
        "n_alert": int((y_true == 0).sum()),
        "n_fatigued": int((y_true == 1).sum()),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        "recall_alert": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_fatigued": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_alert": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_fatigued": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": np.nan,
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    return out
