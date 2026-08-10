"""Anomaly-detection metrics, including a per-anomaly-kind breakdown."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_recall_at_threshold(y_true: np.ndarray, scores: np.ndarray,
                                  threshold: float) -> Dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "alert_rate": float(y_pred.mean()),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
    }


def evaluate(y_true: np.ndarray, scores: np.ndarray,
             threshold: float | None = None) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(np.unique(y_true)) < 2:
        return {"auprc": float("nan"), "auroc": float("nan")}

    results: Dict[str, float] = {
        # auPRC is the headline: anomalies are rare by definition, and AUROC
        # stays flatteringly high even when almost every alert is a false one.
        "auprc": float(average_precision_score(y_true, scores)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "base_rate": float(y_true.mean()),
    }
    if threshold is not None:
        results.update(precision_recall_at_threshold(y_true, scores, threshold))
    return results


def evaluate_by_kind(y_true: np.ndarray, kinds: np.ndarray, scores: np.ndarray,
                     threshold: float) -> Dict[str, Dict[str, float]]:
    """Recall for each anomaly type at a fixed threshold.

    An aggregate auPRC hides the failure that matters: a detector can score 0.9
    overall while catching zero contextual anomalies, because point anomalies
    are numerous and trivial. Reporting per kind is what exposes that.
    """
    normal_mask = y_true == 0
    out: Dict[str, Dict[str, float]] = {}
    for kind in sorted(set(kinds[y_true == 1])):
        mask = normal_mask | (kinds == kind)
        subset_true = y_true[mask]
        subset_scores = scores[mask]
        detected = (scores[kinds == kind] >= threshold)
        out[str(kind)] = {
            "n": int((kinds == kind).sum()),
            "recall": float(detected.mean()) if len(detected) else float("nan"),
            "auroc": (float(roc_auc_score(subset_true, subset_scores))
                      if len(np.unique(subset_true)) > 1 else float("nan")),
        }
    return out


def format_report(results: Dict[str, float], name: str = "detector") -> str:
    order = ["auprc", "auroc", "precision", "recall", "f1", "alert_rate",
             "true_positives", "false_positives", "false_negatives", "base_rate"]
    lines = [f"{name}:"]
    for key in order:
        if key in results:
            value = results[key]
            lines.append(f"  {key:<18} "
                         + (f"{value:d}" if isinstance(value, int)
                            else f"{value:.4f}"))
    return "\n".join(lines)


def format_kind_report(by_kind: Dict[str, Dict[str, float]]) -> str:
    lines = [f"{'anomaly kind':<14}{'n':>6}{'recall':>10}{'auroc':>10}"]
    for kind, m in by_kind.items():
        lines.append(f"{kind:<14}{m['n']:>6d}{m['recall']:>10.3f}"
                     f"{m['auroc']:>10.3f}")
    return "\n".join(lines)
