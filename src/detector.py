"""Scoring, threshold selection, and per-feature attribution.

Choosing a threshold is the part of anomaly detection that gets hand-waved.
In deployment there are no labels, so the threshold cannot come from a
precision-recall curve -- it has to come from the *training* score distribution
plus an assumption about how many alerts are tolerable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class Threshold:
    value: float
    method: str
    expected_alert_rate: float


def percentile_threshold(train_scores: np.ndarray, alert_rate: float = 0.01
                         ) -> Threshold:
    """Set the cut-off so a fixed fraction of *normal* traffic alerts.

    This is the method that survives contact with production: the alert rate is
    what the on-call team can actually absorb, and it is known in advance.
    Everything else -- precision, recall -- follows from it.
    """
    value = float(np.quantile(train_scores, 1 - alert_rate))
    return Threshold(value, f"percentile({1 - alert_rate:.4f})", alert_rate)


def sigma_threshold(train_scores: np.ndarray, n_sigma: float = 3.0) -> Threshold:
    """Mean plus ``n_sigma`` standard deviations.

    Included because it is ubiquitous, and unreliable here: reconstruction
    errors are strongly right-skewed, not Gaussian, so "3 sigma" is nowhere near
    the 99.7th percentile. Compare it against the percentile method rather than
    trusting it.
    """
    value = float(train_scores.mean() + n_sigma * train_scores.std())
    rate = float((train_scores > value).mean())
    return Threshold(value, f"mean+{n_sigma}sigma", rate)


def mad_threshold(train_scores: np.ndarray, n_mad: float = 3.0) -> Threshold:
    """Robust alternative using the median absolute deviation.

    Unlike the mean/std version this survives a contaminated training set: a few
    extreme scores barely move the median, whereas they inflate the standard
    deviation and push the threshold above the anomalies it should catch.
    """
    median = float(np.median(train_scores))
    mad = float(np.median(np.abs(train_scores - median)))
    # 1.4826 makes MAD a consistent estimator of sigma for Gaussian data.
    value = median + n_mad * 1.4826 * mad
    rate = float((train_scores > value).mean())
    return Threshold(value, f"median+{n_mad}MAD", rate)


THRESHOLD_METHODS = {
    "percentile": percentile_threshold,
    "sigma": sigma_threshold,
    "mad": mad_threshold,
}


@torch.no_grad()
def score_dataset(model, X: np.ndarray, device: torch.device,
                  batch_size: int = 512) -> np.ndarray:
    """Anomaly score per row. Higher means more anomalous."""
    model.eval()
    scores = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).float().to(device)
        scores.append(model.reconstruction_error(batch).cpu().numpy())
    return np.concatenate(scores)


@torch.no_grad()
def feature_attribution(model, X: np.ndarray, device: torch.device,
                        batch_size: int = 512) -> np.ndarray:
    """Per-feature reconstruction error, (N, F).

    Turns "row 8421 is anomalous" into "row 8421 is anomalous *because* of
    features 3 and 17", which is the difference between an alert someone can act
    on and one they will learn to ignore.
    """
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).float().to(device)
        out.append(model.reconstruction_error(batch, reduce=False).cpu().numpy())
    return np.concatenate(out)


def explain_row(attribution: np.ndarray, feature_names: list[str],
                top_k: int = 5) -> list[tuple[str, float]]:
    """The features contributing most to one row's score."""
    order = np.argsort(-attribution)[:top_k]
    return [(feature_names[i], float(attribution[i])) for i in order]


def baseline_scores(X_train: np.ndarray, X_test: np.ndarray,
                    seed: int = 0) -> Dict[str, np.ndarray]:
    """Classical detectors for comparison.

    Isolation Forest and LOF are strong, cheap and require no training loop.
    An autoencoder that does not beat them is not earning its complexity, and
    on genuinely low-dimensional data it often does not.
    """
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    results: Dict[str, np.ndarray] = {}

    forest = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
    forest.fit(X_train)
    results["isolation_forest"] = -forest.score_samples(X_test)

    lof = LocalOutlierFactor(n_neighbors=25, novelty=True)
    lof.fit(X_train)
    results["local_outlier_factor"] = -lof.score_samples(X_test)

    # Mahalanobis distance: the right linear baseline, since the background is
    # correlated Gaussian. Beating it requires genuinely nonlinear structure.
    cov = EmpiricalCovariance().fit(X_train)
    results["mahalanobis"] = cov.mahalanobis(X_test)

    return results
