"""Multivariate data with injected anomalies of several distinct kinds.

Real anomaly detection fails in characteristic ways depending on *what kind* of
anomaly is present, and a generator that only produces one kind will make a
detector look better than it is. Four types are produced here:

``point``
    A single feature takes an extreme value. Easy; almost anything catches these.
``contextual``
    Every feature is individually within its normal range, but the *combination*
    is impossible. Only a model of the joint distribution catches these, which is
    precisely the case for using an autoencoder over per-feature thresholds.
``collective``
    A cluster of mildly unusual points, none extreme alone.
``subspace``
    Anomalous only in a low-variance direction. PCA-style methods that keep the
    top components discard exactly this signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

ANOMALY_KINDS = ("point", "contextual", "collective", "subspace")


@dataclass
class AnomalyDataset:
    X: np.ndarray            # (N, F)
    y: np.ndarray            # (N,) 1 = anomaly
    kinds: np.ndarray        # (N,) object; "normal" or an entry of ANOMALY_KINDS
    feature_names: list[str]


def make_anomaly_dataset(
    n_samples: int = 10_000,
    n_features: int = 20,
    anomaly_rate: float = 0.05,
    seed: int = 0,
) -> AnomalyDataset:
    """Correlated normal data with a mix of anomaly types injected."""
    rng = np.random.default_rng(seed)
    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Correlated background: a low-rank factor structure plus noise, which is
    # what real sensor and transaction data looks like. Independent features
    # would make contextual anomalies impossible by construction.
    n_factors = max(2, n_features // 4)
    loadings = rng.normal(0, 1, (n_factors, n_features))
    factors = rng.normal(0, 1, (n_normal, n_factors))
    X_normal = factors @ loadings + rng.normal(0, 0.35, (n_normal, n_features))

    # Direction of least variance, used for the subspace anomalies.
    _, _, vt = np.linalg.svd(X_normal - X_normal.mean(0), full_matrices=False)
    weak_direction = vt[-1]

    anomalies, kinds = [], []
    per_kind = max(1, n_anomalies // len(ANOMALY_KINDS))

    for kind in ANOMALY_KINDS:
        count = per_kind if kind != ANOMALY_KINDS[-1] else \
            n_anomalies - per_kind * (len(ANOMALY_KINDS) - 1)
        if count <= 0:
            continue
        base_factors = rng.normal(0, 1, (count, n_factors))
        base = base_factors @ loadings + rng.normal(0, 0.35, (count, n_features))

        if kind == "point":
            for row in base:
                idx = rng.integers(n_features)
                row[idx] += rng.choice([-1, 1]) * rng.uniform(6, 10)

        elif kind == "contextual":
            # Keep the marginals but destroy the correlation: shuffle each
            # feature independently across the batch. Every value is one that
            # genuinely occurs; only the combination is impossible.
            for j in range(n_features):
                base[:, j] = rng.permutation(
                    X_normal[rng.integers(0, n_normal, count), j]
                )

        elif kind == "collective":
            centre = rng.normal(0, 1, n_factors) @ loadings
            offset = rng.normal(0, 1, n_features)
            offset /= np.linalg.norm(offset)
            base = centre + offset * 2.5 + rng.normal(0, 0.3, (count, n_features))

        elif kind == "subspace":
            magnitude = rng.uniform(3, 5, (count, 1))
            base = base + magnitude * weak_direction

        anomalies.append(base)
        kinds.extend([kind] * count)

    X_anom = np.vstack(anomalies)
    X = np.vstack([X_normal, X_anom]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(len(X_anom))]).astype(int)
    kind_array = np.array(["normal"] * n_normal + kinds, dtype=object)

    order = rng.permutation(len(X))
    return AnomalyDataset(
        X=X[order], y=y[order], kinds=kind_array[order],
        feature_names=[f"feature_{i:02d}" for i in range(n_features)],
    )


def contaminated_split(
    dataset: AnomalyDataset,
    train_fraction: float = 0.6,
    contamination: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a training set with a controlled fraction of anomalies in it.

    ``contamination`` is the single most important knob in this repo. An
    autoencoder trained on clean normal data learns to reconstruct normal data
    and fails loudly on anomalies -- which is the whole mechanism. Train it on
    data that already contains anomalies and it learns to reconstruct those too,
    quietly destroying the signal it depends on. Being able to sweep this is what
    turns "it works on my clean benchmark" into a defensible claim.

    Returns ``(X_train, X_test, y_test, kinds_test, train_labels)``.
    """
    rng = np.random.default_rng(seed)
    normal_idx = np.flatnonzero(dataset.y == 0)
    anomaly_idx = np.flatnonzero(dataset.y == 1)
    rng.shuffle(normal_idx)
    rng.shuffle(anomaly_idx)

    n_train_normal = int(len(normal_idx) * train_fraction)
    train_normal = normal_idx[:n_train_normal]

    n_train_anom = int(n_train_normal * contamination / max(1e-9, 1 - contamination))
    n_train_anom = min(n_train_anom, len(anomaly_idx) // 2)
    train_anom = anomaly_idx[:n_train_anom]

    train_idx = np.concatenate([train_normal, train_anom])
    rng.shuffle(train_idx)
    test_idx = np.concatenate([normal_idx[n_train_normal:], anomaly_idx[n_train_anom:]])
    rng.shuffle(test_idx)

    return (
        dataset.X[train_idx], dataset.X[test_idx], dataset.y[test_idx],
        dataset.kinds[test_idx], dataset.y[train_idx],
    )


class StandardScaler:
    """Fitted on training data only; anything else leaks the test distribution."""

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
