"""Tests for the generator, models, thresholding, and metrics."""

import numpy as np
import pytest
import torch

from src.data import (
    ANOMALY_KINDS,
    StandardScaler,
    contaminated_split,
    make_anomaly_dataset,
)
from src.detector import (
    baseline_scores,
    explain_row,
    feature_attribution,
    mad_threshold,
    percentile_threshold,
    score_dataset,
    sigma_threshold,
)
from src.metrics import evaluate, evaluate_by_kind, precision_recall_at_threshold
from src.model import Autoencoder, VariationalAutoencoder, build_model

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def test_dataset_shape_and_rate():
    ds = make_anomaly_dataset(2000, n_features=12, anomaly_rate=0.05, seed=0)
    assert ds.X.shape == (2000, 12)
    assert ds.y.mean() == pytest.approx(0.05, abs=0.01)
    assert len(ds.feature_names) == 12


def test_all_anomaly_kinds_are_present():
    ds = make_anomaly_dataset(4000, anomaly_rate=0.1, seed=0)
    present = set(ds.kinds[ds.y == 1])
    assert present == set(ANOMALY_KINDS)


def test_normal_rows_are_labelled_normal():
    ds = make_anomaly_dataset(1000, seed=0)
    assert set(ds.kinds[ds.y == 0]) == {"normal"}


def test_background_features_are_correlated():
    """Independent features would make contextual anomalies impossible."""
    ds = make_anomaly_dataset(3000, n_features=10, seed=0)
    normal = ds.X[ds.y == 0]
    corr = np.corrcoef(normal.T)
    off_diagonal = corr[~np.eye(10, dtype=bool)]
    assert np.abs(off_diagonal).max() > 0.3


def test_contextual_anomalies_have_normal_marginals():
    """The defining property: each value is ordinary, the combination is not."""
    ds = make_anomaly_dataset(6000, n_features=10, anomaly_rate=0.2, seed=1)
    normal = ds.X[ds.y == 0]
    contextual = ds.X[ds.kinds == "contextual"]

    for j in range(10):
        lo, hi = np.quantile(normal[:, j], [0.001, 0.999])
        inside = ((contextual[:, j] >= lo) & (contextual[:, j] <= hi)).mean()
        assert inside > 0.9, f"feature {j}: contextual values are out of range"


def test_point_anomalies_are_extreme_in_one_feature():
    ds = make_anomaly_dataset(6000, n_features=10, anomaly_rate=0.2, seed=1)
    normal = ds.X[ds.y == 0]
    point = ds.X[ds.kinds == "point"]
    z = np.abs((point - normal.mean(0)) / normal.std(0))
    assert (z.max(axis=1) > 3).mean() > 0.8


def test_dataset_is_deterministic():
    a = make_anomaly_dataset(500, seed=3)
    b = make_anomaly_dataset(500, seed=3)
    assert np.allclose(a.X, b.X) and np.array_equal(a.y, b.y)


# --------------------------------------------------------------------------- #
# Splitting and contamination
# --------------------------------------------------------------------------- #

def test_clean_split_has_no_anomalies_in_training():
    ds = make_anomaly_dataset(2000, seed=0)
    _, _, _, _, train_labels = contaminated_split(ds, 0.6, contamination=0.0)
    assert train_labels.sum() == 0


def test_contamination_puts_the_requested_fraction_into_training():
    ds = make_anomaly_dataset(4000, anomaly_rate=0.2, seed=0)
    for target in (0.02, 0.05, 0.10):
        _, _, _, _, train_labels = contaminated_split(
            ds, 0.6, contamination=target
        )
        assert train_labels.mean() == pytest.approx(target, abs=0.015)


def test_train_and_test_do_not_overlap():
    ds = make_anomaly_dataset(1000, seed=0)
    X_train, X_test, _, _, _ = contaminated_split(ds, 0.6, 0.0)
    train_rows = {row.tobytes() for row in X_train}
    assert not any(row.tobytes() in train_rows for row in X_test)


def test_scaler_standardises_and_survives_constant_features():
    X = np.column_stack([np.ones(100), np.random.default_rng(0).normal(5, 2, 100)])
    Z = StandardScaler().fit_transform(X.astype(np.float32))
    assert np.isfinite(Z).all()
    assert Z[:, 1].mean() == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def test_autoencoder_reconstructs_its_input_shape():
    model = Autoencoder(10, hidden_sizes=(16, 8), latent_dim=4).eval()
    x = torch.randn(5, 10)
    assert model(x).shape == (5, 10)


def test_reconstruction_error_shapes():
    model = Autoencoder(10, hidden_sizes=(16,), latent_dim=3).eval()
    x = torch.randn(7, 10)
    assert model.reconstruction_error(x).shape == (7,)
    assert model.reconstruction_error(x, reduce=False).shape == (7, 10)


def test_reconstruction_error_is_nonnegative():
    model = Autoencoder(8, hidden_sizes=(16,), latent_dim=3).eval()
    assert (model.reconstruction_error(torch.randn(20, 8)) >= 0).all()


def test_bottleneck_is_narrower_than_the_input():
    """A latent as wide as the input learns the identity and detects nothing."""
    model = Autoencoder(20, hidden_sizes=(16, 8), latent_dim=4)
    assert model.latent_dim < 20


def test_vae_is_deterministic_in_eval_mode():
    """A stochastic score would make any fixed threshold meaningless."""
    model = VariationalAutoencoder(10, hidden_sizes=(16,), latent_dim=4).eval()
    x = torch.randn(6, 10)
    a = model.reconstruction_error(x)
    b = model.reconstruction_error(x)
    assert torch.allclose(a, b)


def test_vae_is_stochastic_in_training_mode():
    torch.manual_seed(0)
    model = VariationalAutoencoder(10, hidden_sizes=(16,), latent_dim=4).train()
    x = torch.randn(8, 10)
    assert not torch.allclose(model(x), model(x))


def test_vae_logvar_is_clamped():
    """Unbounded logvar overflows exp() and produces nan losses."""
    model = VariationalAutoencoder(6, hidden_sizes=(8,), latent_dim=2).eval()
    with torch.no_grad():
        model.to_logvar.bias.fill_(500.0)
        _, logvar = model.encode(torch.randn(3, 6))
    assert logvar.max() <= 10.0
    assert torch.isfinite(model.reconstruction_error(torch.randn(3, 6))).all()


def test_losses_are_finite_and_differentiable():
    for model in (Autoencoder(8, (16,), 3), VariationalAutoencoder(8, (16,), 3)):
        loss = model.loss(torch.randn(16, 8))
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())


def test_build_model_dispatch_and_error():
    cfg = {"model": {"arch": "autoencoder", "hidden_sizes": [16, 8],
                     "latent_dim": 4, "dropout": 0.0, "beta": 1.0}}
    assert isinstance(build_model(cfg, 10), Autoencoder)
    cfg["model"]["arch"] = "vae"
    assert isinstance(build_model(cfg, 10), VariationalAutoencoder)
    cfg["model"]["arch"] = "nope"
    with pytest.raises(ValueError, match="unknown arch"):
        build_model(cfg, 10)


def test_trained_autoencoder_separates_anomalies():
    """End-to-end: a briefly trained model must rank anomalies above normals."""
    torch.manual_seed(0)
    np.random.seed(0)
    ds = make_anomaly_dataset(3000, n_features=15, anomaly_rate=0.08, seed=0)
    X_train, X_test, y_test, _, _ = contaminated_split(ds, 0.6, 0.0, seed=0)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    model = Autoencoder(15, hidden_sizes=(32, 16), latent_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    data = torch.from_numpy(X_train_s).float()
    model.train()
    for _ in range(150):
        loss = model.loss(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scores = score_dataset(model, X_test_s, DEVICE)
    assert evaluate(y_test, scores)["auroc"] > 0.75


# --------------------------------------------------------------------------- #
# Thresholding
# --------------------------------------------------------------------------- #

def test_percentile_threshold_delivers_the_requested_alert_rate():
    scores = np.random.default_rng(0).gamma(2, 1, 10_000)
    threshold = percentile_threshold(scores, alert_rate=0.01)
    assert (scores > threshold.value).mean() == pytest.approx(0.01, abs=0.003)


def test_sigma_threshold_misses_its_nominal_rate_on_skewed_scores():
    """Reconstruction errors are right-skewed, so '3 sigma' is not 99.7%."""
    scores = np.random.default_rng(0).gamma(1.5, 1.0, 20_000)
    rate = sigma_threshold(scores, 3.0).expected_alert_rate
    assert abs(rate - 0.003) > 0.002


def test_mad_threshold_is_robust_to_contamination():
    """A few extreme scores must not drag the threshold above the anomalies."""
    rng = np.random.default_rng(0)
    clean = rng.gamma(2, 1, 5000)
    contaminated = np.concatenate([clean, rng.uniform(50, 100, 100)])

    mad_shift = abs(mad_threshold(contaminated).value - mad_threshold(clean).value)
    sigma_shift = abs(
        sigma_threshold(contaminated).value - sigma_threshold(clean).value
    )
    assert mad_shift < sigma_shift


def test_threshold_methods_report_their_own_alert_rate():
    scores = np.random.default_rng(0).gamma(2, 1, 5000)
    for threshold in (percentile_threshold(scores, 0.02),
                      sigma_threshold(scores), mad_threshold(scores)):
        assert 0.0 <= threshold.expected_alert_rate <= 1.0
        assert threshold.method


# --------------------------------------------------------------------------- #
# Attribution and baselines
# --------------------------------------------------------------------------- #

def test_feature_attribution_shape_and_consistency():
    model = Autoencoder(12, hidden_sizes=(16,), latent_dim=4).eval()
    X = np.random.default_rng(0).normal(size=(20, 12)).astype(np.float32)
    attribution = feature_attribution(model, X, DEVICE)
    assert attribution.shape == (20, 12)
    # The row score is the mean of its per-feature errors.
    assert np.allclose(attribution.mean(axis=1), score_dataset(model, X, DEVICE),
                       atol=1e-5)


def test_explain_row_returns_the_largest_contributors():
    attribution = np.array([0.1, 5.0, 0.2, 3.0])
    names = ["a", "b", "c", "d"]
    top = explain_row(attribution, names, top_k=2)
    assert [name for name, _ in top] == ["b", "d"]


def test_render_alert_is_printable_on_a_legacy_console_codepage():
    """render_alert used to build its bar with the Unicode block character
    '█', which crashes with UnicodeEncodeError the moment it reaches
    print() on a console still using cp1252 -- the Windows default outside
    Windows Terminal. The whole point of an alert is that it gets printed,
    so the string this returns must survive that encoding.
    """
    from src.predict import render_alert

    attribution = np.array([0.1, 5.0, 0.2, 3.0])
    names = ["a", "b", "c", "d"]
    text = render_alert(0, 4.2, 3.5, attribution, names, top_k=4)
    text.encode("cp1252")  # raises UnicodeEncodeError on a regression


def test_baselines_return_a_score_per_row():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(300, 8))
    X_test = rng.normal(size=(100, 8))
    scores = baseline_scores(X_train, X_test, seed=0)
    assert set(scores) == {"isolation_forest", "local_outlier_factor",
                           "mahalanobis"}
    for values in scores.values():
        assert values.shape == (100,) and np.isfinite(values).all()


def test_mahalanobis_flags_a_planted_outlier():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(500, 5))
    X_test = np.vstack([rng.normal(size=(20, 5)), np.full((1, 5), 12.0)])
    scores = baseline_scores(X_train, X_test, seed=0)["mahalanobis"]
    assert scores.argmax() == len(X_test) - 1


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_precision_recall_at_threshold_counts_correctly():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1])
    m = precision_recall_at_threshold(y, scores, 0.5)
    assert m["true_positives"] == 1
    assert m["false_positives"] == 1
    assert m["false_negatives"] == 1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)


def test_perfect_detector_scores_one():
    y = np.array([0] * 90 + [1] * 10)
    scores = y.astype(float)
    results = evaluate(y, scores, threshold=0.5)
    assert results["auprc"] == 1.0 and results["recall"] == 1.0


def test_auprc_of_a_random_detector_tracks_the_base_rate():
    rng = np.random.default_rng(0)
    y = (rng.random(10_000) < 0.05).astype(int)
    assert evaluate(y, rng.random(10_000))["auprc"] == pytest.approx(0.05, abs=0.02)


def test_evaluate_by_kind_separates_the_types():
    y = np.array([0, 0, 1, 1])
    kinds = np.array(["normal", "normal", "point", "contextual"], dtype=object)
    scores = np.array([0.1, 0.2, 0.9, 0.15])
    by_kind = evaluate_by_kind(y, kinds, scores, threshold=0.5)

    assert by_kind["point"]["recall"] == 1.0
    # The contextual anomaly scores below threshold -- exactly the failure an
    # aggregate metric would hide.
    assert by_kind["contextual"]["recall"] == 0.0


def test_evaluate_by_kind_counts_every_anomaly():
    ds = make_anomaly_dataset(2000, anomaly_rate=0.1, seed=0)
    scores = np.random.default_rng(0).random(len(ds.y))
    by_kind = evaluate_by_kind(ds.y, ds.kinds, scores, threshold=0.5)
    assert sum(m["n"] for m in by_kind.values()) == int(ds.y.sum())
