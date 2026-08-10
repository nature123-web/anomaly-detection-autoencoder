# Anomaly Detection with Autoencoders

Unsupervised anomaly detection on multivariate data using autoencoders and VAEs,
benchmarked against Isolation Forest, LOF and Mahalanobis distance.

Two questions get honest answers here, and both are usually skipped:

1. **How do you pick a threshold with no labels?**
2. **What happens when your "normal" training data isn't clean?**

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train --config configs/base.yaml
python -m src.train --config configs/base.yaml --arch vae
python -m src.train --config configs/base.yaml --contamination-sweep
python -m src.predict --checkpoint runs/base/best.pt --explain
```

## The mechanism, and how it breaks

Train to reconstruct normal data; score by reconstruction error. Anything the
model never learned to represent reconstructs badly.

This depends entirely on the model **not** being able to reconstruct everything.
Two ways it fails:

- **Bottleneck too wide.** A latent dimension near the input dimension lets the
  network learn the identity function. Error goes to zero everywhere and the
  detector is dead while training loss looks excellent.
- **Contaminated training data.** If anomalies are already in the training set,
  the autoencoder learns to reconstruct those too — destroying exactly the
  signal it relies on.

The second is the realistic one, since "clean normal data" is an assumption
almost nobody can actually verify. `--contamination-sweep` measures it, retraining
at each level and printing the degradation curve:

```bash
python -m src.train --config configs/base.yaml --contamination-sweep
```

```
contamination   0.0%   auPRC ...   auROC ...   recall ...
contamination   1.0%   auPRC ...   auROC ...   recall ...
contamination   2.0%   ...
```

Results are written to `contamination_sweep.json`. Knowing that curve for your
own data is the difference between a detector you can trust and one that silently
degrades.

## Four kinds of anomaly

A generator that only makes one kind of anomaly will flatter any detector. This
one makes four, and results are reported **per kind**:

| Kind | What it is | Who catches it |
| --- | --- | --- |
| `point` | one feature takes an extreme value | almost anything, including a z-score |
| `contextual` | every value is normal, the *combination* is impossible | needs a joint model — this is the case for autoencoders |
| `collective` | a cluster of mildly unusual points | density-based methods |
| `subspace` | anomalous only along a low-variance direction | *missed* by PCA-style methods that keep top components |

An aggregate auPRC of 0.9 can hide catching zero contextual anomalies, because
point anomalies are numerous and trivial. The per-kind table exposes that — from
an actual run on the default config:

```
anomaly kind       n    recall     auroc
collective       150     0.800     0.987
contextual       150     1.000     1.000
point            150     1.000     1.000
subspace         150     1.000     1.000
```

Collective anomalies are the weak spot here: a tight cluster of mildly unusual
points is, from the autoencoder's perspective, just another mode it can learn.

## Choosing a threshold without labels

In production there is no precision-recall curve to optimise. The threshold has
to come from the training score distribution plus a decision about tolerable
alert volume. Three rules are implemented and compared on every run:

- **`percentile`** — set the cut so a fixed fraction of normal traffic alerts.
  This is the one that survives deployment: the alert rate is what the on-call
  team can absorb, and it is known in advance.
- **`sigma`** — mean + 3σ. Ubiquitous and wrong here: reconstruction errors are
  strongly right-skewed, so "3 sigma" is nowhere near the 99.7th percentile. A
  test asserts this deviation rather than letting it pass unnoticed.
- **`mad`** — median + 3·1.4826·MAD. Robust; a handful of extreme scores barely
  move it, whereas they inflate σ and push the threshold above the anomalies it
  should catch. Tested directly.

## Explainable alerts

`reconstruction_error(x, reduce=False)` gives per-feature error, so an alert says
*which* features could not be reproduced:

```
row 8421   score 4.8213   (threshold 0.9042)
  feature_03    2.9104  ████████████████████████████
  feature_17    1.2277  ███████████
  feature_08    0.3391  ███
```

That is the difference between an alert someone acts on and one they mute.

## Baselines

Isolation Forest, Local Outlier Factor, and Mahalanobis distance run on every
job. Mahalanobis is the important one — the background here is correlated
Gaussian, so beating it requires genuinely nonlinear structure.

**On this synthetic dataset, it wins.** A representative run:

```
autoencoder            auPRC 0.9833  auROC 0.9967
isolation_forest       auPRC 0.1751  auROC 0.7113
local_outlier_factor   auPRC 0.9319  auROC 0.9893
mahalanobis            auPRC 0.9994  auROC 0.9999

autoencoder vs best baseline auPRC: -0.0161
the autoencoder is not beating a classical detector here
```

That is the correct answer, not a bug: when the normal manifold really is a
correlated Gaussian, Mahalanobis distance is close to optimal and a neural
network can only approximate it. The autoencoder earns its keep when the normal
manifold is nonlinear — which this generator, by design, only partly is. The run
prints the comparison every time so the question is never left open.

## Configuration

```yaml
data:
  n_features: 20
  anomaly_rate: 0.05
  contamination: 0.0            # anomalies in the training set
  sweep_values: [0.0, 0.01, 0.02, 0.05, 0.10]
model:
  arch: autoencoder             # autoencoder | vae
  hidden_sizes: [64, 32]
  latent_dim: 8                 # must be << n_features
  beta: 1.0                     # VAE KL weight; too high causes posterior collapse
detect:
  alert_rate: 0.01              # fraction of normal traffic allowed to alert
```

## Layout

```
src/
  data.py       generator with four anomaly types, contamination control, scaling
  model.py      Autoencoder, VariationalAutoencoder
  detector.py   scoring, threshold rules, feature attribution, baselines
  metrics.py    auPRC, per-kind breakdown, precision/recall at threshold
  train.py      training loop, contamination sweep, baseline comparison
  predict.py    scoring and explained alerts
tests/          pytest suite
```

## License

MIT
