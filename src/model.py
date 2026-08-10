"""Autoencoder and variational autoencoder detectors.

The detection principle: train to reconstruct normal data, then score by
reconstruction error. Anything the model has never learned to represent
reconstructs badly.

The bottleneck is what makes this work. An autoencoder wide enough to be
lossless learns the identity function and reconstructs anomalies perfectly --
error becomes zero everywhere and the detector is dead. ``latent_dim`` must be
small enough to force the model to spend its capacity on the structure that is
actually common in the data.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes: Sequence[int], dropout: float, final_activation: bool
         ) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        last = i == len(sizes) - 2
        if not last or final_activation:
            layers += [nn.BatchNorm1d(b), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """Symmetric dense autoencoder."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: Sequence[int] = (64, 32),
        latent_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        encoder_sizes = [n_features, *hidden_sizes, latent_dim]
        decoder_sizes = [latent_dim, *reversed(list(hidden_sizes)), n_features]
        self.encoder = _mlp(encoder_sizes, dropout, final_activation=False)
        self.decoder = _mlp(decoder_sizes, dropout, final_activation=False)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor, reduce: bool = True
                             ) -> torch.Tensor:
        """Squared error per sample, or per feature when ``reduce`` is False.

        The unreduced form is what makes a detection explainable: it says *which*
        features the model could not reproduce, which is the first thing an
        operator asks after an alert fires.
        """
        error = (self(x) - x) ** 2
        return error.mean(dim=1) if reduce else error

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(x), x)


class VariationalAutoencoder(nn.Module):
    """VAE scored by the negative ELBO.

    Preferred over a plain autoencoder when a *calibrated* score is wanted: the
    ELBO is a bound on log-likelihood, so its scale carries meaning across
    datasets in a way that raw reconstruction error does not. The cost is the KL
    term, which if weighted too heavily causes posterior collapse -- the latent
    is ignored, every input decodes to the dataset mean, and the detector reports
    the same score for everything. ``beta`` is exposed for that reason.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: Sequence[int] = (64, 32),
        latent_dim: int = 8,
        dropout: float = 0.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = _mlp([n_features, *hidden_sizes], dropout,
                            final_activation=True)
        self.to_mu = nn.Linear(hidden_sizes[-1], latent_dim)
        self.to_logvar = nn.Linear(hidden_sizes[-1], latent_dim)
        self.decoder = _mlp([latent_dim, *reversed(list(hidden_sizes)), n_features],
                            dropout, final_activation=False)
        self.latent_dim = latent_dim
        self.beta = beta

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        # Clamped for numerical safety: an unbounded logvar makes exp() overflow
        # and produces nan losses a few hundred steps into training.
        return self.to_mu(h), self.to_logvar(h).clamp(-10.0, 10.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor
                       ) -> torch.Tensor:
        if not self.training:
            # Deterministic at eval time, or the same input scores differently
            # on every call and the threshold means nothing.
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        return self.decoder(self.reparameterize(mu, logvar))

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        reconstruction = self.decoder(self.reparameterize(mu, logvar))
        recon_loss = F.mse_loss(reconstruction, x, reduction="none").sum(1).mean()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        return recon_loss + self.beta * kl

    def reconstruction_error(self, x: torch.Tensor, reduce: bool = True
                             ) -> torch.Tensor:
        mu, logvar = self.encode(x)
        reconstruction = self.decoder(mu)
        error = (reconstruction - x) ** 2
        if not reduce:
            return error
        # Negative ELBO per sample: reconstruction plus the KL to the prior.
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
        return error.sum(1) + self.beta * kl


def build_model(cfg: dict, n_features: int) -> nn.Module:
    m = cfg["model"]
    if m["arch"] == "autoencoder":
        return Autoencoder(n_features, m["hidden_sizes"], m["latent_dim"],
                           m["dropout"])
    if m["arch"] == "vae":
        return VariationalAutoencoder(n_features, m["hidden_sizes"],
                                      m["latent_dim"], m["dropout"], m["beta"])
    raise ValueError(f"unknown arch '{m['arch']}'; choose autoencoder or vae")
