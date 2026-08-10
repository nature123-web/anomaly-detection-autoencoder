"""Train an anomaly detector and compare it against classical baselines.

    python -m src.train --config configs/base.yaml
    python -m src.train --config configs/base.yaml --arch vae
    python -m src.train --config configs/base.yaml --contamination-sweep
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import StandardScaler, contaminated_split, make_anomaly_dataset
from .detector import (
    THRESHOLD_METHODS,
    baseline_scores,
    percentile_threshold,
    score_dataset,
)
from .metrics import evaluate, evaluate_by_kind, format_kind_report, format_report
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, X_train, cfg, device, verbose: bool = True):
    """Fit the autoencoder on the training rows."""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float()),
        batch_size=cfg["train"]["batch_size"], shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    best_loss, patience, best_state = float("inf"), 0, None
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total, seen = 0.0, 0
        iterator = tqdm(loader, desc=f"epoch {epoch}", leave=False) if verbose \
            else loader
        for (batch,) in iterator:
            batch = batch.to(device)
            loss = model.loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * batch.size(0)
            seen += batch.size(0)
        scheduler.step()

        epoch_loss = total / max(1, seen)
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"epoch {epoch:3d}  loss {epoch_loss:.6f}")

        # Early stopping on *training* loss: there is no labelled validation set
        # in an unsupervised setting, and inventing one would defeat the point.
        if epoch_loss < best_loss - 1e-6:
            best_loss, patience = epoch_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                if verbose:
                    print(f"early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_once(cfg, contamination: float, device, verbose: bool = True):
    """One full train/evaluate cycle at a given training contamination."""
    dataset = make_anomaly_dataset(
        cfg["data"]["n_samples"], cfg["data"]["n_features"],
        cfg["data"]["anomaly_rate"], cfg["seed"],
    )
    X_train, X_test, y_test, kinds_test, train_labels = contaminated_split(
        dataset, cfg["data"]["train_fraction"], contamination, cfg["seed"]
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    model = build_model(cfg, X_train.shape[1]).to(device)
    train_model(model, X_train_s, cfg, device, verbose)

    train_scores = score_dataset(model, X_train_s, device)
    test_scores = score_dataset(model, X_test_s, device)
    threshold = percentile_threshold(train_scores, cfg["detect"]["alert_rate"])

    results = evaluate(y_test, test_scores, threshold.value)
    by_kind = evaluate_by_kind(y_test, kinds_test, test_scores, threshold.value)
    return {
        "model": model, "scaler": scaler, "threshold": threshold,
        "results": results, "by_kind": by_kind,
        "train_scores": train_scores, "test_scores": test_scores,
        "X_train_s": X_train_s, "X_test_s": X_test_s,
        "y_test": y_test, "kinds_test": kinds_test,
        "actual_contamination": float(np.mean(train_labels)),
        "dataset": dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--arch", default=None, choices=["autoencoder", "vae"])
    parser.add_argument("--contamination", type=float, default=None)
    parser.add_argument("--contamination-sweep", action="store_true",
                        help="Show how training contamination degrades detection.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.arch:
        cfg["model"]["arch"] = args.arch
    if args.contamination is not None:
        cfg["data"]["contamination"] = args.contamination
    if args.out_dir:
        cfg["out_dir"] = args.out_dir

    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  arch={cfg['model']['arch']}  out_dir={out_dir}")

    if args.contamination_sweep:
        print("\ncontamination sweep -- how much anomalous data in training "
              "destroys the detector\n")
        sweep = []
        for contamination in cfg["data"]["sweep_values"]:
            outcome = run_once(cfg, contamination, device, verbose=False)
            sweep.append({"contamination": contamination,
                          "auprc": outcome["results"]["auprc"],
                          "auroc": outcome["results"]["auroc"],
                          "recall": outcome["results"].get("recall", float("nan"))})
            print(f"  contamination {contamination:5.1%}  "
                  f"auPRC {outcome['results']['auprc']:.4f}  "
                  f"auROC {outcome['results']['auroc']:.4f}  "
                  f"recall {outcome['results'].get('recall', float('nan')):.4f}")
        (out_dir / "contamination_sweep.json").write_text(
            json.dumps(sweep, indent=2)
        )
        clean, dirty = sweep[0]["auprc"], sweep[-1]["auprc"]
        print(f"\nauPRC fell {100 * (1 - dirty / max(clean, 1e-9)):.1f}% "
              f"from clean training to "
              f"{cfg['data']['sweep_values'][-1]:.0%} contamination")
        return

    contamination = cfg["data"]["contamination"]
    outcome = run_once(cfg, contamination, device)
    print(f"\ntraining contamination: {outcome['actual_contamination']:.3%}")
    print(f"threshold: {outcome['threshold'].value:.5f} "
          f"({outcome['threshold'].method})")
    print("\n" + format_report(outcome["results"], cfg["model"]["arch"]))
    print("\nper anomaly kind:")
    print(format_kind_report(outcome["by_kind"]))

    # How much does the choice of threshold rule matter?
    print("\nthreshold methods on the same scores:")
    for name, method in THRESHOLD_METHODS.items():
        kwargs = {"alert_rate": cfg["detect"]["alert_rate"]} if name == "percentile" else {}
        threshold = method(outcome["train_scores"], **kwargs)
        metrics = evaluate(outcome["y_test"], outcome["test_scores"],
                           threshold.value)
        print(f"  {threshold.method:<22} alert_rate {metrics['alert_rate']:.4f}  "
              f"precision {metrics['precision']:.3f}  "
              f"recall {metrics['recall']:.3f}")

    print("\nclassical baselines:")
    baselines = baseline_scores(outcome["X_train_s"], outcome["X_test_s"],
                                cfg["seed"])
    baseline_results = {}
    for name, scores in baselines.items():
        metrics = evaluate(outcome["y_test"], scores)
        baseline_results[name] = metrics
        print(f"  {name:<22} auPRC {metrics['auprc']:.4f}  "
              f"auROC {metrics['auroc']:.4f}")

    best_baseline = max(baseline_results.values(), key=lambda m: m["auprc"])
    delta = outcome["results"]["auprc"] - best_baseline["auprc"]
    print(f"\n{cfg['model']['arch']} vs best baseline auPRC: {delta:+.4f}")
    if delta <= 0:
        print("the autoencoder is not beating a classical detector here")

    torch.save({"model": outcome["model"].state_dict(), "config": cfg,
                "n_features": cfg["data"]["n_features"],
                "threshold": outcome["threshold"].value,
                "scaler_mean": outcome["scaler"].mean_.tolist(),
                "scaler_scale": outcome["scaler"].scale_.tolist(),
                "feature_names": outcome["dataset"].feature_names},
               out_dir / "best.pt")
    (out_dir / "results.json").write_text(json.dumps(
        {"model": outcome["results"], "by_kind": outcome["by_kind"],
         "baselines": baseline_results}, indent=2, default=float,
    ))
    print(f"\nsaved {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
