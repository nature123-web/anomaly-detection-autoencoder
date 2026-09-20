"""Score rows and print explained alerts.

    python -m src.predict --checkpoint runs/base/best.pt --explain
    python -m src.predict --checkpoint runs/base/best.pt --csv data.csv --out alerts.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import StandardScaler, contaminated_split, make_anomaly_dataset
from .detector import explain_row, feature_attribution, score_dataset
from .metrics import evaluate, format_report
from .model import build_model


def load_detector(checkpoint: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg, ckpt["n_features"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["scaler_mean"])
    scaler.scale_ = np.array(ckpt["scaler_scale"])
    return model, cfg, scaler, ckpt["threshold"], ckpt["feature_names"]


def render_alert(index: int, score: float, threshold: float,
                 attribution: np.ndarray, feature_names: list[str],
                 top_k: int = 5) -> str:
    top = explain_row(attribution, feature_names, top_k)
    largest = max(value for _, value in top) or 1.0
    lines = [f"row {index}   score {score:.4f}   (threshold {threshold:.4f})"]
    for name, value in top:
        # ASCII rather than a Unicode block character: this string goes
        # straight to print(), and a console stuck on a legacy codepage
        # (cp1252 is still the Windows default outside Windows Terminal)
        # raises UnicodeEncodeError on U+2588 instead of printing the alert.
        bar = "#" * int(round(28 * value / largest))
        lines.append(f"  {name:<14} {value:8.4f}  {bar}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", default=None,
                        help="Rows to score; defaults to a fresh synthetic sample.")
    parser.add_argument("--explain", action="store_true",
                        help="Show per-feature attribution for the top alerts.")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg, scaler, threshold, feature_names = load_detector(
        args.checkpoint, device
    )

    y_true = None
    if args.csv:
        raw = np.loadtxt(args.csv, delimiter=",", skiprows=1, dtype=np.float32)
        X = raw[:, : len(feature_names)]
    else:
        dataset = make_anomaly_dataset(
            cfg["data"]["n_samples"], cfg["data"]["n_features"],
            cfg["data"]["anomaly_rate"], cfg["seed"],
        )
        _, X, y_true, _, _ = contaminated_split(
            dataset, cfg["data"]["train_fraction"],
            cfg["data"]["contamination"], cfg["seed"],
        )
        print("scoring the held-out split (pass --csv for your own data)")

    X_scaled = scaler.transform(X)
    scores = score_dataset(model, X_scaled, device)
    flagged = scores >= threshold

    print(f"\n{len(X)} rows scored, {int(flagged.sum())} above threshold "
          f"({flagged.mean():.2%} alert rate)")

    if y_true is not None:
        print("\n" + format_report(
            evaluate(y_true, scores, threshold), cfg["model"]["arch"]
        ))

    if args.explain:
        attribution = feature_attribution(model, X_scaled, device)
        order = np.argsort(-scores)[: args.top]
        print(f"\ntop {len(order)} alerts:\n")
        for idx in order:
            print(render_alert(int(idx), float(scores[idx]), threshold,
                               attribution[idx], feature_names))
            if y_true is not None:
                print(f"  actual: {'ANOMALY' if y_true[idx] else 'normal'}")
            print()

    if args.out:
        header = "row,score,flagged" + (",label" if y_true is not None else "")
        rows = []
        for i, (s, f) in enumerate(zip(scores, flagged)):
            row = f"{i},{s:.6f},{int(f)}"
            if y_true is not None:
                row += f",{int(y_true[i])}"
            rows.append(row)
        Path(args.out).write_text(header + "\n" + "\n".join(rows) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
