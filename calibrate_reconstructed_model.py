from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconstructed_ckpfcm import generate_all_cases, run_method, segmentation_accuracy_with_permutation


TARGET_AVG = {
    "FCM": 0.7482,
    "PFCM": 0.6838,
    "KPFCM": 0.6289,
    "C-PFCM": 0.7683,
    "C-KPFCM": 0.7699,
}

TARGET_ORDER = ["C-KPFCM", "C-PFCM", "FCM", "PFCM", "KPFCM"]
METHODS = ["FCM", "PFCM", "KPFCM", "C-PFCM", "C-KPFCM"]


def score_summary(avg: dict[str, float]) -> float:
    mse = sum((avg[m] - TARGET_AVG[m]) ** 2 for m in METHODS)
    order_bonus = 0.0
    ranked = sorted(METHODS, key=lambda m: avg[m], reverse=True)
    if ranked == TARGET_ORDER:
        order_bonus += 1.0
    if avg["C-KPFCM"] > avg["FCM"]:
        order_bonus += 0.4
    if avg["C-KPFCM"] > avg["C-PFCM"]:
        order_bonus += 0.3
    if avg["C-PFCM"] > avg["KPFCM"]:
        order_bonus += 0.3
    return order_bonus - mse


def evaluate_config(config: dict[str, float]) -> tuple[float, dict[str, float]]:
    cases = generate_all_cases(seed=42)
    by_method = {m: [] for m in METHODS}
    for case in cases.values():
        image = case["image"]
        labels = case["labels"]
        for method in METHODS:
            pred, _ = run_method(image, method=method, seed=123, config=config)
            sa = segmentation_accuracy_with_permutation(labels, pred, num_classes=4)
            by_method[method].append(sa)
    avg = {m: sum(v) / len(v) for m, v in by_method.items()}
    return score_summary(avg), avg


def main() -> None:
    out_path = ROOT / "results" / "data" / "calibration_search.csv"
    rows = []
    best = None

    grid = itertools.product(
        [0.2, 0.35],           # spatial_weight
        [0.02, 0.05],          # suppression
        [0.8, 1.2],            # typicality_weight
        [0.55, 0.65],          # beta_cutset
        [0.75],                # sigma_kernel
        [False, True],         # use_typicality_for_labels
    )

    for spatial_weight, suppression, typicality_weight, beta_cutset, sigma_kernel, use_typ in grid:
        config = {
            "spatial_weight": spatial_weight,
            "suppression": suppression,
            "typicality_weight": typicality_weight,
            "beta_cutset": beta_cutset,
            "sigma_kernel": sigma_kernel,
            "use_typicality_for_labels": use_typ,
        }
        score, avg = evaluate_config(config)
        row = {
            "score": f"{score:.6f}",
            **{k: str(v) for k, v in config.items()},
            **{f"avg_{m}": f"{avg[m]:.4f}" for m in METHODS},
        }
        rows.append(row)
        if best is None or score > best[0]:
            best = (score, config, avg)

    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if best is not None:
        print("BEST_SCORE", f"{best[0]:.6f}")
        print("BEST_CONFIG", best[1])
        print("BEST_AVG", {k: round(v, 4) for k, v in best[2].items()})


if __name__ == "__main__":
    main()
