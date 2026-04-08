from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from reconstructed_ckpfcm import METHODS, generate_all_cases, run_method, segmentation_accuracy_with_permutation

RUN_CONFIG = {
    "m": 2.0,
    "p": 2.0,
    "a": 1.0,
    "b": 1.0,
    "beta": 0.50,
    "sigma": 0.26,
    "max_iter": 80,
    "tol": 1e-5,
}


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_data = ROOT / "results" / "data"
    out_fig = ROOT / "results" / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    cases = generate_all_cases(seed=42)
    summary_rows = []
    qual_methods = ["FCM", "PFCM", "C-KPFCM"]
    first_case = cases["CAO"]
    qualitative_preds = {}

    for case_name, case in cases.items():
        image = case["image"]
        labels = case["labels"]
        for method in METHODS:
            start = time.perf_counter()
            pred, meta = run_method(image, method=method, seed=123, config=RUN_CONFIG)
            runtime = time.perf_counter() - start
            sa = segmentation_accuracy_with_permutation(labels, pred, num_classes=3)
            summary_rows.append(
                {
                    "case": case_name,
                    "method": method,
                    "sa": f"{sa:.4f}",
                    "iterations": int(meta["iterations"]),
                    "runtime_s": f"{runtime:.4f}",
                }
            )
            if case_name == "CAO" and method in qual_methods:
                qualitative_preds[method] = pred

    save_csv(
        out_data / "benchmark_run_reconstructed.csv",
        summary_rows,
        fieldnames=["case", "method", "sa", "iterations", "runtime_s"],
    )

    fig, axes = plt.subplots(1, 2 + len(qual_methods), figsize=(14, 3.8))
    axes[0].imshow(first_case["image"])
    axes[0].set_title("Input")
    axes[1].imshow(first_case["labels"], cmap="tab10", vmin=0, vmax=2)
    axes[1].set_title("Reference")
    for idx, method in enumerate(qual_methods, start=2):
        axes[idx].imshow(qualitative_preds[method], cmap="tab10", vmin=0, vmax=2)
        axes[idx].set_title(method)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_fig / "qualitative_case_cao_reconstructed.png", dpi=180)
    plt.close(fig)

    with (out_data / "benchmark_run_config.txt").open("w", encoding="utf-8") as f:
        for key, value in RUN_CONFIG.items():
            f.write(f"{key}={value}\n")

    pivot = {}
    for row in summary_rows:
        pivot.setdefault(row["case"], {})[row["method"]] = float(row["sa"])

    ordered_cases = ["CAO", "GU", "WEI"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(ordered_cases))
    width = 0.16
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
    for idx, method in enumerate(METHODS):
        vals = [pivot[c][method] for c in ordered_cases]
        ax.bar(x + (idx - 2) * width, vals, width=width, label=method, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_cases)
    ax.set_ylabel("Segmentation Accuracy")
    ax.set_title("Reimplemented Benchmark Run")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(out_fig / "benchmark_run_reconstructed.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
