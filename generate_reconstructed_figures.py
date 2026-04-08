from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"
FIGS = ROOT / "results" / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_main_comparison() -> None:
    rows = read_csv(DATA / "main_comparison.csv")
    cases = [r["case"] for r in rows if r["case"] != "Average"]
    methods = ["FCM", "PFCM", "KPFCM", "C-PFCM", "C-KPFCM"]
    colors = {
        "FCM": "#4C78A8",
        "PFCM": "#F58518",
        "KPFCM": "#E45756",
        "C-PFCM": "#72B7B2",
        "C-KPFCM": "#54A24B",
    }

    x = np.arange(len(cases))
    width = 0.16

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for idx, method in enumerate(methods):
        vals = [float(r[method]) for r in rows if r["case"] != "Average"]
        ax.bar(x + (idx - 2) * width, vals, width=width, label=method, color=colors[method])

    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("Segmentation Accuracy")
    ax.set_ylim(0.4, 0.86)
    ax.set_title("Main Comparison on Reconstructed Benchmark")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(FIGS / "main_comparison_reconstructed.png", dpi=200)
    plt.close(fig)


def plot_ablation_summary() -> None:
    rows = read_csv(DATA / "main_comparison.csv")
    avg_row = next(r for r in rows if r["case"] == "Average")
    methods = ["FCM", "PFCM", "KPFCM", "C-PFCM", "C-KPFCM"]
    values = [float(avg_row[m]) for m in methods]
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(methods, values, color=colors)
    ax.set_ylabel("Average SA")
    ax.set_ylim(0.55, 0.82)
    ax.set_title("Ablation-Oriented Average Performance")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "ablation_reconstructed.png", dpi=200)
    plt.close(fig)


def plot_stability_runtime() -> None:
    rows = read_csv(DATA / "repeated_run_stability.csv")
    methods = [r["method"] for r in rows]
    sa_mean = np.array([float(r["sa_mean"]) for r in rows])
    sa_std = np.array([float(r["sa_std"]) for r in rows])
    runtime = np.array([float(r["time_mean_s"]) for r in rows])

    x = np.arange(len(methods))
    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    bars = ax1.bar(x, sa_mean, yerr=sa_std, capsize=4, color=["#4C78A8", "#F58518", "#54A24B"])
    ax1.set_ylabel("SA Mean")
    ax1.set_ylim(0.55, 0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_title("Repeated-Run Accuracy and Runtime")
    ax1.grid(axis="y", alpha=0.25, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(x, runtime, color="#222222", marker="o", linewidth=2)
    ax2.set_ylabel("Runtime (s)")
    ax2.set_ylim(0.045, 0.078)

    for bar, val in zip(bars, sa_mean):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGS / "stability_runtime_reconstructed.png", dpi=200)
    plt.close(fig)


def plot_partial_robustness() -> None:
    rows = read_csv(DATA / "robustness_partial.csv")
    gaussian = [r for r in rows if r["noise_type"] == "gaussian"]
    sp = [r for r in rows if r["noise_type"] == "salt_pepper"]
    compound = [r for r in rows if r["noise_type"] == "compound_example"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].plot(
        [float(r["level"]) for r in gaussian],
        [float(r["sa"]) for r in gaussian],
        marker="o",
        linewidth=2,
        color="#54A24B",
        label="C-KPFCM",
    )
    axes[0].plot(
        [float(r["level"]) for r in sp],
        [float(r["sa"]) for r in sp],
        marker="s",
        linewidth=2,
        color="#4C78A8",
        label="C-KPFCM",
    )
    axes[0].set_title("Recovered Robustness Values")
    axes[0].set_xlabel("Noise Level / Density")
    axes[0].set_ylabel("SA")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(["Gaussian", "Salt-and-pepper"], frameon=False)

    comp_methods = [r["method"] for r in compound]
    comp_vals = [float(r["sa"]) for r in compound]
    axes[1].bar(comp_methods, comp_vals, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1].set_title("Recovered Compound-Disturbance Example")
    axes[1].set_ylabel("SA")
    axes[1].set_ylim(0.45, 0.76)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(FIGS / "robustness_partial_reconstructed.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_main_comparison()
    plot_ablation_summary()
    plot_stability_runtime()
    plot_partial_robustness()


if __name__ == "__main__":
    main()
