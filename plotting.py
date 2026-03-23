from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


PLOT_COLORS = {
    "FCM": "#4C78A8",
    "PFCM": "#F58518",
    "KPFCM": "#B279A2",
    "C-PFCM": "#54A24B",
    "C-KPFCM": "#E45756",
}


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def qualitative_grid(cases: list[dict], method_order: list[str], path: str) -> None:
    cols = 2 + len(method_order)
    fig, axes = plt.subplots(len(cases), cols, figsize=(3.0 * cols, 3.0 * len(cases)))
    if len(cases) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, case in enumerate(cases):
        axes[row, 0].imshow(case["image"])
        axes[row, 0].set_title(f'{case["name"]}: Input')
        axes[row, 1].imshow(case["gt_rgb"])
        axes[row, 1].set_title("Ground Truth")
        for col, method in enumerate(method_order, start=2):
            axes[row, col].imshow(case["segmentations"][method])
            axes[row, col].set_title(method)
        for ax in axes[row]:
            ax.axis("off")

    _save(fig, path)


def grouped_bar(case_names: list[str], values: dict[str, list[float]], ylabel: str, title: str, path: str) -> None:
    methods = list(values.keys())
    x = np.arange(len(case_names))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, method in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, values[method], width=width, label=method, color=PLOT_COLORS.get(method))
    ax.set_xticks(x, case_names)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    _save(fig, path)


def line_plot(x_values: list[float], series: dict[str, list[float]], xlabel: str, ylabel: str, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, y_values in series.items():
        ax.plot(x_values, y_values, marker="o", linewidth=2.0, label=method, color=PLOT_COLORS.get(method))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    _save(fig, path)


def runtime_bar(methods: list[str], accuracy_mean: list[float], accuracy_std: list[float], runtimes: list[float], path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(methods, accuracy_mean, yerr=accuracy_std, capsize=5, color=[PLOT_COLORS.get(m) for m in methods])
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("SA")
    axes[0].set_title("Repeated-run Accuracy")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].bar(methods, runtimes, color=[PLOT_COLORS.get(m) for m in methods])
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Average Runtime")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    _save(fig, path)


def heatmap_grid(
    row_labels: list[str],
    col_labels: list[str],
    matrices: dict[str, np.ndarray],
    super_title: str,
    path: str,
    cmap: str = "viridis",
) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.3 * len(matrices), 4.4), sharey=True)
    if len(matrices) == 1:
        axes = [axes]

    vmin = min(float(np.min(matrix)) for matrix in matrices.values())
    vmax = max(float(np.max(matrix)) for matrix in matrices.values())
    image = None

    for ax, (name, matrix) in zip(axes, matrices.items()):
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(name)
        ax.set_xticks(np.arange(len(col_labels)), col_labels)
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.set_xlabel("Gaussian sigma")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    axes[0].set_ylabel("Salt-pepper density")
    fig.suptitle(super_title, y=1.02)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.86, label="Average SA")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.15, top=0.82, wspace=0.20)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
