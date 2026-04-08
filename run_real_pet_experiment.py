from __future__ import annotations

import csv
import sys
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reconstructed_ckpfcm.clustering import fcm, kpfcm, pfcm


DATA_ROOT = ROOT / "data" / "real_oxford_pets"
OUT_DATA = ROOT / "results" / "data"
OUT_FIG = ROOT / "results" / "figures"

SUBSET = [
    "Abyssinian_100",
    "Abyssinian_101",
    "Bengal_100",
    "Bengal_101",
    "Birman_100",
    "Birman_101",
    "american_bulldog_100",
    "american_bulldog_101",
    "american_pit_bull_terrier_100",
    "american_pit_bull_terrier_101",
    "basset_hound_100",
    "basset_hound_101",
]

CONFIG = {
    "m": 2.0,
    "p": 2.0,
    "a": 1.0,
    "b": 1.0,
    "beta": 0.50,
    "sigma": 0.26,
    "max_iter": 80,
    "tol": 1e-5,
    "spatial_weight": 0.20,
    "resize": 96,
}

METHODS = ["FCM", "PFCM", "KPFCM", "C-PFCM", "C-KPFCM"]


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_example(name: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(DATA_ROOT / "images" / f"{name}.jpg").convert("RGB").resize((size, size), Image.BILINEAR)
    trimap = Image.open(DATA_ROOT / "annotations" / "trimaps" / f"{name}.png").resize((size, size), Image.NEAREST)
    image_np = np.asarray(image, dtype=np.float64) / 255.0
    trimap_np = np.asarray(trimap, dtype=np.uint8)
    # Oxford-IIIT Pet trimap: 1 = pet, 2 = border, 3 = background.
    mask = np.isin(trimap_np, [1, 2]).astype(np.int64)
    return image_np, mask


def features_from_image(image: np.ndarray, spatial_weight: float) -> np.ndarray:
    h, w, _ = image.shape
    y, x = np.mgrid[0:h, 0:w]
    xy = np.stack([x / max(w - 1, 1), y / max(h - 1, 1)], axis=-1)
    feat = np.concatenate([image, spatial_weight * xy], axis=-1)
    return feat.reshape(-1, feat.shape[-1])


def remap_binary(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float]:
    best_score = -1.0
    best = pred.copy()
    for perm in permutations([0, 1]):
        mapped = np.zeros_like(pred)
        mapped[pred == 0] = perm[0]
        mapped[pred == 1] = perm[1]
        score = float(np.mean(mapped == gt))
        if score > best_score:
            best_score = score
            best = mapped
    return best, best_score


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_fg = pred == 1
    gt_fg = gt == 1
    inter = float(np.logical_and(pred_fg, gt_fg).sum())
    pred_sum = float(pred_fg.sum())
    gt_sum = float(gt_fg.sum())
    union = float(np.logical_or(pred_fg, gt_fg).sum())
    dice = (2.0 * inter) / max(pred_sum + gt_sum, 1.0)
    iou = inter / max(union, 1.0)
    return dice, iou


def run_method(method: str, x: np.ndarray, config: dict, seed: int):
    c = 2
    base = fcm(x, c, m=config["m"], max_iter=config["max_iter"], tol=config["tol"], seed=seed)
    if method == "FCM":
        return base
    if method == "PFCM":
        return pfcm(x, c, m=config["m"], p=config["p"], a=config["a"], b=config["b"], max_iter=config["max_iter"], tol=config["tol"], seed=seed, init_centers=base.centers)
    if method == "C-PFCM":
        return pfcm(x, c, m=config["m"], p=config["p"], a=config["a"], b=config["b"], max_iter=config["max_iter"], tol=config["tol"], seed=seed, init_centers=base.centers, cutset_beta=config["beta"])
    if method == "KPFCM":
        return kpfcm(x, c, m=config["m"], p=config["p"], a=config["a"], b=config["b"], sigma=config["sigma"], max_iter=config["max_iter"], tol=config["tol"], seed=seed, init_centers=base.centers)
    if method == "C-KPFCM":
        return kpfcm(x, c, m=config["m"], p=config["p"], a=config["a"], b=config["b"], sigma=config["sigma"], max_iter=config["max_iter"], tol=config["tol"], seed=seed, init_centers=base.centers, cutset_beta=config["beta"])
    raise ValueError(method)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    rows = []
    summaries = {m: {"sa": [], "dice": [], "iou": [], "runtime": []} for m in METHODS}
    qualitative = []

    for idx, name in enumerate(SUBSET):
        image, gt = load_example(name, CONFIG["resize"])
        x = features_from_image(image, CONFIG["spatial_weight"])
        for method in METHODS:
            model = run_method(method, x, CONFIG, seed=100 + idx)
            pred = model.labels.reshape(gt.shape)
            mapped, sa = remap_binary(pred, gt)
            dice, iou = dice_iou(mapped, gt)
            rows.append(
                {
                    "image": name,
                    "method": method,
                    "sa": f"{sa:.4f}",
                    "dice": f"{dice:.4f}",
                    "iou": f"{iou:.4f}",
                    "iterations": model.iterations,
                    "runtime_s": f"{model.runtime:.4f}",
                }
            )
            summaries[method]["sa"].append(sa)
            summaries[method]["dice"].append(dice)
            summaries[method]["iou"].append(iou)
            summaries[method]["runtime"].append(model.runtime)
            if len(qualitative) < 4 and method == "C-KPFCM":
                qualitative.append((name, image, gt, mapped))

    save_csv(
        OUT_DATA / "real_pet_per_image.csv",
        rows,
        ["image", "method", "sa", "dice", "iou", "iterations", "runtime_s"],
    )

    summary_rows = []
    for method in METHODS:
        summary_rows.append(
            {
                "method": method,
                "sa_mean": f"{np.mean(summaries[method]['sa']):.4f}",
                "dice_mean": f"{np.mean(summaries[method]['dice']):.4f}",
                "iou_mean": f"{np.mean(summaries[method]['iou']):.4f}",
                "runtime_mean_s": f"{np.mean(summaries[method]['runtime']):.4f}",
            }
        )
    save_csv(
        OUT_DATA / "real_pet_summary.csv",
        summary_rows,
        ["method", "sa_mean", "dice_mean", "iou_mean", "runtime_mean_s"],
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    methods = METHODS
    dice_vals = [float(r["dice_mean"]) for r in summary_rows]
    iou_vals = [float(r["iou_mean"]) for r in summary_rows]
    xloc = np.arange(len(methods))
    width = 0.38
    ax.bar(xloc - width / 2, dice_vals, width=width, label="Dice", color="#4C78A8")
    ax.bar(xloc + width / 2, iou_vals, width=width, label="IoU", color="#F58518")
    ax.set_xticks(xloc)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Oxford-IIIT Pet Subset: Real-Image Segmentation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "real_pet_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(qualitative), 3, figsize=(9, 3 * len(qualitative)))
    if len(qualitative) == 1:
        axes = np.array([axes])
    for row, (name, image, gt, pred) in enumerate(qualitative):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{name} image")
        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("GT mask")
        axes[row, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("C-KPFCM")
        for col in range(3):
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "real_pet_qualitative.png", dpi=180)
    plt.close(fig)

    with (OUT_DATA / "real_pet_config.txt").open("w", encoding="utf-8") as f:
        for key, value in CONFIG.items():
            f.write(f"{key}={value}\n")
        f.write("subset_size=12\n")
        f.write("dataset=Oxford-IIIT Pet official images+trimaps subset\n")


if __name__ == "__main__":
    main()
