from __future__ import annotations

import csv
import itertools
import os
from typing import Iterable

import numpy as np


def save_csv(path: str, header: list[str], rows: Iterable[Iterable[object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def optimal_accuracy(pred: np.ndarray, gt: np.ndarray, classes: int) -> tuple[float, dict[int, int]]:
    best_acc = -1.0
    best_map: dict[int, int] = {}
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    for perm in itertools.permutations(range(classes)):
        mapping = {src: dst for src, dst in enumerate(perm)}
        mapped = np.vectorize(mapping.get)(pred_flat)
        acc = float(np.mean(mapped == gt_flat))
        if acc > best_acc:
            best_acc = acc
            best_map = mapping
    return best_acc, best_map


def remap_labels(pred: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    mapped = np.vectorize(mapping.get)(pred.reshape(-1))
    return mapped.reshape(pred.shape)
