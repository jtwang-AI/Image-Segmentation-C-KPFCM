from __future__ import annotations

from itertools import permutations

import numpy as np


def segmentation_accuracy_with_permutation(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    best = 0.0
    pred_classes = list(range(num_classes))
    true_classes = list(range(num_classes))
    for perm in permutations(true_classes):
        mapping = {pred_classes[i]: perm[i] for i in range(num_classes)}
        mapped = np.vectorize(mapping.get)(y_pred)
        score = float(np.mean(mapped == y_true))
        if score > best:
            best = score
    return best
