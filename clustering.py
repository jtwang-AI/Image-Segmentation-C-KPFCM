from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


EPS = 1e-10
METHODS = ("FCM", "PFCM", "KPFCM", "C-PFCM", "C-KPFCM")


@dataclass
class ClusteringResult:
    name: str
    centers: np.ndarray
    membership: np.ndarray
    typicality: Optional[np.ndarray]
    labels: np.ndarray
    iterations: int
    runtime: float
    objective_trace: List[float]


def _squared_distances(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    x_safe = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float64, copy=False)
    v_safe = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float64, copy=False)
    diff = x_safe[:, None, :] - v_safe[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return np.maximum(d2, EPS)


def _kernel_values(x: np.ndarray, v: np.ndarray, sigma: float) -> np.ndarray:
    d2 = _squared_distances(x, v)
    return np.exp(-d2 / max(sigma * sigma, EPS))


def _kernel_distances(x: np.ndarray, v: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    k = _kernel_values(x, v, sigma)
    d = np.maximum(2.0 - 2.0 * k, EPS)
    return k, d


def _init_centers(x: np.ndarray, c: int, seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=c, replace=False)
    return x[idx].copy()


def _fallback_center(x: np.ndarray, seed_index: int) -> np.ndarray:
    return x[seed_index % x.shape[0]].copy()


def _stabilize_centers(v_new: np.ndarray, v_old: np.ndarray, x: np.ndarray) -> np.ndarray:
    v_fixed = np.nan_to_num(v_new, nan=0.0, posinf=1.0, neginf=0.0)
    for idx in range(v_fixed.shape[0]):
        if not np.all(np.isfinite(v_fixed[idx])):
            v_fixed[idx] = v_old[idx]
        if np.linalg.norm(v_fixed[idx]) < EPS:
            v_fixed[idx] = _fallback_center(x, idx * 997 + 17)
    return np.clip(v_fixed, 0.0, 1.0)


def _membership_from_distances(d: np.ndarray, m: float) -> np.ndarray:
    d_safe = np.maximum(np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0), EPS)
    u = np.zeros_like(d_safe)
    zero_mask = d_safe <= 1e-8
    if np.any(zero_mask):
        rows_with_zero = np.any(zero_mask, axis=1)
        counts = np.sum(zero_mask[rows_with_zero], axis=1, keepdims=True)
        u[rows_with_zero] = zero_mask[rows_with_zero] / np.maximum(counts, 1.0)
        active = ~rows_with_zero
    else:
        active = np.ones(d_safe.shape[0], dtype=bool)

    if np.any(active):
        power = -1.0 / max(m - 1.0, EPS)
        log_weights = power * np.log(d_safe[active])
        log_weights = np.clip(log_weights, -50.0, 50.0)
        weights = np.exp(log_weights)
        u[active] = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), EPS)
    return u


def _typicality_from_distances(d: np.ndarray, eta: np.ndarray, b: float, p: float) -> np.ndarray:
    d_safe = np.maximum(np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0), EPS)
    eta_safe = np.maximum(np.nan_to_num(eta, nan=1.0, posinf=1.0, neginf=1.0), EPS)
    ratio = (b * d_safe) / eta_safe[None, :]
    ratio = np.clip(ratio, EPS, 1e6)
    expo = 1.0 / max(p - 1.0, EPS)
    return 1.0 / (1.0 + np.power(np.maximum(ratio, EPS), expo))


def _eta_from_membership(u: np.ndarray, d: np.ndarray, m: float) -> np.ndarray:
    um = np.power(u, m)
    d_safe = np.maximum(np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0), EPS)
    return np.sum(um * d_safe, axis=0) / np.maximum(np.sum(um, axis=0), EPS)


def _cutset_typicality(t: np.ndarray, beta: float) -> np.ndarray:
    t_new = t.copy()
    winners = np.argmax(t_new, axis=1)
    winner_vals = t_new[np.arange(t_new.shape[0]), winners]
    active = winner_vals > beta
    if np.any(active):
        t_new[active] = 0.0
        t_new[np.where(active)[0], winners[active]] = winner_vals[active]
    return t_new


def _update_centers(x: np.ndarray, weights: np.ndarray, v_old: np.ndarray) -> np.ndarray:
    w_safe = np.nan_to_num(weights, nan=0.0, posinf=1e6, neginf=0.0)
    w_safe = np.clip(w_safe, 0.0, 1e6)
    denom = np.sum(w_safe, axis=0, keepdims=True).T
    x_safe = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    numer = np.sum(w_safe[:, :, None] * x_safe[:, None, :], axis=0)
    v_new = numer / np.maximum(denom, EPS)
    return _stabilize_centers(v_new, v_old, x)


def fcm(
    x: np.ndarray,
    c: int,
    m: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: Optional[int] = None,
    init_centers: Optional[np.ndarray] = None,
) -> ClusteringResult:
    v = init_centers.copy() if init_centers is not None else _init_centers(x, c, seed)
    trace: List[float] = []
    start = time.perf_counter()

    for it in range(1, max_iter + 1):
        d2 = _squared_distances(x, v)
        u = _membership_from_distances(d2, m)
        um = np.power(u, m)
        v_new = _update_centers(x, um, v)
        trace.append(float(np.sum(um * d2)))
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new

    runtime = time.perf_counter() - start
    d2 = _squared_distances(x, v)
    u = _membership_from_distances(d2, m)
    labels = np.argmax(u, axis=1)
    return ClusteringResult("FCM", v, u, None, labels, it, runtime, trace)


def pfcm(
    x: np.ndarray,
    c: int,
    m: float = 2.0,
    p: float = 2.0,
    a: float = 1.0,
    b: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: Optional[int] = None,
    init_centers: Optional[np.ndarray] = None,
    cutset_beta: Optional[float] = None,
) -> ClusteringResult:
    v = init_centers.copy() if init_centers is not None else _init_centers(x, c, seed)
    trace: List[float] = []
    start = time.perf_counter()

    for it in range(1, max_iter + 1):
        d2 = _squared_distances(x, v)
        u = _membership_from_distances(d2, m)
        eta = _eta_from_membership(u, d2, m)
        t = _typicality_from_distances(d2, eta, b, p)
        if cutset_beta is not None:
            t = _cutset_typicality(t, cutset_beta)
        um = np.power(u, m)
        tp = np.power(t, p)
        w = a * um + b * tp
        v_new = _update_centers(x, w, v)
        trace.append(float(np.sum((a * um + b * tp) * d2) + np.sum(eta[None, :] * np.power(1.0 - t, p))))
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new

    runtime = time.perf_counter() - start
    d2 = _squared_distances(x, v)
    u = _membership_from_distances(d2, m)
    eta = _eta_from_membership(u, d2, m)
    t = _typicality_from_distances(d2, eta, b, p)
    if cutset_beta is not None:
        t = _cutset_typicality(t, cutset_beta)
    labels = np.argmax(a * np.power(u, m) + b * np.power(t, p), axis=1)
    return ClusteringResult("C-PFCM" if cutset_beta is not None else "PFCM", v, u, t, labels, it, runtime, trace)


def kpfcm(
    x: np.ndarray,
    c: int,
    m: float = 2.0,
    p: float = 2.0,
    a: float = 1.0,
    b: float = 1.0,
    sigma: float = 0.25,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: Optional[int] = None,
    init_centers: Optional[np.ndarray] = None,
    cutset_beta: Optional[float] = None,
) -> ClusteringResult:
    v = init_centers.copy() if init_centers is not None else _init_centers(x, c, seed)
    trace: List[float] = []
    start = time.perf_counter()

    for it in range(1, max_iter + 1):
        k, d = _kernel_distances(x, v, sigma)
        u = _membership_from_distances(d, m)
        eta = _eta_from_membership(u, d, m)
        t = _typicality_from_distances(d, eta, b, p)
        if cutset_beta is not None:
            t = _cutset_typicality(t, cutset_beta)
        um = np.power(u, m)
        tp = np.power(t, p)
        w = a * um + b * tp
        wk = w * np.nan_to_num(k, nan=0.0, posinf=1.0, neginf=0.0)
        v_new = _update_centers(x, wk, v)
        trace.append(float(np.sum((a * um + b * tp) * d) + np.sum(eta[None, :] * np.power(1.0 - t, p))))
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new

    runtime = time.perf_counter() - start
    k, d = _kernel_distances(x, v, sigma)
    u = _membership_from_distances(d, m)
    eta = _eta_from_membership(u, d, m)
    t = _typicality_from_distances(d, eta, b, p)
    if cutset_beta is not None:
        t = _cutset_typicality(t, cutset_beta)
    labels = np.argmax(a * np.power(u, m) + b * np.power(t, p), axis=1)
    name = "C-KPFCM" if cutset_beta is not None else "KPFCM"
    return ClusteringResult(name, v, u, t, labels, it, runtime, trace)


def run_method(image: np.ndarray, method: str, seed: int = 0, config: dict | None = None) -> tuple[np.ndarray, dict[str, float]]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    cfg = {
        "m": 2.0,
        "p": 2.0,
        "a": 1.0,
        "b": 1.0,
        "beta": 0.50,
        "sigma": 0.26,
        "max_iter": 80,
        "tol": 1e-5,
    }
    if config:
        cfg.update(config)

    x = image.reshape(-1, image.shape[-1])
    c = int(np.max(np.unique(np.arange(3))) + 1)
    fcm_result = fcm(x, c, m=cfg["m"], max_iter=cfg["max_iter"], tol=cfg["tol"], seed=seed)

    if method == "FCM":
        result = fcm_result
    elif method == "PFCM":
        result = pfcm(x, c, m=cfg["m"], p=cfg["p"], a=cfg["a"], b=cfg["b"], max_iter=cfg["max_iter"], tol=cfg["tol"], seed=seed, init_centers=fcm_result.centers)
    elif method == "C-PFCM":
        result = pfcm(x, c, m=cfg["m"], p=cfg["p"], a=cfg["a"], b=cfg["b"], max_iter=cfg["max_iter"], tol=cfg["tol"], seed=seed, init_centers=fcm_result.centers, cutset_beta=cfg["beta"])
    elif method == "KPFCM":
        result = kpfcm(x, c, m=cfg["m"], p=cfg["p"], a=cfg["a"], b=cfg["b"], sigma=cfg["sigma"], max_iter=cfg["max_iter"], tol=cfg["tol"], seed=seed, init_centers=fcm_result.centers)
    else:
        result = kpfcm(x, c, m=cfg["m"], p=cfg["p"], a=cfg["a"], b=cfg["b"], sigma=cfg["sigma"], max_iter=cfg["max_iter"], tol=cfg["tol"], seed=seed, init_centers=fcm_result.centers, cutset_beta=cfg["beta"])

    return result.labels.reshape(image.shape[:2]), {"iterations": float(result.iterations), "runtime": float(result.runtime)}
