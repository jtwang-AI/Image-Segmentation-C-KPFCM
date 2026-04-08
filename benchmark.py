from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


CLASS_COLORS = np.array(
    [
        [32, 74, 145],
        [202, 93, 54],
        [175, 188, 55],
    ],
    dtype=np.uint8,
)


def _xy_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(coords, coords)
    return xx, yy


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float64) / 255.0, 0.0, 1.0)


def _base_texture(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xx, yy = _xy_grid(size)
    noise = rng.normal(0.0, 0.03, size=(size, size, 3))
    tex = np.zeros((size, size, 3), dtype=np.float64)
    tex[..., 0] = 0.10 * np.sin(4.0 * np.pi * xx) + 0.06 * np.cos(3.5 * np.pi * yy)
    tex[..., 1] = 0.09 * np.cos(5.0 * np.pi * xx * yy) + 0.07 * np.sin(4.0 * np.pi * yy)
    tex[..., 2] = 0.08 * np.sin(6.0 * np.pi * (xx + yy)) + 0.05 * np.cos(4.0 * np.pi * xx)
    return tex + noise


def _render_case(labels: np.ndarray, seed: int) -> np.ndarray:
    size = labels.shape[0]
    rng = np.random.default_rng(seed)
    xx, yy = _xy_grid(size)
    theta = np.arctan2(yy + 0.08, xx - 0.05)
    radius = np.sqrt((xx - 0.05) ** 2 + (yy + 0.04) ** 2)
    texture = _base_texture(size, seed + 17)
    image = np.zeros((size, size, 3), dtype=np.float64)

    mask0 = labels == 0
    mask1 = labels == 1
    mask2 = labels == 2

    image[..., 0][mask0] = 0.44 + 0.012 * np.sin(2.5 * np.pi * xx[mask0]) + 0.008 * np.cos(2.0 * np.pi * yy[mask0])
    image[..., 1][mask0] = 0.44 + 0.012 * np.cos(3.0 * np.pi * yy[mask0]) + 0.008 * np.sin(2.0 * np.pi * xx[mask0] * yy[mask0])
    image[..., 2][mask0] = 0.43 + 0.010 * np.sin(2.0 * np.pi * (xx[mask0] + yy[mask0]))

    ring = 0.20 + 0.05 * np.sin(3.5 * theta[mask1]) + 0.02 * np.cos(4.0 * np.pi * radius[mask1])
    image[..., 0][mask1] = 0.44 + ring * np.cos(theta[mask1])
    image[..., 1][mask1] = 0.44 + ring * np.sin(theta[mask1])
    image[..., 2][mask1] = 0.43 + 0.012 * np.sin(4.5 * theta[mask1])

    moon = yy[mask2] - 0.62 * xx[mask2] ** 2 + 0.12 * np.sin(2.2 * np.pi * xx[mask2])
    image[..., 0][mask2] = 0.58 + 0.06 * np.tanh(2.5 * moon)
    image[..., 1][mask2] = 0.33 + 0.10 * np.sin(2.0 * np.pi * xx[mask2]) - 0.02 * yy[mask2]
    image[..., 2][mask2] = 0.28 + 0.08 * np.cos(2.0 * np.pi * moon) + 0.012 * xx[mask2]

    contam_mask = rng.random(labels.shape) < (0.04 + 0.05 * (np.abs(texture[..., 0]) > 0.10))
    contam0 = contam_mask & mask0
    contam1 = contam_mask & mask1
    contam2 = contam_mask & mask2
    image[..., 0][contam0] = 0.56 + 0.03 * rng.standard_normal(np.sum(contam0))
    image[..., 1][contam0] = 0.31 + 0.03 * rng.standard_normal(np.sum(contam0))
    image[..., 2][contam0] = 0.28 + 0.03 * rng.standard_normal(np.sum(contam0))
    image[..., 0][contam1] = 0.43 + 0.02 * rng.standard_normal(np.sum(contam1))
    image[..., 1][contam1] = 0.45 + 0.02 * rng.standard_normal(np.sum(contam1))
    image[..., 2][contam1] = 0.44 + 0.02 * rng.standard_normal(np.sum(contam1))
    image[..., 0][contam2] = 0.47 + 0.05 * rng.standard_normal(np.sum(contam2))
    image[..., 1][contam2] = 0.47 + 0.05 * rng.standard_normal(np.sum(contam2))
    image[..., 2][contam2] = 0.47 + 0.05 * rng.standard_normal(np.sum(contam2))

    local_noise = rng.normal(0.0, 0.016, size=(size, size, 3))
    image = np.clip(image + 0.18 * texture + local_noise, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def _draw_case_cao(size: int) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, size - 1, size - 1), fill=0)
    draw.ellipse((22, 18, 108, 102), fill=1)
    draw.polygon([(8, 90), (44, 118), (8, 126)], fill=2)
    draw.polygon([(88, 8), (126, 28), (104, 70), (74, 48)], fill=2)
    draw.rectangle((42, 0, 58, 28), fill=2)
    return np.array(img, dtype=np.int64)


def _draw_case_gu(size: int) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((16, 24, 114, 108), radius=30, fill=1)
    draw.polygon([(0, 18), (40, 10), (52, 32), (18, 46)], fill=2)
    draw.polygon([(80, 82), (126, 94), (126, 126), (74, 116)], fill=2)
    draw.ellipse((88, 6, 120, 40), fill=2)
    return np.array(img, dtype=np.int64)


def _draw_case_wei(size: int) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([(24, 20), (100, 14), (110, 62), (94, 110), (30, 106), (14, 60)], fill=1)
    draw.ellipse((0, 72, 38, 126), fill=2)
    draw.rectangle((92, 0, 126, 26), fill=2)
    draw.polygon([(62, 48), (118, 74), (118, 98), (74, 80)], fill=2)
    return np.array(img, dtype=np.int64)


def generate_all_cases(seed: int = 0, size: int = 96) -> dict[str, dict[str, np.ndarray]]:
    del seed
    builders = {
        "CAO": (_draw_case_cao, 11),
        "GU": (_draw_case_gu, 23),
        "WEI": (_draw_case_wei, 37),
    }
    cases: dict[str, dict[str, np.ndarray]] = {}
    for name, (builder, render_seed) in builders.items():
        labels = builder(size)
        rgb = _render_case(labels, render_seed)
        cases[name] = {
            "name": name,
            "image": _normalize_rgb(rgb),
            "image_uint8": rgb,
            "labels": labels,
        }
    return cases


def generate_case(name: str, size: tuple[int, int] = (96, 96), seed: int = 0) -> dict[str, np.ndarray]:
    del seed
    if size[0] != size[1]:
        raise ValueError("Only square sizes are supported")
    return generate_all_cases(size=size[0])[name]
