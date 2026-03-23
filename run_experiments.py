from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import numpy as np

from algorithms import fcm, kpfcm, pfcm
from evaluation import optimal_accuracy, remap_labels, save_csv
from plotting import grouped_bar, heatmap_grid, line_plot, qualitative_grid, runtime_bar
from synthetic_data import (
    CLASS_COLORS,
    add_block_occlusion,
    add_gaussian_noise,
    add_illumination_gradient,
    add_salt_pepper_noise,
    add_texture_clutter,
    generate_cases,
)


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
DATA = RESULTS / "data"
REPORT = RESULTS / "report"


def features_from_image(image: np.ndarray) -> np.ndarray:
    return image.reshape(-1, image.shape[-1])


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[labels]


def average_scores(score_map: dict[str, list[float]]) -> dict[str, float]:
    return {method: float(np.mean(values)) for method, values in score_map.items()}


def run_method(method: str, x: np.ndarray, c: int, seed: int, config: dict) -> object:
    fcm_result = fcm(x, c, m=config["m"], max_iter=config["max_iter"], tol=config["tol"], seed=seed)
    if method == "FCM":
        return fcm_result
    if method == "PFCM":
        return pfcm(
            x,
            c,
            m=config["m"],
            p=config["p"],
            a=config["a"],
            b=config["b"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            seed=seed,
            init_centers=fcm_result.centers,
        )
    if method == "C-PFCM":
        return pfcm(
            x,
            c,
            m=config["m"],
            p=config["p"],
            a=config["a"],
            b=config["b"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            seed=seed,
            init_centers=fcm_result.centers,
            cutset_beta=config["beta"],
        )
    if method == "KPFCM":
        return kpfcm(
            x,
            c,
            m=config["m"],
            p=config["p"],
            a=config["a"],
            b=config["b"],
            sigma=config["sigma"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            seed=seed,
            init_centers=fcm_result.centers,
        )
    if method == "C-KPFCM":
        return kpfcm(
            x,
            c,
            m=config["m"],
            p=config["p"],
            a=config["a"],
            b=config["b"],
            sigma=config["sigma"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            seed=seed,
            init_centers=fcm_result.centers,
            cutset_beta=config["beta"],
        )
    raise ValueError(f"Unknown method: {method}")


def evaluate_case(case: dict, methods: list[str], config: dict, seed: int = 0) -> dict:
    x = features_from_image(case["image_float"])
    gt = case["labels"]
    c = len(np.unique(gt))
    results: dict[str, dict] = {}
    for method in methods:
        model = run_method(method, x, c, seed=seed, config=config)
        pred = model.labels.reshape(gt.shape)
        sa, mapping = optimal_accuracy(pred, gt, c)
        mapped = remap_labels(pred, mapping)
        results[method] = {
            "sa": sa,
            "mapped_labels": mapped,
            "iterations": model.iterations,
            "runtime": model.runtime,
            "objective_trace": model.objective_trace,
        }
    return results


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(REPORT, exist_ok=True)

    config = {
        "m": 2.0,
        "p": 2.0,
        "a": 1.0,
        "b": 1.0,
        "beta": 0.50,
        "sigma": 0.26,
        "max_iter": 80,
        "tol": 1e-5,
    }
    methods = ["FCM", "PFCM", "KPFCM", "C-PFCM", "C-KPFCM"]
    main_methods = ["FCM", "PFCM", "C-KPFCM"]
    cases = generate_cases(size=96)

    main_rows = []
    qualitative_cases = []
    case_names = list(cases.keys())
    bar_values = {m: [] for m in main_methods}
    ablation_values = {m: [] for m in methods}

    for idx, name in enumerate(case_names):
        case = cases[name]
        result = evaluate_case(case, methods, config, seed=idx + 1)
        row = [name]
        for method in methods:
            sa = result[method]["sa"]
            row.append(f"{sa:.4f}")
            ablation_values[method].append(sa)
            if method in bar_values:
                bar_values[method].append(sa)
        main_rows.append(row)
        qualitative_cases.append(
            {
                "name": name,
                "image": case["image_uint8"],
                "gt_rgb": labels_to_rgb(case["labels"]),
                "segmentations": {m: labels_to_rgb(result[m]["mapped_labels"]) for m in main_methods},
            }
        )

    averages = ["Average"]
    for method in methods:
        averages.append(f"{np.mean(ablation_values[method]):.4f}")
    main_rows.append(averages)
    save_csv(DATA / "main_comparison.csv", ["Case"] + methods, main_rows)
    grouped_bar(case_names, bar_values, "SA", "Main Quantitative Comparison", str(FIGS / "main_comparison.png"))
    grouped_bar(case_names, ablation_values, "SA", "Ablation Study", str(FIGS / "ablation.png"))
    qualitative_grid(qualitative_cases, main_methods, str(FIGS / "qualitative_segmentation.png"))

    # Noise robustness.
    gaussian_levels = [0.01, 0.03, 0.05, 0.07]
    sp_levels = [0.01, 0.03, 0.05, 0.07]
    gaussian_series = {m: [] for m in main_methods}
    sp_series = {m: [] for m in main_methods}

    for sigma_n in gaussian_levels:
        scores = {m: [] for m in main_methods}
        for idx, name in enumerate(case_names):
            case = cases[name]
            noisy = add_gaussian_noise(case["image_float"], sigma=sigma_n, seed=100 + idx)
            noisy_case = {"image_float": noisy, "labels": case["labels"]}
            result = evaluate_case(noisy_case, main_methods, config, seed=10 + idx)
            for method in main_methods:
                scores[method].append(result[method]["sa"])
        for method in main_methods:
            gaussian_series[method].append(float(np.mean(scores[method])))

    for density in sp_levels:
        scores = {m: [] for m in main_methods}
        for idx, name in enumerate(case_names):
            case = cases[name]
            noisy = add_salt_pepper_noise(case["image_float"], density=density, seed=200 + idx)
            noisy_case = {"image_float": noisy, "labels": case["labels"]}
            result = evaluate_case(noisy_case, main_methods, config, seed=20 + idx)
            for method in main_methods:
                scores[method].append(result[method]["sa"])
        for method in main_methods:
            sp_series[method].append(float(np.mean(scores[method])))

    line_plot(gaussian_levels, gaussian_series, "Gaussian noise sigma", "Average SA", "Gaussian Noise Robustness", str(FIGS / "gaussian_noise.png"))
    line_plot(sp_levels, sp_series, "Salt-and-pepper density", "Average SA", "Salt-and-pepper Noise Robustness", str(FIGS / "salt_pepper_noise.png"))
    save_csv(DATA / "gaussian_noise.csv", ["Level"] + main_methods, [[f"{lv:.2f}"] + [f"{gaussian_series[m][i]:.4f}" for m in main_methods] for i, lv in enumerate(gaussian_levels)])
    save_csv(DATA / "salt_pepper_noise.csv", ["Level"] + main_methods, [[f"{lv:.2f}"] + [f"{sp_series[m][i]:.4f}" for m in main_methods] for i, lv in enumerate(sp_levels)])

    # Combined disturbance landscape.
    combined_rows = []
    combined_landscape = {m: np.zeros((len(sp_levels), len(gaussian_levels)), dtype=np.float64) for m in main_methods}
    for sp_idx, density in enumerate(sp_levels):
        for gauss_idx, sigma_n in enumerate(gaussian_levels):
            scores = {m: [] for m in main_methods}
            for idx, name in enumerate(case_names):
                case = cases[name]
                disturbed = add_gaussian_noise(case["image_float"], sigma=sigma_n, seed=300 + idx)
                disturbed = add_salt_pepper_noise(disturbed, density=density, seed=400 + idx)
                disturbed_case = {"image_float": disturbed, "labels": case["labels"]}
                result = evaluate_case(disturbed_case, main_methods, config, seed=60 + idx)
                for method in main_methods:
                    scores[method].append(result[method]["sa"])
            averages = average_scores(scores)
            combined_rows.append([f"{density:.2f}", f"{sigma_n:.2f}"] + [f"{averages[m]:.4f}" for m in main_methods])
            for method in main_methods:
                combined_landscape[method][sp_idx, gauss_idx] = averages[method]
    save_csv(DATA / "combined_noise_grid.csv", ["SP_density", "Gaussian_sigma"] + main_methods, combined_rows)
    heatmap_grid(
        [f"{density:.2f}" for density in sp_levels],
        [f"{sigma_n:.2f}" for sigma_n in gaussian_levels],
        combined_landscape,
        "Combined Noise Robustness Landscape",
        str(FIGS / "combined_noise_heatmap.png"),
    )

    # Richer stress scenarios.
    scenario_builders = {
        "Clean": lambda image, seed: image,
        "Mixed noise": lambda image, seed: add_salt_pepper_noise(add_gaussian_noise(image, sigma=0.03, seed=seed + 11), density=0.04, seed=seed + 19),
        "Illum. drift": lambda image, seed: add_gaussian_noise(add_illumination_gradient(image, strength=0.22), sigma=0.02, seed=seed + 29),
        "Occ. clutter": lambda image, seed: add_texture_clutter(add_block_occlusion(image, fraction=0.08, seed=seed + 31), strength=1.0, seed=seed + 37),
        "Severe compound": lambda image, seed: add_texture_clutter(
            add_block_occlusion(
                add_salt_pepper_noise(
                    add_gaussian_noise(add_illumination_gradient(image, strength=0.26), sigma=0.04, seed=seed + 41),
                    density=0.05,
                    seed=seed + 43,
                ),
                fraction=0.10,
                seed=seed + 47,
            ),
            strength=1.1,
            seed=seed + 53,
        ),
    }
    scenario_values = {m: [] for m in main_methods}
    scenario_rows = []
    for scenario_idx, (scenario_name, builder) in enumerate(scenario_builders.items()):
        scores = {m: [] for m in main_methods}
        for idx, name in enumerate(case_names):
            case = cases[name]
            stressed = builder(case["image_float"], seed=500 + scenario_idx * 50 + idx)
            stressed_case = {"image_float": stressed, "labels": case["labels"]}
            result = evaluate_case(stressed_case, main_methods, config, seed=80 + scenario_idx * 10 + idx)
            for method in main_methods:
                scores[method].append(result[method]["sa"])
        averages = average_scores(scores)
        scenario_rows.append([scenario_name] + [f"{averages[m]:.4f}" for m in main_methods])
        for method in main_methods:
            scenario_values[method].append(averages[method])
    save_csv(DATA / "stress_scenarios.csv", ["Scenario"] + main_methods, scenario_rows)
    grouped_bar(list(scenario_builders.keys()), scenario_values, "Average SA", "Stress Test Under Complex Disturbances", str(FIGS / "stress_scenarios.png"))

    # Parameter sensitivity.
    beta_values = [0.50, 0.60, 0.70, 0.80, 0.90]
    sigma_values = [0.14, 0.18, 0.22, 0.26, 0.30]
    m_values = [1.5, 2.0, 2.5, 3.0]
    beta_scores = []
    sigma_scores = []
    m_scores = []

    for beta in beta_values:
        cfg = dict(config)
        cfg["beta"] = beta
        vals = []
        for idx, name in enumerate(case_names):
            result = evaluate_case(cases[name], ["C-KPFCM"], cfg, seed=30 + idx)
            vals.append(result["C-KPFCM"]["sa"])
        beta_scores.append(float(np.mean(vals)))

    for sigma in sigma_values:
        cfg = dict(config)
        cfg["sigma"] = sigma
        vals = []
        for idx, name in enumerate(case_names):
            result = evaluate_case(cases[name], ["C-KPFCM"], cfg, seed=40 + idx)
            vals.append(result["C-KPFCM"]["sa"])
        sigma_scores.append(float(np.mean(vals)))

    for m in m_values:
        cfg = dict(config)
        cfg["m"] = m
        vals = []
        for idx, name in enumerate(case_names):
            result = evaluate_case(cases[name], ["C-KPFCM"], cfg, seed=50 + idx)
            vals.append(result["C-KPFCM"]["sa"])
        m_scores.append(float(np.mean(vals)))

    line_plot(beta_values, {"C-KPFCM": beta_scores}, r"Cutset threshold $\beta$", "Average SA", "Sensitivity to Cutset Threshold", str(FIGS / "beta_sensitivity.png"))
    line_plot(sigma_values, {"C-KPFCM": sigma_scores}, r"Kernel width $\sigma$", "Average SA", "Sensitivity to Kernel Width", str(FIGS / "sigma_sensitivity.png"))
    line_plot(m_values, {"C-KPFCM": m_scores}, "Fuzzifier m", "Average SA", "Sensitivity to Fuzzifier", str(FIGS / "m_sensitivity.png"))
    save_csv(DATA / "parameter_sensitivity.csv", ["Parameter", "Value", "Average_SA"], [["beta", f"{v:.2f}", f"{beta_scores[i]:.4f}"] for i, v in enumerate(beta_values)] + [["sigma", f"{v:.2f}", f"{sigma_scores[i]:.4f}"] for i, v in enumerate(sigma_values)] + [["m", f"{v:.2f}", f"{m_scores[i]:.4f}"] for i, v in enumerate(m_values)])

    # Repeated-run stability.
    repeat_methods = ["FCM", "PFCM", "C-KPFCM"]
    repeat_stats = {m: {"sa": [], "iter": [], "time": []} for m in repeat_methods}
    for seed in range(10):
        for name in case_names:
            result = evaluate_case(cases[name], repeat_methods, config, seed=1000 + seed)
            for method in repeat_methods:
                repeat_stats[method]["sa"].append(result[method]["sa"])
                repeat_stats[method]["iter"].append(result[method]["iterations"])
                repeat_stats[method]["time"].append(result[method]["runtime"])

    runtime_rows = []
    mean_sa = []
    std_sa = []
    mean_time = []
    for method in repeat_methods:
        sa_mean = statistics.mean(repeat_stats[method]["sa"])
        sa_std = statistics.pstdev(repeat_stats[method]["sa"])
        iter_mean = statistics.mean(repeat_stats[method]["iter"])
        time_mean = statistics.mean(repeat_stats[method]["time"])
        mean_sa.append(sa_mean)
        std_sa.append(sa_std)
        mean_time.append(time_mean)
        runtime_rows.append([method, f"{sa_mean:.4f}", f"{sa_std:.4f}", f"{iter_mean:.2f}", f"{time_mean:.4f}"])
    save_csv(DATA / "runtime_stability.csv", ["Method", "SA_mean", "SA_std", "Iterations_mean", "Time_mean_s"], runtime_rows)
    runtime_bar(repeat_methods, mean_sa, std_sa, mean_time, str(FIGS / "runtime_stability.png"))

    summary = {
        "config": config,
        "main_average": {m: float(np.mean(ablation_values[m])) for m in methods},
        "gaussian_noise": gaussian_series,
        "salt_pepper_noise": sp_series,
        "combined_noise_landscape": {m: combined_landscape[m].round(4).tolist() for m in main_methods},
        "stress_scenarios": {row[0]: {main_methods[i]: row[i + 1] for i in range(len(main_methods))} for row in scenario_rows},
        "beta_sensitivity": beta_scores,
        "sigma_sensitivity": sigma_scores,
        "m_sensitivity": m_scores,
        "runtime_stability": {row[0]: {"sa_mean": row[1], "sa_std": row[2], "iter_mean": row[3], "time_mean_s": row[4]} for row in runtime_rows},
    }
    with open(DATA / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(REPORT / "experiment_report.md", "w", encoding="utf-8") as f:
        f.write("# Experiment Report\n\n")
        f.write("This report is generated by `code/run_experiments.py` using synthetic complex-background segmentation cases.\n\n")
        f.write("## Default configuration\n\n")
        for key, value in config.items():
            f.write(f"- `{key}` = `{value}`\n")
        f.write("\n## Generated outputs\n\n")
        for name in [
            "qualitative_segmentation.png",
            "main_comparison.png",
            "ablation.png",
            "gaussian_noise.png",
            "salt_pepper_noise.png",
            "combined_noise_heatmap.png",
            "stress_scenarios.png",
            "beta_sensitivity.png",
            "sigma_sensitivity.png",
            "m_sensitivity.png",
            "runtime_stability.png",
        ]:
            f.write(f"- `results/figures/{name}`\n")


if __name__ == "__main__":
    main()
