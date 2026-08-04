import csv
from pathlib import Path

import pytest

from admpo_repro.plotting import plot_figure2, plot_figure4


def test_plotters_create_png_pdf_and_summaries(tmp_path: Path):
    fig2 = tmp_path / "results" / "figure2" / "per_seed"
    fig2.mkdir(parents=True)
    with (fig2 / "sample.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "task", "seed", "model", "rollout_length", "error_mean",
            "error_std", "n_active", "rollout_repeats",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in ("hopper-medium-v2", "hopper-medium-replay-v2", "walker2d-medium-v2", "walker2d-medium-replay-v2"):
            for model in ("adm", "ensemble", "rnn"):
                writer.writerow({"task": task, "seed": 0, "model": model, "rollout_length": 1, "error_mean": 0.1, "error_std": 0, "n_active": 2, "rollout_repeats": 20})
    assert all(path.exists() for path in plot_figure2(tmp_path))
    assert (tmp_path / "results" / "figure2" / "summary.csv").exists()

    fig4 = tmp_path / "results" / "figure4" / "points"
    fig4.mkdir(parents=True)
    with (fig4 / "sample.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "task", "seed", "model", "policy", "rollout_step",
            "trajectory_error_rmse", "trajectory_error_mse",
            "local_error_rmse", "local_error_mse", "uncertainty_std",
            "uncertainty_var", "total_uncertainty_std", "total_uncertainty_var",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in ("hopper-medium-replay-v2", "walker2d-medium-replay-v2"):
            for model in ("adm", "ensemble"):
                for policy in ("random", "learned", "dataset"):
                    for i in range(3):
                        writer.writerow({
                            "task": task, "seed": 0, "model": model,
                            "policy": policy, "rollout_step": 1,
                            "trajectory_error_rmse": i + 0.1,
                            "trajectory_error_mse": (i + 0.1) ** 2,
                            "local_error_rmse": i + 0.2,
                            "local_error_mse": (i + 0.2) ** 2,
                            "uncertainty_std": i + 0.3,
                            "uncertainty_var": (i + 0.3) ** 2,
                            "total_uncertainty_std": i + 0.4,
                            "total_uncertainty_var": (i + 0.4) ** 2,
                        })
    assert all(path.exists() for path in plot_figure4(tmp_path))
    assert (tmp_path / "results" / "figure4" / "correlations.csv").exists()

    with (fig2 / "deadline48h_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "seed", "model", "rollout_length", "error_mean", "error_std", "n_active"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in ("hopper-medium-replay-v2", "walker2d-medium-replay-v2"):
            for seed in (0, 1, 2):
                for model in ("adm", "ensemble", "rnn"):
                    writer.writerow({"task": task, "seed": seed, "model": model, "rollout_length": 1, "error_mean": 0.1 + seed, "error_std": 0, "n_active": 2})
    assert all(path.exists() for path in plot_figure2(tmp_path, "deadline48h_"))
    assert (tmp_path / "results" / "figure2" / "deadline48h_summary.csv").exists()

    assert (tmp_path / "results" / "figure4" / "policy_summary.csv").exists()


def test_figure2_plot_rejects_repeats_misrepresented_as_seed_rows(tmp_path: Path):
    per_seed = tmp_path / "results" / "figure2" / "per_seed"
    per_seed.mkdir(parents=True)
    path = per_seed / "duplicate.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "task",
            "seed",
            "model",
            "rollout_length",
            "error_mean",
            "rollout_repeats",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for value in (0.1, 0.2):
            writer.writerow(
                {
                    "task": "hopper-medium-replay-v2",
                    "seed": 0,
                    "model": "adm",
                    "rollout_length": 1,
                    "error_mean": value,
                    "rollout_repeats": 20,
                }
            )
    with pytest.raises(RuntimeError, match="repeats must be pooled"):
        plot_figure2(tmp_path)
