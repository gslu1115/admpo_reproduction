import csv
from pathlib import Path

from admpo_repro.plotting import plot_figure2, plot_figure4


def test_plotters_create_png_pdf_and_summaries(tmp_path: Path):
    fig2 = tmp_path / "results" / "figure2" / "per_seed"
    fig2.mkdir(parents=True)
    with (fig2 / "sample.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "seed", "model", "rollout_length", "error_mean", "error_std", "n_active"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in ("hopper-medium-v2", "hopper-medium-replay-v2", "walker2d-medium-v2", "walker2d-medium-replay-v2"):
            for model in ("adm", "ensemble", "rnn"):
                writer.writerow({"task": task, "seed": 0, "model": model, "rollout_length": 1, "error_mean": 0.1, "error_std": 0, "n_active": 2})
    assert all(path.exists() for path in plot_figure2(tmp_path))
    assert (tmp_path / "results" / "figure2" / "summary.csv").exists()

    fig4 = tmp_path / "results" / "figure4" / "points"
    fig4.mkdir(parents=True)
    with (fig4 / "sample.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "seed", "model", "policy", "rollout_step", "model_error", "uncertainty", "total_uncertainty"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in ("hopper-medium-replay-v2", "walker2d-medium-replay-v2"):
            for model in ("adm", "ensemble"):
                for policy in ("random", "learned", "behavior"):
                    for i in range(3):
                        writer.writerow({"task": task, "seed": 0, "model": model, "policy": policy, "rollout_step": 1, "model_error": i + 0.1, "uncertainty": i + 0.2, "total_uncertainty": i + 0.3})
    assert all(path.exists() for path in plot_figure4(tmp_path))
    assert (tmp_path / "results" / "figure4" / "correlations.csv").exists()
