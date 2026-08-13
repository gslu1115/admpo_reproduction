from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "adm": "ADM (ours)",
    "ensemble": "Ensemble Dynamics Model",
    "rnn": "Bootstrapping RNN Dynamics Model",
}
MODEL_COLORS = {"adm": "#d62728", "ensemble": "#1f77b4", "rnn": "#2ca02c"}


def _read_csvs(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _phase_paths(paths: list[Path], prefix: str) -> list[Path]:
    if prefix:
        return [path for path in paths if path.name.startswith(prefix)]
    reserved = ("smoke_", "deadline48h_")
    return [path for path in paths if not path.name.startswith(reserved)]


def _present_tasks(rows: list[dict[str, str]], preferred: tuple[str, ...]) -> list[str]:
    present = {row["task"] for row in rows}
    return [task for task in preferred if task in present]


def plot_figure2(
    root: Path, prefix: str = "", result_dir: Path | None = None
) -> tuple[Path, Path]:
    result_dir = root / "results" / "figure2" if result_dir is None else result_dir
    paths = _phase_paths(sorted((result_dir / "per_seed").glob("*.csv")), prefix)
    rows = _read_csvs(paths)
    tasks = _present_tasks(
        rows,
        (
            "hopper-medium-v2",
            "hopper-medium-replay-v2",
            "walker2d-medium-v2",
            "walker2d-medium-replay-v2",
        ),
    )
    if not tasks:
        raise RuntimeError(f"no Figure 2 CSV rows found for prefix {prefix!r}")

    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    repeat_counts: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    seed_values: dict[tuple[str, str, int, int], float] = {}
    for row in rows:
        group_key = (row["task"], row["model"], int(row["rollout_length"]))
        seed_key = (*group_key, int(row["seed"]))
        if seed_key in seed_values:
            raise RuntimeError(
                "Figure 2 per_seed CSV contains duplicate rows for "
                f"{seed_key}; rollout repeats must be pooled before plotting"
            )
        seed_values[seed_key] = float(row["error_mean"])
        grouped[group_key].append(seed_values[seed_key])
        repeat_counts[group_key].add(int(row.get("rollout_repeats") or 1))

    summary_rows = []
    fig, axes_grid = plt.subplots(
        1, len(tasks), figsize=(3.9 * len(tasks), 3.6), sharey=False, squeeze=False
    )
    axes = axes_grid[0]
    for axis, task in zip(axes, tasks):
        task_steps: set[int] = set()
        for model in ("adm", "ensemble", "rnn"):
            steps = sorted(
                key[2] for key in grouped if key[0] == task and key[1] == model
            )
            if not steps:
                continue
            task_steps.update(steps)
            means, sems = [], []
            for step in steps:
                values = np.asarray(grouped[(task, model, step)], dtype=np.float64)
                configured_repeats = repeat_counts[(task, model, step)]
                if len(configured_repeats) != 1:
                    raise RuntimeError(
                        f"inconsistent rollout repeat counts for {task}/{model}/step-{step}"
                    )
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                sem = std / math.sqrt(values.size) if values.size > 1 else 0.0
                means.append(max(mean, np.finfo(np.float32).tiny))
                sems.append(sem)
                summary_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "rollout_length": step,
                        "mean": mean,
                        "std": std,
                        "sem": sem,
                        "n_seeds": values.size,
                        "rollout_repeats": next(iter(configured_repeats)),
                    }
                )
            means_array = np.asarray(means)
            sem_array = np.asarray(sems)
            axis.plot(
                steps,
                means_array,
                color=MODEL_COLORS[model],
                linewidth=1.8,
                label=MODEL_LABELS[model],
            )
            axis.fill_between(
                steps,
                np.maximum(means_array - sem_array, np.finfo(np.float32).tiny),
                means_array + sem_array,
                color=MODEL_COLORS[model],
                alpha=0.18,
                linewidth=0,
            )
        axis.set_title(task.replace("-v2", ""), fontsize=10)
        axis.set_xlabel("Roll-out Length")
        axis.set_ylabel("Prediction Error")
        axis.set_yscale("log")
        if task_steps:
            axis.set_xlim(0, max(task_steps))
        axis.grid(True, which="both", alpha=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    result_dir.mkdir(parents=True, exist_ok=True)
    png = result_dir / f"{prefix}figure2.png"
    pdf = result_dir / f"{prefix}figure2.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    fields = [
        "task",
        "model",
        "rollout_length",
        "mean",
        "std",
        "sem",
        "n_seeds",
        "rollout_repeats",
    ]
    with (result_dir / f"{prefix}summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return png, pdf
