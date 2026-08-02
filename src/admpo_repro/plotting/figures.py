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
POLICY_LABELS = {"random": "random action", "learned": "learned policy", "behavior": "behavior (BC)"}
POLICY_COLORS = {"random": "#d62728", "learned": "#1f77b4", "behavior": "#2ca02c"}


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


def plot_figure2(root: Path, prefix: str = "") -> tuple[Path, Path]:
    result_dir = root / "results" / "figure2"
    paths = sorted((result_dir / "per_seed").glob("*.csv"))
    paths = _phase_paths(paths, prefix)
    rows = _read_csvs(paths)
    tasks = _present_tasks(rows, (
        "hopper-medium-v2",
        "hopper-medium-replay-v2",
        "walker2d-medium-v2",
        "walker2d-medium-replay-v2",
    ))
    if not tasks:
        raise RuntimeError(f"no Figure 2 CSV rows found for prefix {prefix!r}")
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["model"], int(row["rollout_length"]))].append(float(row["error_mean"]))
    summary_rows = []
    fig, axes_grid = plt.subplots(
        1, len(tasks), figsize=(3.9 * len(tasks), 3.6), sharey=False, squeeze=False
    )
    axes = axes_grid[0]
    for axis, task in zip(axes, tasks):
        task_steps: set[int] = set()
        for model in ("adm", "ensemble", "rnn"):
            steps = sorted(k[2] for k in grouped if k[0] == task and k[1] == model)
            if not steps:
                continue
            task_steps.update(steps)
            means, sems = [], []
            for step in steps:
                values = np.asarray(grouped[(task, model, step)], dtype=np.float64)
                mean = float(np.mean(values))
                sem = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
                means.append(max(mean, np.finfo(np.float32).tiny))
                sems.append(sem)
                summary_rows.append(
                    {"task": task, "model": model, "rollout_length": step, "mean": mean, "sem": sem, "n_seeds": values.size}
                )
            means_np = np.asarray(means)
            sem_np = np.asarray(sems)
            axis.plot(steps, means_np, color=MODEL_COLORS[model], linewidth=1.8, label=MODEL_LABELS[model])
            axis.fill_between(
                steps,
                np.maximum(means_np - sem_np, np.finfo(np.float32).tiny),
                means_np + sem_np,
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
    png = result_dir / f"{prefix}figure2.png"
    pdf = result_dir / f"{prefix}figure2.pdf"
    result_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    with (result_dir / f"{prefix}summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "model", "rollout_length", "mean", "sem", "n_seeds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return png, pdf


def _correlation(rows: list[dict[str, str]]) -> float:
    if len(rows) < 2:
        return float("nan")
    x = np.asarray([float(row["model_error"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["uncertainty"]) for row in rows], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return float("nan")
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def plot_figure4(root: Path, prefix: str = "") -> tuple[Path, Path]:
    result_dir = root / "results" / "figure4"
    paths = sorted((result_dir / "points").glob("*.csv"))
    paths = _phase_paths(paths, prefix)
    rows = _read_csvs(paths)
    rows = [row for row in rows if row["policy"] in POLICY_LABELS]
    tasks = _present_tasks(
        rows, ("hopper-medium-replay-v2", "walker2d-medium-replay-v2")
    )
    if not tasks:
        raise RuntimeError(f"no Figure 4 CSV rows found for prefix {prefix!r}")
    correlations = []
    for task in tasks:
        for model in ("adm", "ensemble"):
            selected = [row for row in rows if row["task"] == task and row["model"] == model]
            correlations.append({"task": task, "model": model, "seed": "pooled", "pearson_r": _correlation(selected), "n": len(selected)})
            for seed in sorted({int(row["seed"]) for row in selected}):
                seed_rows = [row for row in selected if int(row["seed"]) == seed]
                correlations.append({"task": task, "model": model, "seed": seed, "pearson_r": _correlation(seed_rows), "n": len(seed_rows)})
    panel_count = 2 * len(tasks)
    fig, axes_grid = plt.subplots(
        1, panel_count, figsize=(3.9 * panel_count, 3.7), squeeze=False
    )
    axes = axes_grid[0]
    position = 0
    for task in tasks:
        for model in ("adm", "ensemble"):
            axis = axes[position]
            position += 1
            selected = [row for row in rows if row["task"] == task and row["model"] == model]
            for policy in ("random", "learned", "behavior"):
                policy_rows = [row for row in selected if row["policy"] == policy]
                axis.scatter(
                    [float(row["model_error"]) for row in policy_rows],
                    [float(row["uncertainty"]) for row in policy_rows],
                    s=7,
                    alpha=0.28,
                    c=POLICY_COLORS[policy],
                    edgecolors="none",
                    label=POLICY_LABELS[policy],
                )
            r = _correlation(selected)
            axis.set_title(f"{MODEL_LABELS[model].split(' ')[0]} (Correlation: {r:.2f})", fontsize=10)
            axis.set_xlabel("Model Error")
            axis.set_ylabel("Model Uncertainty")
            axis.grid(True, alpha=0.16)
    for index, task in enumerate(tasks):
        center = (2 * index + 1) / panel_count
        fig.text(center, 0.115, task, ha="center", va="top", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    png = result_dir / f"{prefix}figure4.png"
    pdf = result_dir / f"{prefix}figure4.pdf"
    result_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    with (result_dir / f"{prefix}correlations.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "model", "seed", "pearson_r", "n"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(correlations)
    return png, pdf
