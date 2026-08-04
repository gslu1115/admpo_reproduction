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
POLICY_LABELS = {
    "random": "random action",
    "learned": "learned policy",
    "dataset": "dataset",
}
POLICY_COLORS = {
    "random": "#4c72b0",
    "learned": "#55a868",
    "dataset": "#c44e52",
}


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
            steps = sorted(k[2] for k in grouped if k[0] == task and k[1] == model)
            if not steps:
                continue
            task_steps.update(steps)
            means, stds, sems = [], [], []
            for step in steps:
                values = np.asarray(grouped[(task, model, step)], dtype=np.float64)
                configured_repeats = repeat_counts[(task, model, step)]
                if len(configured_repeats) != 1:
                    raise RuntimeError(
                        f"inconsistent rollout repeat counts for {task}/{model}/step-{step}"
                    )
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                sem = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
                means.append(max(mean, np.finfo(np.float32).tiny))
                stds.append(std)
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return png, pdf


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks without adding a SciPy dependency."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation_values(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), int(x.size)
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])
    return pearson, spearman, int(x.size)


def _row_values(
    rows: list[dict[str, str]], x_field: str, y_field: str
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([float(row[x_field]) for row in rows], dtype=np.float64),
        np.asarray([float(row[y_field]) for row in rows], dtype=np.float64),
    )


def _centered_values(
    rows: list[dict[str, str]],
    x_field: str,
    y_field: str,
    group_fields: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    x, y = _row_values(rows, x_field, y_field)
    keys = [tuple(row[field] for field in group_fields) for row in rows]
    for key in set(keys):
        mask = np.asarray([candidate == key for candidate in keys], dtype=bool)
        x[mask] -= np.mean(x[mask])
        y[mask] -= np.mean(y[mask])
    return x, y


def _statistic_row(
    task: str,
    model: str,
    seed: str | int,
    scope: str,
    policy: str,
    metric_name: str,
    error_field: str,
    uncertainty_field: str,
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    pearson, spearman, count = _correlation_values(x, y)
    return {
        "task": task,
        "model": model,
        "seed": seed,
        "scope": scope,
        "policy": policy,
        "metric": metric_name,
        "error_field": error_field,
        "uncertainty_field": uncertainty_field,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "n": count,
    }


def plot_figure4(
    root: Path, prefix: str = "", result_dir: Path | None = None
) -> tuple[Path, Path]:
    result_dir = root / "results" / "figure4" if result_dir is None else result_dir
    paths = sorted((result_dir / "points").glob("*.csv"))
    paths = _phase_paths(paths, prefix)
    rows = _read_csvs(paths)
    rows = [row for row in rows if row["policy"] in POLICY_LABELS]
    tasks = _present_tasks(
        rows, ("hopper-medium-replay-v2", "walker2d-medium-replay-v2")
    )
    if not tasks:
        raise RuntimeError(f"no Figure 4 CSV rows found for prefix {prefix!r}")
    metric_pairs = (
        (
            "trajectory-rmse_std",
            "trajectory_error_rmse",
            "uncertainty_std",
        ),
        ("local-rmse_std", "local_error_rmse", "uncertainty_std"),
        (
            "trajectory-mse_variance",
            "trajectory_error_mse",
            "uncertainty_var",
        ),
        ("local-mse_variance", "local_error_mse", "uncertainty_var"),
        (
            "trajectory-rmse_total-std",
            "trajectory_error_rmse",
            "total_uncertainty_std",
        ),
        (
            "local-rmse_total-std",
            "local_error_rmse",
            "total_uncertainty_std",
        ),
    )
    correlations: list[dict] = []
    policy_summary: list[dict] = []
    for task in tasks:
        for model in ("adm", "ensemble"):
            selected = [row for row in rows if row["task"] == task and row["model"] == model]
            seed_scopes: list[tuple[str | int, list[dict[str, str]]]] = [
                ("pooled", selected)
            ]
            seed_scopes.extend(
                (seed, [row for row in selected if int(row["seed"]) == seed])
                for seed in sorted({int(row["seed"]) for row in selected})
            )
            for seed_label, seed_rows in seed_scopes:
                for metric_name, error_field, uncertainty_field in metric_pairs:
                    x, y = _row_values(seed_rows, error_field, uncertainty_field)
                    correlations.append(
                        _statistic_row(
                            task,
                            model,
                            seed_label,
                            "pooled",
                            "all",
                            metric_name,
                            error_field,
                            uncertainty_field,
                            x,
                            y,
                        )
                    )
                    for policy in POLICY_LABELS:
                        policy_rows = [
                            row for row in seed_rows if row["policy"] == policy
                        ]
                        policy_x, policy_y = _row_values(
                            policy_rows, error_field, uncertainty_field
                        )
                        correlations.append(
                            _statistic_row(
                                task,
                                model,
                                seed_label,
                                "policy",
                                policy,
                                metric_name,
                                error_field,
                                uncertainty_field,
                                policy_x,
                                policy_y,
                            )
                        )
                for scope, group_fields in (
                    ("within-policy-centered", ("policy",)),
                    ("within-step-centered", ("rollout_step",)),
                    (
                        "within-policy-step-centered",
                        ("policy", "rollout_step"),
                    ),
                ):
                    x, y = _centered_values(
                        seed_rows,
                        "trajectory_error_rmse",
                        "uncertainty_std",
                        group_fields,
                    )
                    correlations.append(
                        _statistic_row(
                            task,
                            model,
                            seed_label,
                            scope,
                            "all",
                            "trajectory-rmse_std",
                            "trajectory_error_rmse",
                            "uncertainty_std",
                            x,
                            y,
                        )
                    )
            for policy in POLICY_LABELS:
                policy_rows = [row for row in selected if row["policy"] == policy]
                for field in (
                    "trajectory_error_rmse",
                    "local_error_rmse",
                    "uncertainty_std",
                ):
                    values = np.asarray(
                        [float(row[field]) for row in policy_rows], dtype=np.float64
                    )
                    policy_summary.append(
                        {
                            "task": task,
                            "model": model,
                            "policy": policy,
                            "field": field,
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values, ddof=1))
                            if values.size > 1
                            else 0.0,
                            "n": int(values.size),
                        }
                    )
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
            for policy in POLICY_LABELS:
                policy_rows = [row for row in selected if row["policy"] == policy]
                axis.scatter(
                    [float(row["trajectory_error_rmse"]) for row in policy_rows],
                    [float(row["uncertainty_std"]) for row in policy_rows],
                    s=7,
                    alpha=0.38,
                    c=POLICY_COLORS[policy],
                    edgecolors="none",
                    label=POLICY_LABELS[policy],
                )
            x, y = _row_values(
                selected, "trajectory_error_rmse", "uncertainty_std"
            )
            r, _, _ = _correlation_values(x, y)
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
        fields = [
            "task",
            "model",
            "seed",
            "scope",
            "policy",
            "metric",
            "error_field",
            "uncertainty_field",
            "pearson_r",
            "spearman_r",
            "n",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(correlations)
    with (result_dir / f"{prefix}policy_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["task", "model", "policy", "field", "mean", "std", "n"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(policy_summary)
    return png, pdf
