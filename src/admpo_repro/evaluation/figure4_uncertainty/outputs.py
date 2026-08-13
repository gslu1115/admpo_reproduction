from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .rollout import EndpointBatch


PLOT_POLICY_ORDER = ("random", "learned_sac", "bc")
POLICY_LABELS = {
    "random": "random action",
    "bc": "dataset",
    "learned_sac": "learned policy",
}
# Color and marker encode policy together.  Dataset intentionally uses the
# lighter Okabe-Ito bluish-green requested for the final figure.
POLICY_COLORS = {"random": "#0072B2", "bc": "#56B4A9", "learned_sac": "#D55E00"}
POLICY_MARKERS = {"random": "o", "bc": "s", "learned_sac": "^"}


def _atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


@dataclass(frozen=True)
class ScalarSlice:
    dynamics_type: str
    model_seed: int
    policy: str
    horizon: int
    initial_window_ids: np.ndarray
    trajectory_ids: np.ndarray
    valid: np.ndarray
    uncertainty: np.ndarray
    error: np.ndarray


def scalar_slices(batch: EndpointBatch) -> list[ScalarSlice]:
    return [
        ScalarSlice(
            dynamics_type=batch.dynamics_type,
            model_seed=batch.model_seed,
            policy=batch.policy,
            horizon=int(horizon),
            initial_window_ids=batch.initial_window_ids,
            trajectory_ids=batch.trajectory_ids,
            valid=batch.valid_endpoint[index],
            uncertainty=batch.uncertainty[index],
            error=batch.model_error[index],
        )
        for index, horizon in enumerate(batch.horizons)
    ]


def sanity_check_batch(
    batch: EndpointBatch, source_sampling: str | None = None
) -> dict[str, Any]:
    batch.validate()
    valid = batch.valid_endpoint
    if not valid.any():
        raise RuntimeError("sanity audit received no valid endpoints")
    means = batch.all_predictor_means[valid].astype(np.float64)
    logvars = batch.all_predictor_logvars[valid].astype(np.float64)
    epistemic_variance = means.var(axis=1, ddof=0)
    manual_uncertainty = (
        epistemic_variance + np.exp(logvars).mean(axis=1)
    ).sum(axis=-1)
    if not np.allclose(manual_uncertainty, batch.uncertainty[valid], rtol=1e-5, atol=1e-7):
        raise RuntimeError("manual paper Eq. 4 uncertainty audit failed")
    truth = batch.true_next_state[valid].astype(np.float64)
    manual_model_error = (
        np.square(means - truth[:, None, :]) + np.exp(logvars)
    ).mean(axis=1).sum(axis=-1)
    if not np.allclose(
        manual_model_error, batch.model_error[valid], rtol=1e-5, atol=1e-7
    ):
        raise RuntimeError("manual Gaussian-mixture expected squared error audit failed")
    mixture_mean_bias_squared = np.square(means.mean(axis=1) - truth).sum(axis=-1)
    if not np.allclose(
        manual_model_error,
        manual_uncertainty + mixture_mean_bias_squared,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise RuntimeError("model-error/Eq. 4 decomposition audit failed")
    manual_sample = batch.selected_mean_next_state[valid] + np.exp(
        0.5 * batch.selected_logvar[valid]
    ) * batch.gaussian_epsilon[valid]
    valid_h_indexes, valid_window_indexes = np.nonzero(valid)
    valid_rollout_steps = batch.horizons[valid_h_indexes]
    recorded_sample = batch.sampled_next_state_schedule[
        valid_rollout_steps, valid_window_indexes
    ]
    if not np.allclose(
        manual_sample,
        recorded_sample,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("manual Gaussian propagation audit failed")
    if not np.allclose(
        recorded_sample,
        batch.sampled_next_state[valid],
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("endpoint Gaussian sample disagrees with rollout schedule")
    if np.array_equal(
        batch.selected_mean_next_state[valid], recorded_sample
    ):
        raise RuntimeError("Gaussian samples unexpectedly equal every selected mean")
    for h_index, horizon in enumerate(batch.horizons):
        endpoint_valid = batch.valid_endpoint[h_index]
        if not endpoint_valid.any():
            continue
        expected_endpoint_state = batch.sampled_next_state_schedule[
            int(horizon) - 1, endpoint_valid
        ]
        if not np.allclose(
            expected_endpoint_state,
            batch.endpoint_state[h_index, endpoint_valid],
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                f"sampled rollout state continuity failed at h={int(horizon)}"
            )
    selected = batch.chosen_source[valid]
    if not np.isin(selected, batch.source_ids).all():
        raise RuntimeError("selected predictor is outside the configured source set")
    source_position = {int(source_id): index for index, source_id in enumerate(batch.source_ids)}
    positions = np.asarray([source_position[int(source_id)] for source_id in selected])
    selected_from_all = means[np.arange(means.shape[0]), positions]
    if not np.allclose(
        selected_from_all,
        batch.selected_mean_next_state[valid],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("selected mean does not match the recorded predictor source")
    selected_source_counts = {}
    for h_index, horizon in enumerate(batch.horizons):
        selected_at_horizon = batch.chosen_source[h_index, batch.valid_endpoint[h_index]]
        ids, counts = np.unique(selected_at_horizon, return_counts=True)
        selected_source_counts[str(int(horizon))] = {
            str(int(source_id)): int(count)
            for source_id, count in zip(ids, counts)
        }
        if not np.array_equal(
            selected_at_horizon,
            batch.rollout_source_schedule[int(horizon), batch.valid_endpoint[h_index]],
        ):
            raise RuntimeError(
                f"recorded endpoint source disagrees with rollout schedule at h={int(horizon)}"
            )
    rollout_source_counts = {}
    for rollout_step in range(batch.rollout_source_schedule.shape[0]):
        selected_at_step = batch.rollout_source_schedule[rollout_step]
        selected_at_step = selected_at_step[selected_at_step >= 0]
        ids, counts = np.unique(selected_at_step, return_counts=True)
        rollout_source_counts[str(rollout_step)] = {
            str(int(source_id)): int(count)
            for source_id, count in zip(ids, counts)
        }
        if (
            batch.dynamics_type == "adm"
            and source_sampling == "vendor_batch_shared"
            and selected_at_step.size > 0
            and ids.size != 1
        ):
            raise RuntimeError(
                "vendor batch-shared ADM sampling selected more than one source "
                f"at rollout step {rollout_step}"
            )
    for h_index, horizon in enumerate(batch.horizons):
        later = batch.terminal_step >= 0
        later &= batch.terminal_step <= int(horizon)
        if np.any(later & batch.valid_endpoint[h_index]):
            raise RuntimeError("a terminated trajectory contributed a later endpoint")
    return {
        "dynamics_type": batch.dynamics_type,
        "model_seed": batch.model_seed,
        "policy": batch.policy,
        "source_ids": batch.source_ids.tolist(),
        "source_sampling": source_sampling or "unspecified",
        "selected_source_counts": selected_source_counts,
        "rollout_source_counts": rollout_source_counts,
        "valid_counts": {
            str(int(h)): int(batch.valid_endpoint[index].sum())
            for index, h in enumerate(batch.horizons)
        },
        "uncertainty_formula_checked": True,
        "mixture_expected_squared_error_checked": True,
        "model_error_eq4_decomposition_checked": True,
        "selected_predictor_identity_checked": True,
        "gaussian_sample_checked": True,
        "rollout_state_continuity_checked": True,
        "termination_order_checked": True,
    }


def _combined_plot_data(
    slices: list[ScalarSlice], dynamics_type: str, horizon: int, policy: str
) -> list[tuple[int, int, int, float, float]]:
    rows = []
    for item in slices:
        if item.dynamics_type != dynamics_type or item.horizon != horizon or item.policy != policy:
            continue
        for index in np.flatnonzero(item.valid):
            rows.append(
                (
                    item.model_seed,
                    int(item.trajectory_ids[index]),
                    int(item.initial_window_ids[index]),
                    float(item.error[index]),
                    float(item.uncertainty[index]),
                )
            )
    return rows


def _pooled_pearson(error: np.ndarray, uncertainty: np.ndarray) -> float:
    if error.size < 2 or float(np.std(error)) == 0.0 or float(np.std(uncertainty)) == 0.0:
        raise RuntimeError("pooled plot correlation is undefined")
    return float(np.corrcoef(uncertainty, error)[0, 1])


def write_final_figure(
    slices: list[ScalarSlice],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    all_errors = np.concatenate([item.error[item.valid] for item in slices])
    all_uncertainties = np.concatenate([item.uncertainty[item.valid] for item in slices])
    if not np.isfinite(all_errors).all() or not np.isfinite(all_uncertainties).all():
        raise RuntimeError("non-finite data reached plotting")
    plot_cfg = config["plot"]
    x_limits = tuple(float(value) for value in plot_cfg["x_limits"])
    y_limits = tuple(float(value) for value in plot_cfg["y_limits"])
    formats = list(plot_cfg["formats"])
    if x_limits != (0.0, 1.5) or y_limits != (0.0, 1.0):
        raise RuntimeError("Figure 4 requires fixed paper-matched axes x=[0,1.5], y=[0,1]")
    if formats != ["pdf"]:
        raise RuntimeError("Figure 4 image output must contain PDF only")
    dpi = int(plot_cfg["dpi"])
    width = math.ceil(float(plot_cfg["width_mm"]) / 25.4 * dpi) / dpi
    height = math.ceil(float(plot_cfg["height_mm"]) / 25.4 * dpi) / dpi

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(
            1, 2, figsize=(width, height), layout="constrained", squeeze=False
        )
        source_rows: list[dict[str, Any]] = []
        correlations: dict[str, float] = {}
        horizon = int(config["protocol"]["horizons"][0])
        for axis, dynamics_type in zip(axes[0], ("adm", "ensemble")):
            display = "ADM" if dynamics_type == "adm" else "Ensemble"
            pooled_error, pooled_uncertainty = [], []
            for policy in PLOT_POLICY_ORDER:
                rows = _combined_plot_data(slices, dynamics_type, horizon, policy)
                if not rows:
                    raise RuntimeError(f"no plot data for {dynamics_type}/{policy}/h{horizon}")
                error = np.asarray([row[3] for row in rows], dtype=np.float64)
                uncertainty = np.asarray([row[4] for row in rows], dtype=np.float64)
                axis.scatter(
                    error,
                    uncertainty,
                    s=10,
                    alpha=0.78,
                    color=POLICY_COLORS[policy],
                    marker=POLICY_MARKERS[policy],
                    edgecolors="none",
                    rasterized=True,
                    label=POLICY_LABELS[policy],
                )
                pooled_error.extend(error)
                pooled_uncertainty.extend(uncertainty)
                source_rows.extend(
                    {
                        "dynamics_type": dynamics_type,
                        "horizon": horizon,
                        "policy": policy,
                        "model_seed": row[0],
                        "trajectory_id": row[1],
                        "initial_window_id": row[2],
                        "model_error": row[3],
                        "model_uncertainty": row[4],
                    }
                    for row in rows
                )
            pearson = _pooled_pearson(
                np.asarray(pooled_error), np.asarray(pooled_uncertainty)
            )
            correlations[dynamics_type] = pearson
            axis.set_title(f"{display} (Correlation: {pearson:.2f})")
            axis.set_xlabel("Model Error")
            axis.set_ylabel("Model Uncertainty")
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_xticks(np.arange(0.0, 1.51, 0.25))
            axis.set_yticks(np.arange(0.0, 1.01, 0.2))
            axis.xaxis.set_major_formatter("{x:.2f}")
            axis.yaxis.set_major_formatter("{x:.1f}")
            axis.grid(False)
            axis.legend(frameon=False, loc="best", labelspacing=0.3)
        pdf = output_dir / "Hopper.pdf"
        fig.savefig(pdf, dpi=dpi, facecolor="white", transparent=False)
        plt.close(fig)
    source_path = output_dir / "Hopper.csv"
    _atomic_csv(source_path, list(source_rows[0]), source_rows)
    return {
        "pdf": str(pdf),
        "source_data": str(source_path),
        "horizon": horizon,
        "pooled_pearson_all_valid": correlations,
    }
