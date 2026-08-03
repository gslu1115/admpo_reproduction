from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from admpo_repro.data import D4RLDataset, TrajectorySplit
from admpo_repro.dynamics.models import ADMDynamics, EnsembleDynamics

from .oracle import MujocoOracle, termination


ActionFunction = Callable[[np.ndarray], np.ndarray]
POLICY_SOURCES = ("random", "learned", "dataset")


@dataclass(frozen=True)
class RolloutBatch:
    """A test-trajectory window shared by both dynamics-model families."""

    observations: np.ndarray
    previous_actions: np.ndarray
    dataset_actions: np.ndarray
    window_starts: np.ndarray


@dataclass(frozen=True)
class PredictionStatistics:
    """Common scalarization of ADM backtracks or ensemble elites."""

    mean_prediction: np.ndarray
    uncertainty_std: np.ndarray
    uncertainty_var: np.ndarray
    total_uncertainty_std: np.ndarray
    total_uncertainty_var: np.ndarray
    component_means: np.ndarray
    component_stds: np.ndarray


def build_rollout_batch(
    dataset: D4RLDataset,
    split: TrajectorySplit,
    starts: int,
    history: int,
    horizon: int,
    seed: int,
) -> RolloutBatch:
    """Sample windows wholly contained in held-out trajectories.

    A window contains ``history`` states, the ``history - 1`` actions between
    them, and ``horizon`` future actions.  The latter are used directly by the
    dataset-action-sequence condition.
    """

    length = history + horizon - 1
    pool = split.valid_starts("test", length)
    if pool.size == 0:
        raise RuntimeError(
            f"test split has no Figure 4 window with {history=} and {horizon=}"
        )
    rng = np.random.default_rng(seed)
    indexes = rng.choice(pool, size=starts, replace=pool.size < starts)
    sequence = dataset.sequences(indexes, length)
    observations = sequence["observations"][:, :history]
    previous_actions = sequence["actions"][:, : history - 1]
    dataset_actions = sequence["actions"][:, history - 1 : history - 1 + horizon]
    return RolloutBatch(
        observations=observations.astype(np.float32, copy=False),
        previous_actions=previous_actions.astype(np.float32, copy=False),
        dataset_actions=dataset_actions.astype(np.float32, copy=False),
        window_starts=np.asarray(indexes, dtype=np.int64),
    )


@torch.no_grad()
def prediction_statistics(
    kind: str,
    model: ADMDynamics | EnsembleDynamics,
    obs_hist: np.ndarray,
    act_hist: np.ndarray,
    action: np.ndarray,
    device: str,
) -> PredictionStatistics:
    """Evaluate every backtrack/elite on the same current model input.

    The primary uncertainty is RMS disagreement of predictive means.  The
    total diagnostic additionally includes the learned Gaussian variance.
    Population variance (``correction=0``) matches NumPy's ``var`` convention
    in the fixed ADMPO implementation.
    """

    current = torch.as_tensor(obs_hist[:, -1], device=device)
    action_t = torch.as_tensor(action, device=device)
    if kind == "adm":
        all_actions = np.concatenate((act_hist, action[:, None]), axis=1)
        action_history = torch.as_tensor(all_actions, device=device)
        means, stds = [], []
        obs_tensor = torch.as_tensor(obs_hist, device=device)
        for k in range(1, obs_hist.shape[1] + 1):
            mean, std, _, _ = model.dyna_dist(
                obs_tensor[:, -k], action_history[:, -k:]
            )
            means.append(mean)
            stds.append(std)
        mean_stack = torch.stack(means)
        std_stack = torch.stack(stds)
    elif kind == "ensemble":
        mean_stack, std_stack, _, _ = model.dyna_dist(current, action_t)
    else:
        raise ValueError(kind)

    mean_prediction = mean_stack.mean(dim=0)
    state_epistemic_var = mean_stack.var(dim=0, correction=0).clamp_min(0)
    uncertainty_var = state_epistemic_var.mean(dim=-1)
    uncertainty_std = torch.sqrt(uncertainty_var)
    state_total_var = state_epistemic_var + std_stack.square().mean(dim=0)
    total_uncertainty_var = state_total_var.mean(dim=-1).clamp_min(0)
    total_uncertainty_std = torch.sqrt(total_uncertainty_var)
    return PredictionStatistics(
        mean_prediction=mean_prediction.cpu().numpy(),
        uncertainty_std=uncertainty_std.cpu().numpy(),
        uncertainty_var=uncertainty_var.cpu().numpy(),
        total_uncertainty_std=total_uncertainty_std.cpu().numpy(),
        total_uncertainty_var=total_uncertainty_var.cpu().numpy(),
        component_means=mean_stack.cpu().numpy(),
        component_stds=std_stack.cpu().numpy(),
    )


def sample_next_state(
    statistics: PredictionStatistics, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one component Gaussian per rollout trajectory."""

    components, batch, _ = statistics.component_means.shape
    choices = rng.integers(0, components, size=batch)
    columns = np.arange(batch)
    means = statistics.component_means[choices, columns]
    stds = np.nan_to_num(
        statistics.component_stds[choices, columns],
        nan=1e-6,
        posinf=1e-6,
        neginf=1e-6,
    )
    sampled = means + rng.normal(size=means.shape) * np.maximum(stds, 1e-6)
    return sampled.astype(np.float32), choices.astype(np.int64)


def _actions_for_step(
    policy_name: str,
    policy: ActionFunction | None,
    model_observation: np.ndarray,
    dataset_actions: np.ndarray,
    rollout_step: int,
    env: object,
    seed: int,
) -> np.ndarray:
    if policy_name == "random":
        rng = np.random.default_rng(seed)
        action = rng.uniform(
            env.action_space.low,
            env.action_space.high,
            size=(model_observation.shape[0], env.action_space.shape[0]),
        )
    elif policy_name == "learned":
        if policy is None:
            raise ValueError("learned policy callable is required")
        action = policy(model_observation)
    elif policy_name == "dataset":
        action = dataset_actions[:, rollout_step]
    else:
        raise ValueError(f"unknown Figure 4 action source: {policy_name}")
    return np.clip(
        np.asarray(action, dtype=np.float32),
        env.action_space.low,
        env.action_space.high,
    ).astype(np.float32, copy=False)


def _squared_state_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.square(prediction - target), axis=-1, dtype=np.float64)


def collect_rollout_points(
    task: str,
    kind: str,
    model: ADMDynamics | EnsembleDynamics,
    policy_name: str,
    policy: ActionFunction | None,
    dataset: D4RLDataset,
    split: TrajectorySplit,
    env: object,
    seed: int,
    config: dict,
) -> list[dict]:
    """Collect cumulative-trajectory and local one-step calibration points."""

    if policy_name not in POLICY_SOURCES:
        raise ValueError(policy_name)
    cfg = config["evaluation"]
    device = config["device"]
    history = int(config["model"]["max_backtrack"])
    target_points = int(cfg["points_per_policy"])
    starts = int(cfg["starts"])
    horizon = int(cfg["rollout_horizon"])
    sampling_seed = int(cfg["sampling_seed"])
    oracle = MujocoOracle(env)
    rows: list[dict] = []
    attempts = 0
    max_attempts = int(cfg.get("max_collection_attempts", 20))
    model_offset = {"adm": 11, "ensemble": 17}[kind]
    source_offset = {"random": 101, "learned": 211, "dataset": 307}[policy_name]

    while len(rows) < target_points and attempts < max_attempts:
        batch = build_rollout_batch(
            dataset,
            split,
            starts,
            history,
            horizon,
            sampling_seed + seed * 1009 + attempts * 104729,
        )
        obs_hist = batch.observations.copy()
        act_hist = batch.previous_actions.copy()
        real_observation = obs_hist[:, -1].copy()
        alive = np.ones(starts, dtype=bool)

        for rollout_step in range(horizon):
            action_seed = (
                sampling_seed
                + seed * 1_000_003
                + source_offset * 10_007
                + attempts * 10_000_019
                + rollout_step * 100_003
            )
            action = _actions_for_step(
                policy_name,
                policy,
                obs_hist[:, -1],
                batch.dataset_actions,
                rollout_step,
                env,
                action_seed,
            )
            statistics = prediction_statistics(
                kind, model, obs_hist, act_hist, action, device
            )
            rollout_rng = np.random.default_rng(
                action_seed + model_offset * 1_000_000_007
            )
            predicted, component_choices = sample_next_state(statistics, rollout_rng)
            active_ids = np.where(alive)[0]
            real_next, _, real_done = oracle.batch_step(
                real_observation[active_ids], action[active_ids]
            )
            local_truth, _, _ = oracle.batch_step(
                obs_hist[active_ids, -1], action[active_ids]
            )
            trajectory_mse = _squared_state_error(
                predicted[active_ids], real_next
            )
            local_mse = _squared_state_error(
                statistics.mean_prediction[active_ids], local_truth
            )
            trajectory_rmse = np.sqrt(trajectory_mse)
            local_rmse = np.sqrt(local_mse)

            for local_index, batch_index in enumerate(active_ids):
                values = (
                    trajectory_mse[local_index],
                    local_mse[local_index],
                    statistics.uncertainty_std[batch_index],
                    statistics.total_uncertainty_std[batch_index],
                )
                if not np.isfinite(values).all():
                    continue
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "model": kind,
                        "policy": policy_name,
                        "window_start": int(batch.window_starts[batch_index]),
                        "rollout_attempt": attempts,
                        "rollout_step": rollout_step + 1,
                        "component_index": int(component_choices[batch_index]),
                        "trajectory_error_rmse": float(
                            trajectory_rmse[local_index]
                        ),
                        "trajectory_error_mse": float(trajectory_mse[local_index]),
                        "local_error_rmse": float(local_rmse[local_index]),
                        "local_error_mse": float(local_mse[local_index]),
                        "uncertainty_std": float(
                            statistics.uncertainty_std[batch_index]
                        ),
                        "uncertainty_var": float(
                            statistics.uncertainty_var[batch_index]
                        ),
                        "total_uncertainty_std": float(
                            statistics.total_uncertainty_std[batch_index]
                        ),
                        "total_uncertainty_var": float(
                            statistics.total_uncertainty_var[batch_index]
                        ),
                    }
                )
                if len(rows) >= target_points:
                    break
            if len(rows) >= target_points:
                break

            invalid_model = ~np.isfinite(predicted).all(axis=-1)
            invalid_real = ~np.isfinite(real_next).all(axis=-1)
            ended = (
                termination(task, predicted[active_ids])
                | real_done.reshape(-1)
                | invalid_model[active_ids]
                | invalid_real
            )
            alive[active_ids[ended]] = False
            real_observation[active_ids] = real_next
            obs_hist = np.concatenate((obs_hist[:, 1:], predicted[:, None]), axis=1)
            act_hist = np.concatenate((act_hist[:, 1:], action[:, None]), axis=1)
            if not alive.any():
                break
        attempts += 1

    if len(rows) < target_points:
        raise RuntimeError(
            f"collected only {len(rows)}/{target_points} valid points for "
            f"{task}/{kind}/{policy_name}"
        )
    return rows[:target_points]


def collect_dataset_diagnostic(
    task: str,
    kind: str,
    model: ADMDynamics | EnsembleDynamics,
    dataset: D4RLDataset,
    split: TrajectorySplit,
    env: object,
    seed: int,
    config: dict,
) -> list[dict]:
    """Evaluate local calibration directly on held-out D4RL transitions."""

    history = int(config["model"]["max_backtrack"])
    points = int(config["evaluation"]["dataset_diagnostic_points"])
    pool = split.valid_starts("test", history)
    rng = np.random.default_rng(
        int(config["evaluation"]["sampling_seed"])
        + seed * 3571
        + (1 if kind == "adm" else 2)
    )
    starts = rng.choice(pool, size=points, replace=pool.size < points)
    sequence = dataset.sequences(starts, history)
    obs_hist = sequence["observations"]
    act_hist = sequence["actions"][:, :-1]
    action = sequence["actions"][:, -1]
    statistics = prediction_statistics(
        kind, model, obs_hist, act_hist, action, config["device"]
    )
    truth, _, _ = MujocoOracle(env).batch_step(obs_hist[:, -1], action)
    local_mse = _squared_state_error(statistics.mean_prediction, truth)
    rows = []
    for index, error_mse in enumerate(local_mse):
        values = (error_mse, statistics.uncertainty_std[index])
        if not np.isfinite(values).all():
            continue
        rows.append(
            {
                "task": task,
                "seed": seed,
                "model": kind,
                "policy": "dataset-direct",
                "window_start": int(starts[index]),
                "rollout_attempt": 0,
                "rollout_step": 0,
                "component_index": -1,
                "trajectory_error_rmse": float(np.sqrt(error_mse)),
                "trajectory_error_mse": float(error_mse),
                "local_error_rmse": float(np.sqrt(error_mse)),
                "local_error_mse": float(error_mse),
                "uncertainty_std": float(statistics.uncertainty_std[index]),
                "uncertainty_var": float(statistics.uncertainty_var[index]),
                "total_uncertainty_std": float(
                    statistics.total_uncertainty_std[index]
                ),
                "total_uncertainty_var": float(
                    statistics.total_uncertainty_var[index]
                ),
            }
        )
    return rows


FIGURE4_FIELDS = [
    "task",
    "seed",
    "model",
    "policy",
    "window_start",
    "rollout_attempt",
    "rollout_step",
    "component_index",
    "trajectory_error_rmse",
    "trajectory_error_mse",
    "local_error_rmse",
    "local_error_mse",
    "uncertainty_std",
    "uncertainty_var",
    "total_uncertainty_std",
    "total_uncertainty_var",
]


def write_figure4_rows(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIGURE4_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
