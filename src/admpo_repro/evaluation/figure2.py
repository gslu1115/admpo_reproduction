from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from admpo_repro.data import D4RLDataset, TrajectorySplit
from admpo_repro.dynamics.models import ADMDynamics, EnsembleDynamics, RNNDynamics


@dataclass(frozen=True)
class Figure2Windows:
    """Common held-out windows shared by every model and training seed."""

    starts: np.ndarray
    observation_history: np.ndarray
    action_history: np.ndarray
    future_actions: np.ndarray
    true_next_observations: np.ndarray


def build_test_windows(
    dataset: D4RLDataset,
    split: TrajectorySplit,
    history: int,
    horizon: int,
    count: int,
    seed: int,
) -> Figure2Windows:
    """Sample fixed windows wholly contained in the held-out test trajectories."""
    if history < 1 or horizon < 1 or count < 1:
        raise ValueError("history, horizon, and count must all be positive")
    required_transitions = history - 1 + horizon
    pool = split.valid_starts("test", required_transitions)
    if pool.size == 0:
        raise RuntimeError(
            f"test split has no window with {required_transitions} consecutive transitions"
        )
    rng = np.random.default_rng(seed)
    starts = rng.choice(pool, size=count, replace=pool.size < count).astype(np.int64)
    obs_idx = starts[:, None] + np.arange(history, dtype=np.int64)[None, :]
    if history > 1:
        past_action_idx = starts[:, None] + np.arange(history - 1, dtype=np.int64)[None, :]
        action_history = dataset.actions[past_action_idx]
    else:
        action_history = np.empty((count, 0, dataset.action_dim), dtype=np.float32)
    future_idx = (
        starts[:, None]
        + (history - 1)
        + np.arange(horizon, dtype=np.int64)[None, :]
    )
    return Figure2Windows(
        starts=starts,
        observation_history=dataset.observations[obs_idx].astype(np.float32, copy=True),
        action_history=action_history.astype(np.float32, copy=True),
        future_actions=dataset.actions[future_idx].astype(np.float32, copy=True),
        true_next_observations=dataset.next_observations[future_idx].astype(np.float32, copy=True),
    )


@torch.no_grad()
def _sample_prediction_step(
    kind: str,
    model: ADMDynamics | RNNDynamics | EnsembleDynamics,
    observation_history: np.ndarray,
    action_history: np.ndarray,
    action: np.ndarray,
    device: str,
    max_backtrack: int,
    backtrack_rng: np.random.Generator,
    noise_rng: np.random.Generator,
    elite_rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Sample one stochastic model step using the official rollout semantics.

    ADM and Bootstrapping RNN share a scalar ``k`` sampled uniformly for the
    current batched rollout step.  Ensemble selects an elite independently for
    every trajectory.  Every family then samples its predicted Gaussian.
    """

    obs_t = torch.as_tensor(observation_history, device=device)
    action_t = torch.as_tensor(action, device=device)
    all_actions = np.concatenate((action_history, action[:, None]), axis=1)
    actions_t = torch.as_tensor(all_actions, device=device)
    if kind in ("adm", "rnn"):
        available = min(
            int(max_backtrack), observation_history.shape[1], all_actions.shape[1]
        )
        k = int(backtrack_rng.integers(1, available + 1))
        if kind == "adm":
            mean, std, _, _ = model.dyna_dist(
                obs_t[:, -k], actions_t[:, -k:]
            )
        else:
            mean, std, _, _ = model.dyna_dist(
                obs_t[:, -k:], actions_t[:, -k:]
            )
    elif kind == "ensemble":
        member_means, member_stds, _, _ = model.dyna_dist(
            obs_t[:, -1], action_t
        )
        batch = member_means.shape[1]
        choices = elite_rng.integers(0, member_means.shape[0], size=batch)
        columns = torch.arange(batch, device=member_means.device)
        choices_t = torch.as_tensor(choices, device=member_means.device)
        mean = member_means[choices_t, columns]
        std = member_stds[choices_t, columns]
        k = 1
    else:
        raise ValueError(f"unknown dynamics kind: {kind}")
    mean_np = mean.detach().cpu().numpy()
    std_np = std.detach().cpu().numpy()
    with np.errstate(over="ignore", invalid="ignore"):
        sampled = mean_np + noise_rng.normal(size=mean_np.shape) * std_np
    return sampled.astype(np.float32), k


def evaluate_model_rollout(
    task: str,
    kind: str,
    model: ADMDynamics | RNNDynamics | EnsembleDynamics,
    dataset: D4RLDataset,
    split: TrajectorySplit,
    seed: int,
    config: dict,
    windows: Figure2Windows | None = None,
) -> list[dict[str, float | int | str]]:
    """Evaluate autoregressive prediction against held-out recorded trajectories.

    Actions always come from the D4RL test trajectory.  Predicted states are fed
    back into the model, while the target at each horizon is the corresponding
    recorded next state.  ADM/RNN sample a shared per-step backtrack length and
    all model families sample their predictive Gaussian.  No BC policy, MuJoCo
    oracle, teacher forcing, or terminal-based filtering is used.
    """
    evaluation = config["evaluation"]
    history = int(config["model"]["max_backtrack"])
    count = int(evaluation["starts"])
    horizon = int(evaluation["horizon"])
    overflow = float(evaluation["overflow"])
    device = config["device"]
    rollout_seed = int(evaluation["rollout_seed"])
    common_seed = rollout_seed + seed * 1_000_003
    backtrack_rng = np.random.default_rng(common_seed + 11)
    noise_rng = np.random.default_rng(
        common_seed + {"adm": 101, "rnn": 211, "ensemble": 307}[kind]
    )
    elite_rng = np.random.default_rng(common_seed + 401)
    if windows is None:
        windows = build_test_windows(
            dataset,
            split,
            history,
            horizon,
            count,
            int(evaluation["window_seed"]),
        )
    observation_history = windows.observation_history.copy()
    action_history = windows.action_history.copy()
    rows: list[dict[str, float | int | str]] = []
    for step in range(horizon):
        action = windows.future_actions[:, step]
        predicted, backtrack_k = _sample_prediction_step(
            kind,
            model,
            observation_history,
            action_history,
            action,
            device,
            history,
            backtrack_rng,
            noise_rng,
            elite_rng,
        )
        target = windows.true_next_observations[:, step]
        difference = predicted.astype(np.float64) - target.astype(np.float64)
        squared = np.mean(np.square(difference), axis=-1, dtype=np.float64)
        squared = np.nan_to_num(squared, nan=overflow, posinf=overflow, neginf=overflow)
        squared = np.minimum(squared, overflow)
        rows.append(
            {
                "task": task,
                "seed": seed,
                "model": kind,
                "rollout_length": step + 1,
                "backtrack_k": backtrack_k,
                "error_mean": float(np.mean(squared, dtype=np.float64)),
                "error_std": float(np.std(squared, dtype=np.float64)),
                "error_sem": float(np.std(squared, dtype=np.float64) / np.sqrt(squared.size)),
                "n_windows": int(squared.size),
            }
        )
        observation_history = np.concatenate(
            (observation_history[:, 1:], predicted[:, None]), axis=1
        )
        if history > 1:
            action_history = np.concatenate(
                (action_history[:, 1:], action[:, None]), axis=1
            )
    return rows


def write_figure2_rows(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "seed",
        "model",
        "rollout_length",
        "backtrack_k",
        "error_mean",
        "error_std",
        "error_sem",
        "n_windows",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
