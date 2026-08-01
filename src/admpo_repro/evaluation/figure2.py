from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from admpo_repro.data import D4RLDataset
from admpo_repro.dynamics.models import ADMDynamics, EnsembleDynamics, RNNDynamics
from admpo_repro.policies.bc import BCPolicy

from .oracle import MujocoOracle, termination


def _initial_histories(
    dataset: D4RLDataset, starts: int, history: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    pool = dataset.valid_starts(history, 0, dataset.size)
    rng = np.random.default_rng(seed)
    indexes = rng.choice(pool, size=starts, replace=pool.size < starts)
    seq = dataset.sequences(indexes, history)
    obs_hist = np.concatenate(
        (seq["observations"][:, :1], seq["next_observations"]), axis=1
    )[:, -history:]
    act_hist = seq["actions"][:, -(history - 1) :] if history > 1 else seq["actions"][:, :0]
    return obs_hist.astype(np.float32), act_hist.astype(np.float32)


@torch.no_grad()
def _sample_model_step(
    kind: str,
    model: ADMDynamics | RNNDynamics | EnsembleDynamics,
    obs_hist: np.ndarray,
    act_hist: np.ndarray,
    action: np.ndarray,
    rng: np.random.Generator,
    device: str,
) -> np.ndarray:
    obs_t = torch.as_tensor(obs_hist, device=device)
    action_t = torch.as_tensor(action, device=device)
    all_actions = np.concatenate((act_hist, action[:, None]), axis=1)
    actions_t = torch.as_tensor(all_actions, device=device)
    if kind == "adm":
        k = int(rng.integers(1, obs_hist.shape[1] + 1))
        mean, std, _, _ = model.dyna_dist(obs_t[:, -k], actions_t[:, -k:])
    elif kind == "rnn":
        mean, std, _, _ = model.dyna_dist(obs_t, actions_t)
    else:
        means, stds, _, _ = model.dyna_dist(obs_t[:, -1], action_t)
        chosen = rng.integers(0, means.shape[0], size=means.shape[1])
        columns = torch.arange(means.shape[1], device=device)
        mean = means[torch.as_tensor(chosen, device=device), columns]
        std = stds[torch.as_tensor(chosen, device=device), columns]
    mean_np = mean.cpu().numpy()
    std_np = np.nan_to_num(std.cpu().numpy(), nan=1e-6, posinf=1e-6, neginf=1e-6)
    return (mean_np + rng.normal(size=mean_np.shape) * np.maximum(std_np, 1e-6)).astype(np.float32)


def evaluate_model_rollout(
    task: str,
    kind: str,
    model: ADMDynamics | RNNDynamics | EnsembleDynamics,
    policy: BCPolicy,
    dataset: D4RLDataset,
    env: object,
    seed: int,
    config: dict,
) -> list[dict[str, float | int | str]]:
    evaluation = config["evaluation"]
    history = int(config["model"]["max_backtrack"])
    count = int(evaluation["starts"])
    horizon = int(evaluation["horizon"])
    overflow = float(evaluation["overflow"])
    device = config["device"]
    rng = np.random.default_rng(seed * 1009 + {"adm": 1, "ensemble": 2, "rnn": 3}[kind])
    obs_hist, act_hist = _initial_histories(dataset, count, history, seed)
    oracle_obs = obs_hist[:, -1].copy()
    alive = np.ones(count, dtype=bool)
    oracle = MujocoOracle(env)
    rows: list[dict[str, float | int | str]] = []
    for step in range(1, horizon + 1):
        if not alive.any():
            rows.append(
                {"task": task, "seed": seed, "model": kind, "rollout_length": step, "error_mean": overflow, "error_std": 0.0, "n_active": 0}
            )
            continue
        action = policy.act(obs_hist[:, -1])
        predicted = _sample_model_step(kind, model, obs_hist, act_hist, action, rng, device)
        oracle_next = np.full_like(oracle_obs, np.nan)
        active_ids = np.where(alive)[0]
        stepped, _, active_oracle_done = oracle.batch_step(oracle_obs[active_ids], action[active_ids])
        oracle_next[active_ids] = stepped
        squared = np.mean(np.square(predicted - oracle_next), axis=-1)
        invalid = ~np.isfinite(squared)
        squared = np.nan_to_num(squared, nan=overflow, posinf=overflow, neginf=overflow)
        squared = np.minimum(squared, overflow)
        current = squared[alive]
        rows.append(
            {
                "task": task,
                "seed": seed,
                "model": kind,
                "rollout_length": step,
                "error_mean": float(np.mean(current, dtype=np.float64)),
                "error_std": float(np.std(current, dtype=np.float64)),
                "n_active": int(alive.sum()),
            }
        )
        model_done = termination(task, predicted)
        oracle_done = np.zeros(count, dtype=bool)
        oracle_done[active_ids] = active_oracle_done.reshape(-1)
        just_done = invalid | model_done | oracle_done
        obs_hist = np.concatenate((obs_hist[:, 1:], predicted[:, None]), axis=1)
        act_hist = np.concatenate((act_hist[:, 1:], action[:, None]), axis=1)
        oracle_obs = oracle_next
        alive &= ~just_done
    return rows


def write_figure2_rows(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "seed", "model", "rollout_length", "error_mean", "error_std", "n_active"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
