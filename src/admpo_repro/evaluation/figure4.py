from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from admpo_repro.data import D4RLDataset
from admpo_repro.dynamics.models import ADMDynamics, EnsembleDynamics

from .figure2 import _initial_histories
from .oracle import MujocoOracle, termination


ActionFunction = Callable[[np.ndarray], np.ndarray]


@torch.no_grad()
def prediction_statistics(
    kind: str,
    model: ADMDynamics | EnsembleDynamics,
    obs_hist: np.ndarray,
    act_hist: np.ndarray,
    action: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = torch.as_tensor(obs_hist[:, -1], device=device)
    action_t = torch.as_tensor(action, device=device)
    if kind == "adm":
        all_actions = np.concatenate((act_hist, action[:, None]), axis=1)
        action_history = torch.as_tensor(all_actions, device=device)
        means, stds = [], []
        obs_tensor = torch.as_tensor(obs_hist, device=device)
        for k in range(1, obs_hist.shape[1] + 1):
            mean, std, _, _ = model.dyna_dist(obs_tensor[:, -k], action_history[:, -k:])
            means.append(mean)
            stds.append(std)
        mean_stack = torch.stack(means)
        std_stack = torch.stack(stds)
    elif kind == "ensemble":
        mean_stack, std_stack, _, _ = model.dyna_dist(current, action_t)
    else:
        raise ValueError(kind)
    mean_prediction = mean_stack.mean(dim=0)
    epistemic = torch.sqrt(mean_stack.var(dim=0).mean(dim=-1).clamp_min(0))
    total_variance = mean_stack.var(dim=0) + std_stack.square().mean(dim=0)
    total = torch.sqrt(total_variance.mean(dim=-1).clamp_min(0))
    return (
        mean_prediction.cpu().numpy(),
        epistemic.cpu().numpy(),
        total.cpu().numpy(),
        std_stack.cpu().numpy(),
    )


@torch.no_grad()
def _rollout_distribution(
    kind: str,
    model: ADMDynamics | EnsembleDynamics,
    obs_hist: np.ndarray,
    act_hist: np.ndarray,
    action: np.ndarray,
    rng: np.random.Generator,
    device: str,
) -> np.ndarray:
    current = torch.as_tensor(obs_hist[:, -1], device=device)
    if kind == "adm":
        actions = torch.as_tensor(np.concatenate((act_hist, action[:, None]), axis=1), device=device)
        obs = torch.as_tensor(obs_hist, device=device)
        k = int(rng.integers(1, obs_hist.shape[1] + 1))
        mean, std, _, _ = model.dyna_dist(obs[:, -k], actions[:, -k:])
    else:
        means, stds, _, _ = model.dyna_dist(current, torch.as_tensor(action, device=device))
        choices = rng.integers(0, means.shape[0], size=means.shape[1])
        columns = torch.arange(means.shape[1], device=device)
        choice_t = torch.as_tensor(choices, device=device)
        mean, std = means[choice_t, columns], stds[choice_t, columns]
    mean_np = mean.cpu().numpy()
    std_np = np.nan_to_num(std.cpu().numpy(), nan=1e-6, posinf=1e-6, neginf=1e-6)
    return (mean_np + rng.normal(size=mean_np.shape) * np.maximum(std_np, 1e-6)).astype(np.float32)


def collect_rollout_points(
    task: str,
    kind: str,
    model: ADMDynamics | EnsembleDynamics,
    policy_name: str,
    policy: ActionFunction | None,
    dataset: D4RLDataset,
    env: object,
    seed: int,
    config: dict,
) -> list[dict]:
    cfg = config["evaluation"]
    device = config["device"]
    history = int(config["model"]["max_backtrack"])
    target_points = int(cfg["points_per_policy"])
    starts = int(cfg["starts"])
    horizon = int(cfg["rollout_horizon"])
    rng = np.random.default_rng(seed * 7919 + {"adm": 11, "ensemble": 17}[kind])
    oracle = MujocoOracle(env)
    rows: list[dict] = []
    attempts = 0
    while len(rows) < target_points and attempts < 20:
        obs_hist, act_hist = _initial_histories(dataset, starts, history, seed + attempts * 101)
        alive = np.ones(starts, dtype=bool)
        for rollout_step in range(horizon):
            if policy_name == "random":
                action = rng.uniform(env.action_space.low, env.action_space.high, size=(starts, dataset.action_dim)).astype(np.float32)
            else:
                if policy is None:
                    raise ValueError(f"policy callable required for {policy_name}")
                action = np.asarray(policy(obs_hist[:, -1]), dtype=np.float32)
            mean, uncertainty, total_uncertainty, _ = prediction_statistics(
                kind, model, obs_hist, act_hist, action, device
            )
            active_ids = np.where(alive)[0]
            truth, _, _ = oracle.batch_step(obs_hist[active_ids, -1], action[active_ids])
            errors = np.mean(np.square(mean[active_ids] - truth), axis=-1)
            for local, idx in enumerate(active_ids):
                if np.isfinite(errors[local]) and np.isfinite(uncertainty[idx]):
                    rows.append(
                        {
                            "task": task,
                            "seed": seed,
                            "model": kind,
                            "policy": policy_name,
                            "rollout_step": rollout_step + 1,
                            "model_error": float(errors[local]),
                            "uncertainty": float(uncertainty[idx]),
                            "total_uncertainty": float(total_uncertainty[idx]),
                        }
                    )
                    if len(rows) >= target_points:
                        break
            if len(rows) >= target_points:
                break
            predicted = _rollout_distribution(kind, model, obs_hist, act_hist, action, rng, device)
            invalid = ~np.isfinite(predicted).all(axis=-1)
            alive &= ~(termination(task, predicted) | invalid)
            obs_hist = np.concatenate((obs_hist[:, 1:], predicted[:, None]), axis=1)
            act_hist = np.concatenate((act_hist[:, 1:], action[:, None]), axis=1)
            if not alive.any():
                break
        attempts += 1
    if len(rows) < target_points:
        raise RuntimeError(
            f"collected only {len(rows)}/{target_points} valid points for {task}/{kind}/{policy_name}"
        )
    return rows[:target_points]


def collect_dataset_diagnostic(
    task: str,
    kind: str,
    model: ADMDynamics | EnsembleDynamics,
    dataset: D4RLDataset,
    env: object,
    seed: int,
    config: dict,
) -> list[dict]:
    history = int(config["model"]["max_backtrack"])
    points = int(config["evaluation"]["dataset_diagnostic_points"])
    pool = dataset.valid_starts(history)
    rng = np.random.default_rng(seed * 3571 + (1 if kind == "adm" else 2))
    starts = rng.choice(pool, size=points, replace=pool.size < points)
    seq = dataset.sequences(starts, history)
    obs_hist = seq["observations"]
    act_hist = seq["actions"][:, :-1]
    action = seq["actions"][:, -1]
    mean, uncertainty, total, _ = prediction_statistics(
        kind, model, obs_hist, act_hist, action, config["device"]
    )
    truth, _, _ = MujocoOracle(env).batch_step(obs_hist[:, -1], action)
    errors = np.mean(np.square(mean - truth), axis=-1)
    rows = []
    for error, unc, total_unc in zip(errors, uncertainty, total):
        if np.isfinite(error) and np.isfinite(unc):
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "model": kind,
                    "policy": "dataset-direct",
                    "rollout_step": 0,
                    "model_error": float(error),
                    "uncertainty": float(unc),
                    "total_uncertainty": float(total_unc),
                }
            )
    return rows


def write_figure4_rows(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task", "seed", "model", "policy", "rollout_step",
        "model_error", "uncertainty", "total_uncertainty",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
