from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from admpo_repro.data import D4RLDataset
from admpo_repro.runtime import capture_rng_state, restore_rng_state

from .models import ADMDynamics, EnsembleDynamics, RNNDynamics


def build_dynamics(kind: str, dataset: D4RLDataset, config: dict, device: str) -> torch.nn.Module:
    model_cfg = config["model"]
    if kind == "adm":
        model = ADMDynamics(
            dataset.obs_dim,
            dataset.action_dim,
            hidden_dim=model_cfg["hidden_dim"],
            rnn_layers=model_cfg.get("rnn_layers", 3),
            residual_blocks=model_cfg.get("residual_blocks", 4),
            dropout=model_cfg.get("dropout", 0.1),
        )
    elif kind == "rnn":
        model = RNNDynamics(
            dataset.obs_dim,
            dataset.action_dim,
            hidden_dim=model_cfg["hidden_dim"],
            rnn_layers=model_cfg.get("rnn_layers", 3),
            residual_blocks=model_cfg.get("residual_blocks", 4),
            dropout=model_cfg.get("dropout", 0.1),
        )
    elif kind == "ensemble":
        model = EnsembleDynamics(
            dataset.obs_dim,
            dataset.action_dim,
            hidden_dims=model_cfg.get("ensemble_hidden", [200, 200, 200, 200]),
            size=model_cfg.get("ensemble_size", 7),
            elites=model_cfg.get("elite_size", 5),
        )
    else:
        raise ValueError(f"unknown dynamics kind: {kind}")
    model.to(device)
    model.set_statistics(*dataset.statistics())
    return model


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sample_starts(pool: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    if pool.size == 0:
        raise RuntimeError("no valid sequences available; dataset boundary logic is invalid")
    return rng.choice(pool, size=batch_size, replace=pool.size < batch_size)


@torch.no_grad()
def _validate_sequence_model(
    model: torch.nn.Module,
    dataset: D4RLDataset,
    kind: str,
    split: int,
    max_backtrack: int,
    batch_size: int,
    device: str,
) -> list[float]:
    model.eval()
    losses: list[float] = []
    for k in range(1, max_backtrack + 1):
        pool = dataset.valid_starts(k, split, dataset.size)
        starts = pool[: min(pool.size, max(batch_size, 4096))]
        seq = dataset.sequences(starts, k)
        obs = torch.as_tensor(seq["observations"], device=device)
        actions = torch.as_tensor(seq["actions"], device=device)
        rewards = torch.as_tensor(seq["rewards"][:, -1], device=device)
        next_obs = torch.as_tensor(seq["next_observations"][:, -1], device=device)
        if kind == "adm":
            model_obs = obs[:, 0]
            target_delta = next_obs - obs[:, 0]
        else:
            model_obs = obs
            target_delta = next_obs - obs[:, -1]
        target = torch.cat((target_delta, rewards), dim=-1)
        mean, _ = model(model_obs, actions)
        losses.append(float((mean - target).square().mean().cpu()))
    model.train()
    return losses


def train_sequence_model(
    kind: str,
    model: ADMDynamics | RNNDynamics,
    dataset: D4RLDataset,
    config: dict,
    seed: int,
    checkpoint: Path,
    resume: bool,
) -> dict[str, Any]:
    train_cfg = config["training"]
    device = config["device"]
    batch_size = int(train_cfg["batch_size"])
    max_backtrack = int(config["model"]["max_backtrack"])
    split = dataset.split_point(int(train_cfg["holdout_size"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))
    patience = int(train_cfg["adm_patience"])
    best_losses = [float("inf")] * max_backtrack
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    start_epoch = 0
    history: list[dict[str, Any]] = []
    last_path = checkpoint.with_name(checkpoint.stem + ".last.pt")
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_state = state["best_state"]
        best_losses = state["best_losses"]
        stale = state["stale"]
        start_epoch = state["epoch"] + 1
        history = state["history"]
        restore_rng_state(state["rng"])
    steps = train_cfg.get("steps_per_epoch")
    default_steps = max(1, split // batch_size) * (max_backtrack if kind == "adm" else 1)
    steps_per_epoch = int(steps if steps is not None else default_steps)
    rng = np.random.default_rng(seed + start_epoch * 100003)
    for epoch in range(start_epoch, int(train_cfg["max_epochs"])):
        model.train()
        epoch_losses = []
        for _ in range(steps_per_epoch):
            k = int(rng.integers(1, max_backtrack + 1))
            pool = dataset.valid_starts(k, 0, split)
            seq = dataset.sequences(_sample_starts(pool, batch_size, rng), k)
            obs = torch.as_tensor(seq["observations"], device=device)
            actions = torch.as_tensor(seq["actions"], device=device)
            rewards = torch.as_tensor(seq["rewards"][:, -1], device=device)
            next_obs = torch.as_tensor(seq["next_observations"][:, -1], device=device)
            if kind == "adm":
                model_obs = obs[:, 0]
                delta = next_obs - obs[:, 0]
            else:
                model_obs = obs
                delta = next_obs - obs[:, -1]
            target = torch.cat((delta, rewards), dim=-1)
            mean, logvar = model(model_obs, actions)
            loss = ((mean - target).square() * torch.exp(-logvar)).mean() + logvar.mean()
            loss = loss + 0.01 * model.max_logvar.sum() - 0.01 * model.min_logvar.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        validation = _validate_sequence_model(
            model, dataset, kind, split, max_backtrack, batch_size, device
        )
        improved = np.mean(validation) < np.mean(best_losses)
        if improved:
            best_losses = validation
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "validation": validation}
        )
        _atomic_torch_save(
            {
                "kind": kind,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_state": best_state,
                "best_losses": best_losses,
                "stale": stale,
                "history": history,
                "rng": capture_rng_state(),
            },
            last_path,
        )
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    payload = {
        "kind": kind,
        "model": model.state_dict(),
        "validation": best_losses,
        "history": history,
        "seed": seed,
        "task": dataset.task,
    }
    _atomic_torch_save(payload, checkpoint)
    checkpoint.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "model"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


@torch.no_grad()
def _validate_ensemble(
    model: EnsembleDynamics,
    dataset: D4RLDataset,
    starts: np.ndarray,
    device: str,
) -> list[float]:
    model.eval()
    obs = torch.as_tensor(dataset.observations[starts], device=device)
    action = torch.as_tensor(dataset.actions[starts], device=device)
    target = torch.cat(
        (
            torch.as_tensor(dataset.next_observations[starts] - dataset.observations[starts], device=device),
            torch.as_tensor(dataset.rewards[starts], device=device),
        ),
        dim=-1,
    )
    mean, _ = model(obs, action)
    losses = (mean - target).square().mean(dim=(1, 2))
    model.train()
    return [float(v) for v in losses.cpu()]


def _update_best_members(
    best: dict[str, torch.Tensor], current: dict[str, torch.Tensor], indexes: list[int], size: int
) -> None:
    for key, value in current.items():
        if value.ndim > 0 and value.shape[0] == size and (
            ".weight" in key or ".bias" in key
        ):
            best[key][indexes] = value[indexes]
        elif "max_logvar" in key or "min_logvar" in key:
            best[key] = value.clone()


def _ensemble_member_improved(new: float, old: float, threshold: float) -> bool:
    """Return whether a finite validation loss improves an ensemble member."""
    if not np.isfinite(new):
        return False
    if not np.isfinite(old):
        return True
    denominator = max(abs(old), np.finfo(np.float64).eps)
    return (old - new) / denominator > threshold


def train_ensemble(
    model: EnsembleDynamics,
    dataset: D4RLDataset,
    config: dict,
    seed: int,
    checkpoint: Path,
    resume: bool,
) -> dict[str, Any]:
    cfg = config["training"]
    device = config["device"]
    batch_size = int(cfg["batch_size"])
    split = dataset.split_point(int(cfg["holdout_size"]))
    train_pool = np.arange(split, dtype=np.int64)
    holdout_pool = np.arange(split, dataset.size, dtype=np.int64)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    size = model.model.size
    best_losses = [float("inf")] * size
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    start_epoch = 0
    history: list[dict[str, Any]] = []
    last_path = checkpoint.with_name(checkpoint.stem + ".last.pt")
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_state = state["best_state"]
        best_losses = state["best_losses"]
        stale = state["stale"]
        start_epoch = state["epoch"] + 1
        history = state["history"]
        restore_rng_state(state["rng"])
    rng = np.random.default_rng(seed + start_epoch * 100003)
    steps = cfg.get("steps_per_epoch")
    steps_per_epoch = int(steps if steps is not None else max(1, split // batch_size))
    improvement_threshold = float(cfg["ensemble_improvement"])
    for epoch in range(start_epoch, int(cfg["max_epochs"])):
        bootstrap = rng.choice(train_pool, size=(size, train_pool.size), replace=True)
        epoch_losses = []
        for step in range(steps_per_epoch):
            offset = (step * batch_size) % train_pool.size
            select = bootstrap[:, offset : offset + batch_size]
            if select.shape[1] < batch_size:
                extra = bootstrap[:, : batch_size - select.shape[1]]
                select = np.concatenate((select, extra), axis=1)
            obs = torch.as_tensor(dataset.observations[select], device=device)
            action = torch.as_tensor(dataset.actions[select], device=device)
            target = torch.cat(
                (
                    torch.as_tensor(dataset.next_observations[select] - dataset.observations[select], device=device),
                    torch.as_tensor(dataset.rewards[select], device=device),
                ),
                dim=-1,
            )
            mean, logvar = model(obs, action)
            loss = (((mean - target).square() * torch.exp(-logvar)).mean((1, 2)) + logvar.mean((1, 2))).sum()
            loss = loss + model.model.decay_loss()
            loss = loss + 0.01 * model.model.max_logvar.sum() - 0.01 * model.model.min_logvar.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        validation = _validate_ensemble(model, dataset, holdout_pool, device)
        improved = [
            i for i, (new, old) in enumerate(zip(validation, best_losses))
            if _ensemble_member_improved(new, old, improvement_threshold)
        ]
        if improved:
            _update_best_members(best_state, model.state_dict(), improved, size)
            for i in improved:
                best_losses[i] = validation[i]
            stale = 0
        else:
            stale += 1
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "validation": validation}
        )
        _atomic_torch_save(
            {
                "kind": "ensemble",
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_state": best_state,
                "best_losses": best_losses,
                "stale": stale,
                "history": history,
                "rng": capture_rng_state(),
            },
            last_path,
        )
        if stale >= int(cfg["ensemble_patience"]):
            break
    model.load_state_dict(best_state)
    elite = np.argsort(best_losses)[: model.model.num_elites]
    model.model.elite_indices.copy_(torch.as_tensor(elite, device=model.model.elite_indices.device))
    payload = {
        "kind": "ensemble",
        "model": model.state_dict(),
        "validation": best_losses,
        "elites": elite.tolist(),
        "history": history,
        "seed": seed,
        "task": dataset.task,
    }
    _atomic_torch_save(payload, checkpoint)
    checkpoint.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "model"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def load_dynamics_checkpoint(
    kind: str, dataset: D4RLDataset, config: dict, path: Path
) -> torch.nn.Module:
    model = build_dynamics(kind, dataset, config, config["device"])
    payload = torch.load(path, map_location=config["device"])
    model.load_state_dict(payload["model"])
    model.eval()
    return model
