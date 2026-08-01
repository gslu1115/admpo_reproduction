from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from admpo_repro.data import D4RLDataset
from admpo_repro.runtime import capture_rng_state, restore_rng_state


class BCPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: list[int] | tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = obs_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, int(np.asarray(action_low).size)))
        self.network = nn.Sequential(*layers)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.network(obs))
        return self.action_low + 0.5 * (raw + 1.0) * (self.action_high - self.action_low)

    @torch.no_grad()
    def act(self, obs: np.ndarray | torch.Tensor) -> np.ndarray:
        device = next(self.parameters()).device
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return self(tensor).cpu().numpy()


def _save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train_bc(
    dataset: D4RLDataset,
    env: object,
    config: dict,
    seed: int,
    checkpoint: Path,
    resume: bool,
) -> dict[str, Any]:
    cfg = config["bc"]
    device = config["device"]
    policy = BCPolicy(
        dataset.obs_dim,
        env.action_space.low,
        env.action_space.high,
        cfg["hidden_dims"],
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(cfg["learning_rate"]))
    start_epoch = 0
    history: list[dict[str, float]] = []
    last_path = checkpoint.with_name(checkpoint.stem + ".last.pt")
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        history = state["history"]
        restore_rng_state(state["rng"])
    rng = np.random.default_rng(seed + start_epoch * 100003)
    for epoch in range(start_epoch, int(cfg["epochs"])):
        losses = []
        for _ in range(int(cfg["steps_per_epoch"])):
            idx = rng.integers(0, dataset.size, size=int(cfg["batch_size"]))
            obs = torch.as_tensor(dataset.observations[idx], device=device)
            action = torch.as_tensor(dataset.actions[idx], device=device)
            loss = (policy(obs) - action).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        _save(
            {
                "epoch": epoch,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "rng": capture_rng_state(),
            },
            last_path,
        )
    payload = {
        "model": policy.state_dict(),
        "history": history,
        "seed": seed,
        "task": dataset.task,
        "hidden_dims": list(cfg["hidden_dims"]),
    }
    _save(payload, checkpoint)
    checkpoint.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "model"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def load_bc(
    dataset: D4RLDataset, env: object, config: dict, checkpoint: Path
) -> BCPolicy:
    cfg = config["bc"]
    policy = BCPolicy(
        dataset.obs_dim,
        env.action_space.low,
        env.action_space.high,
        cfg["hidden_dims"],
    ).to(config["device"])
    policy.load_state_dict(torch.load(checkpoint, map_location=config["device"])["model"])
    policy.eval()
    return policy
