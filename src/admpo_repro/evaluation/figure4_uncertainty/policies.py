from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class RolloutPolicy(Protocol):
    name: str

    def act(self, states: np.ndarray, sample_ids: np.ndarray, rollout_step: int) -> np.ndarray:
        ...


class RandomUniformPolicy:
    name = "random"

    def __init__(self, action_table: np.ndarray, action_low: np.ndarray, action_high: np.ndarray):
        self.action_table = np.asarray(action_table, dtype=np.float32)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        if np.any(self.action_table < self.action_low) or np.any(self.action_table > self.action_high):
            raise RuntimeError("precomputed random actions exceed environment bounds")

    def act(self, states: np.ndarray, sample_ids: np.ndarray, rollout_step: int) -> np.ndarray:
        del states
        return self.action_table[np.asarray(sample_ids, dtype=np.int64), int(rollout_step)].copy()


class DeterministicBCPolicy:
    name = "bc"

    def __init__(self, model: torch.nn.Module, action_low: np.ndarray, action_high: np.ndarray):
        self.model = model
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)

    @torch.no_grad()
    def act(self, states: np.ndarray, sample_ids: np.ndarray, rollout_step: int) -> np.ndarray:
        del sample_ids, rollout_step
        action = np.asarray(self.model.act(states), dtype=np.float32)
        if np.any(action < self.action_low - 1e-6) or np.any(action > self.action_high + 1e-6):
            raise RuntimeError("BC emitted an action outside the environment bounds")
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)


class SACPolicy:
    """Official SAC actor evaluated with its deterministic mean action."""

    name = "learned_sac"

    def __init__(
        self,
        actor: torch.nn.Module,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> None:
        self.actor = actor
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.actor.eval()

    @torch.no_grad()
    def act(self, states: np.ndarray, sample_ids: np.ndarray, rollout_step: int) -> np.ndarray:
        del sample_ids, rollout_step
        device = next(self.actor.parameters()).device
        observations = torch.as_tensor(states, dtype=torch.float32, device=device)
        distribution = self.actor(observations)
        raw_action = distribution.mean
        scale = torch.as_tensor(
            (self.action_high - self.action_low) / 2.0, dtype=torch.float32, device=device
        )
        # This intentionally matches vendor SACAgent.actor4ward. Hopper bounds are symmetric.
        action = (scale * torch.tanh(raw_action)).cpu().numpy().astype(np.float32)
        if np.any(action < self.action_low - 1e-6) or np.any(action > self.action_high + 1e-6):
            raise RuntimeError("SAC emitted an action outside the environment bounds")
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)


def random_action_table(
    rng: np.random.Generator,
    trajectories: int,
    steps: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    return rng.uniform(
        np.asarray(action_low, dtype=np.float32),
        np.asarray(action_high, dtype=np.float32),
        size=(int(trajectories), int(steps), int(np.asarray(action_low).size)),
    ).astype(np.float32)
