from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class D4RLDataset:
    task: str
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    max_episode_steps: int
    source_path: Path | None = None

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])

    @property
    def boundaries(self) -> np.ndarray:
        return np.logical_or(self.terminals.reshape(-1), self.timeouts.reshape(-1))

    def statistics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_mu = self.observations.mean(axis=0, dtype=np.float64).astype(np.float32)
        obs_std = self.observations.std(axis=0, dtype=np.float64).astype(np.float32)
        act_mu = self.actions.mean(axis=0, dtype=np.float64).astype(np.float32)
        act_std = self.actions.std(axis=0, dtype=np.float64).astype(np.float32)
        obs_std[obs_std < 1e-12] = 1.0
        act_std[act_std < 1e-12] = 1.0
        return obs_mu, obs_std, act_mu, act_std

    def valid_starts(self, length: int, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Return starts for `length` consecutive transitions inside one episode."""
        if length < 1:
            raise ValueError("length must be >= 1")
        stop = self.size if stop is None else min(stop, self.size)
        last = stop - length
        if last < start:
            return np.empty(0, dtype=np.int64)
        candidates = np.arange(start, last + 1, dtype=np.int64)
        if length == 1:
            return candidates
        boundary = self.boundaries.astype(np.int64)
        prefix = np.concatenate([[0], np.cumsum(boundary)])
        # A boundary on the final transition is allowed; earlier boundaries are not.
        internal = prefix[candidates + length - 1] - prefix[candidates]
        return candidates[internal == 0]

    def sequences(self, starts: np.ndarray, length: int) -> dict[str, np.ndarray]:
        starts = np.asarray(starts, dtype=np.int64)
        offsets = np.arange(length, dtype=np.int64)
        idx = starts[:, None] + offsets[None, :]
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "terminals": self.terminals[idx],
            "timeouts": self.timeouts[idx],
        }

    def split_point(self, holdout_size: int) -> int:
        return max(1, self.size - min(int(self.size * 0.2), int(holdout_size)))


def _infer_dataset_path(task: str) -> Path | None:
    stem = task.replace("-", "_")
    if stem.endswith("_v2"):
        stem = stem[:-3] + "-v2"
    candidates = [
        Path.home() / ".d4rl" / f"{stem}.hdf5",
        Path.home() / ".d4rl" / "datasets" / f"{stem}.hdf5",
    ]
    return next((path for path in candidates if path.exists()), None)


def load_d4rl_dataset(task: str, seed: int = 0) -> tuple[D4RLDataset, object]:
    # Imports happen only after runtime.configure_mujoco_environment().
    import d4rl  # noqa: F401
    import gym

    env = gym.make(task)
    try:
        env.seed(seed)
    except AttributeError:
        env.reset(seed=seed)
    env.action_space.seed(seed)
    raw = env.get_dataset()
    n = int(raw["rewards"].shape[0])
    observations = np.asarray(raw["observations"], dtype=np.float32)
    actions = np.asarray(raw["actions"], dtype=np.float32)
    rewards = np.asarray(raw["rewards"], dtype=np.float32).reshape(n, 1)
    terminals = np.asarray(raw["terminals"], dtype=np.float32).reshape(n, 1)
    timeouts = np.asarray(raw.get("timeouts", np.zeros(n)), dtype=np.float32).reshape(n, 1)
    if "next_observations" in raw:
        next_observations = np.asarray(raw["next_observations"], dtype=np.float32)
    else:
        next_observations = np.concatenate(
            [observations[1:], observations[-1:]], axis=0
        ).astype(np.float32)
    dataset = D4RLDataset(
        task=task,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        timeouts=timeouts,
        max_episode_steps=int(getattr(env, "_max_episode_steps", 1000)),
        source_path=_infer_dataset_path(task),
    )
    return dataset, env
