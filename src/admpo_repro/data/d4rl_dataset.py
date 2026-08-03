from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Partition = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class TrajectorySplit:
    """A deterministic split whose unit is a complete D4RL episode."""

    episode_ranges: tuple[tuple[int, int], ...]
    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]
    test_episodes: tuple[int, ...]
    seed: int
    ratios: tuple[float, float, float]

    def episode_ids(self, partition: Partition) -> tuple[int, ...]:
        if partition == "train":
            return self.train_episodes
        if partition == "validation":
            return self.validation_episodes
        if partition == "test":
            return self.test_episodes
        raise ValueError(f"unknown partition: {partition}")

    def ranges(self, partition: Partition) -> tuple[tuple[int, int], ...]:
        return tuple(self.episode_ranges[i] for i in self.episode_ids(partition))

    def indices(self, partition: Partition) -> np.ndarray:
        chunks = [np.arange(start, stop, dtype=np.int64) for start, stop in self.ranges(partition)]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def valid_starts(self, partition: Partition, length: int) -> np.ndarray:
        """Return starts for windows wholly contained in the selected episodes."""
        if length < 1:
            raise ValueError("length must be >= 1")
        chunks = [
            np.arange(start, stop - length + 1, dtype=np.int64)
            for start, stop in self.ranges(partition)
            if stop - start >= length
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def to_dict(self) -> dict:
        total_transitions = sum(stop - start for start, stop in self.episode_ranges)

        def describe(partition: Partition) -> dict:
            ids = self.episode_ids(partition)
            transition_count = int(
                sum(self.episode_ranges[i][1] - self.episode_ranges[i][0] for i in ids)
            )
            return {
                "episode_ids": list(ids),
                "episode_count": len(ids),
                "transition_count": transition_count,
                "transition_fraction": transition_count / total_transitions,
            }

        return {
            "unit": "complete_trajectory",
            "seed": self.seed,
            "ratios": {
                "train": self.ratios[0],
                "validation": self.ratios[1],
                "test": self.ratios[2],
            },
            "episode_count": len(self.episode_ranges),
            "train": describe("train"),
            "validation": describe("validation"),
            "test": describe("test"),
        }


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

    def statistics(
        self, indices: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return normalization statistics, optionally from training transitions only."""
        observations = self.observations if indices is None else self.observations[indices]
        actions = self.actions if indices is None else self.actions[indices]
        if observations.shape[0] == 0:
            raise ValueError("cannot compute statistics from an empty partition")
        obs_mu = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
        obs_std = observations.std(axis=0, dtype=np.float64).astype(np.float32)
        act_mu = actions.mean(axis=0, dtype=np.float64).astype(np.float32)
        act_std = actions.std(axis=0, dtype=np.float64).astype(np.float32)
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

    def episode_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return half-open transition ranges for every complete or trailing episode."""
        ends = (np.flatnonzero(self.boundaries) + 1).tolist()
        if not ends or ends[-1] != self.size:
            ends.append(self.size)
        starts = [0, *ends[:-1]]
        return tuple((int(start), int(stop)) for start, stop in zip(starts, ends) if stop > start)

    def trajectory_split(
        self,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 0,
    ) -> TrajectorySplit:
        ratios = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=np.float64)
        if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
            raise ValueError("trajectory split ratios must be positive and sum to 1")
        ranges = self.episode_ranges()
        count = len(ranges)
        if count < 3:
            raise ValueError("at least three episodes are required for an 80/10/10 split")
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(count)
        lengths = np.asarray([stop - start for start, stop in ranges], dtype=np.int64)

        def closest_prefix(ids: np.ndarray, target: float, maximum: int) -> int:
            cumulative = np.cumsum(lengths[ids], dtype=np.int64)
            candidates = cumulative[:maximum]
            return int(np.argmin(np.abs(candidates - target))) + 1

        # Keep trajectories intact while making transition counts as close as
        # possible to 80/10/10. Equal-length trajectories reduce exactly to an
        # episode-count split.
        train_count = closest_prefix(shuffled, self.size * train_ratio, count - 2)
        remaining = shuffled[train_count:]
        validation_count = closest_prefix(
            remaining, self.size * validation_ratio, remaining.size - 1
        )
        train = tuple(sorted(int(i) for i in shuffled[:train_count]))
        validation = tuple(
            sorted(int(i) for i in shuffled[train_count : train_count + validation_count])
        )
        test = tuple(sorted(int(i) for i in shuffled[train_count + validation_count :]))
        return TrajectorySplit(
            episode_ranges=ranges,
            train_episodes=train,
            validation_episodes=validation,
            test_episodes=test,
            seed=int(seed),
            ratios=(float(train_ratio), float(validation_ratio), float(test_ratio)),
        )


def _infer_dataset_path(task: str) -> Path | None:
    stem = task.replace("-", "_")
    if stem.endswith("_v2"):
        stem = stem[:-3] + "-v2"
    candidates = [
        Path.home() / ".d4rl" / f"{stem}.hdf5",
        Path.home() / ".d4rl" / "datasets" / f"{stem}.hdf5",
    ]
    return next((path for path in candidates if path.exists()), None)


def load_cached_d4rl_dataset(task: str) -> D4RLDataset:
    """Load a cached D4RL HDF5 file without importing Gym or MuJoCo.

    Corrected Figure 2 only needs offline trajectories.  Keeping this loader free
    of environment imports makes the RTX 5090 training image considerably easier
    to reproduce and prevents an oracle from leaking into the evaluation protocol.
    """
    import h5py

    path = _infer_dataset_path(task)
    if path is None:
        raise FileNotFoundError(
            f"cached D4RL dataset for {task!r} was not found under ~/.d4rl; "
            "download it before starting the paid server run"
        )
    with h5py.File(path, "r") as handle:
        required = {"observations", "actions", "rewards", "terminals", "next_observations"}
        missing = required.difference(handle.keys())
        if missing:
            raise KeyError(f"{path} is missing required D4RL fields: {sorted(missing)}")
        observations = np.asarray(handle["observations"], dtype=np.float32)
        actions = np.asarray(handle["actions"], dtype=np.float32)
        rewards = np.asarray(handle["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.asarray(handle["terminals"], dtype=np.float32).reshape(-1, 1)
        next_observations = np.asarray(handle["next_observations"], dtype=np.float32)
        if "timeouts" in handle:
            timeouts = np.asarray(handle["timeouts"], dtype=np.float32).reshape(-1, 1)
        else:
            timeouts = np.zeros_like(terminals)
    n = observations.shape[0]
    arrays = (actions, rewards, terminals, timeouts, next_observations)
    if any(array.shape[0] != n for array in arrays):
        raise ValueError(f"inconsistent array lengths in {path}")
    return D4RLDataset(
        task=task,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        timeouts=timeouts,
        max_episode_steps=1000,
        source_path=path,
    )


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
