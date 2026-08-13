from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from admpo_repro.data import D4RLDataset, TrajectorySplit


class SeedManager:
    """Derive independent, stable uint32 seeds from named experiment streams."""

    def __init__(self, configured_seeds: dict[str, int]) -> None:
        self._configured = {str(key): int(value) for key, value in configured_seeds.items()}

    def seed(self, stream: str, *identity: Any) -> int:
        if stream not in self._configured:
            raise KeyError(f"unconfigured random stream: {stream}")
        payload = "|".join([stream, str(self._configured[stream]), *map(str, identity)])
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="little", signed=False)

    def rng(self, stream: str, *identity: Any) -> np.random.Generator:
        return np.random.default_rng(self.seed(stream, *identity))

    def manifest(self) -> dict[str, int]:
        return dict(self._configured)


@dataclass(frozen=True)
class InitialWindows:
    history_states: np.ndarray
    history_actions: np.ndarray
    initial_window_ids: np.ndarray
    current_transition_ids: np.ndarray
    trajectory_ids: np.ndarray

    @property
    def count(self) -> int:
        return int(self.history_states.shape[0])


def _episode_lookup(split: TrajectorySplit, transition_ids: np.ndarray) -> np.ndarray:
    lookup = np.full(max(stop for _, stop in split.episode_ranges), -1, dtype=np.int64)
    for episode_id, (start, stop) in enumerate(split.episode_ranges):
        lookup[start:stop] = episode_id
    result = lookup[np.asarray(transition_ids, dtype=np.int64)]
    if np.any(result < 0):
        raise RuntimeError("a sampled transition does not belong to an episode")
    return result


def sample_initial_windows(
    dataset: D4RLDataset,
    split: TrajectorySplit,
    max_backtrack: int,
    count: int,
    seed: int,
) -> InitialWindows:
    """Sample shared, no-replacement history windows from Figure 2's test split."""
    history = int(max_backtrack)
    if history < 1:
        raise ValueError("max_backtrack must be positive")
    starts = split.valid_starts("test", history)
    if starts.size < count:
        raise RuntimeError(
            f"requested {count} initial windows without replacement, but only "
            f"{starts.size} test windows are valid"
        )
    chosen = np.random.default_rng(int(seed)).choice(starts, size=count, replace=False)
    chosen = np.asarray(chosen, dtype=np.int64)
    offsets = np.arange(history, dtype=np.int64)
    state_indices = chosen[:, None] + offsets[None, :]
    action_indices = chosen[:, None] + np.arange(history - 1, dtype=np.int64)[None, :]
    current = chosen + history - 1
    trajectories = _episode_lookup(split, current)

    boundaries = dataset.boundaries.astype(bool).reshape(-1)
    if history > 1:
        internal = state_indices[:, :-1]
        if boundaries[internal].any():
            raise RuntimeError("initial history crosses a done/timeout boundary")
    for row, trajectory_id in zip(state_indices, trajectories):
        start, stop = split.episode_ranges[int(trajectory_id)]
        if int(row[0]) < start or int(row[-1]) >= stop:
            raise RuntimeError("initial history crosses an episode boundary")
    test_episode_ids = set(split.test_episodes)
    if any(int(value) not in test_episode_ids for value in trajectories):
        raise RuntimeError("initial history leaked outside Figure 2's test split")

    return InitialWindows(
        history_states=dataset.observations[state_indices].astype(np.float32, copy=False),
        history_actions=dataset.actions[action_indices].astype(np.float32, copy=False),
        initial_window_ids=chosen,
        current_transition_ids=current,
        trajectory_ids=trajectories,
    )
