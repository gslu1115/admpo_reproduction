import numpy as np
import torch

from admpo_repro.data import D4RLDataset
from admpo_repro.evaluation.figure2 import build_test_windows, evaluate_model_rollout


def make_linear_episodes() -> D4RLDataset:
    episode_count, episode_length = 10, 12
    observations, actions, next_observations = [], [], []
    timeouts = []
    for _ in range(episode_count):
        state = np.arange(episode_length, dtype=np.float32)[:, None]
        observations.append(state)
        actions.append(np.ones((episode_length, 1), dtype=np.float32))
        next_observations.append(state + 1)
        timeout = np.zeros((episode_length, 1), dtype=np.float32)
        timeout[-1] = 1
        timeouts.append(timeout)
    n = episode_count * episode_length
    return D4RLDataset(
        task="linear",
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        rewards=np.zeros((n, 1), dtype=np.float32),
        next_observations=np.concatenate(next_observations),
        terminals=np.zeros((n, 1), dtype=np.float32),
        timeouts=np.concatenate(timeouts),
        max_episode_steps=episode_length,
    )


class ExactAnyStepModel:
    def dyna_dist(self, obs: torch.Tensor, actions: torch.Tensor):
        mean = obs + actions.sum(dim=1)
        std = torch.zeros_like(mean)
        reward = torch.zeros((obs.shape[0], 1), dtype=obs.dtype)
        return mean, std, reward, reward


class ExactRNNModel:
    def dyna_dist(self, obs: torch.Tensor, actions: torch.Tensor):
        mean = obs[:, -1] + actions[:, -1]
        std = torch.zeros_like(mean)
        reward = torch.zeros((obs.shape[0], 1), dtype=obs.dtype)
        return mean, std, reward, reward


class CenteredEnsembleModel:
    def dyna_dist(self, obs: torch.Tensor, action: torch.Tensor):
        center = obs + action
        offsets = torch.arange(-2, 3, dtype=obs.dtype)[:, None, None]
        means = center[None] + offsets
        stds = torch.zeros_like(means)
        rewards = torch.zeros((5, obs.shape[0], 1), dtype=obs.dtype)
        return means, stds, rewards, rewards


def test_test_windows_are_fixed_and_wholly_held_out():
    dataset = make_linear_episodes()
    split = dataset.trajectory_split(seed=9)
    first = build_test_windows(dataset, split, history=3, horizon=5, count=4, seed=11)
    second = build_test_windows(dataset, split, history=3, horizon=5, count=4, seed=11)
    np.testing.assert_array_equal(first.starts, second.starts)
    valid = set(split.valid_starts("test", 7).tolist())
    assert set(first.starts.tolist()).issubset(valid)
    np.testing.assert_array_equal(
        first.future_actions, np.ones((4, 5, 1), dtype=np.float32)
    )


def test_adm_rollout_uses_recorded_actions_and_autoregressive_means():
    dataset = make_linear_episodes()
    split = dataset.trajectory_split(seed=9)
    config = {
        "device": "cpu",
        "model": {"max_backtrack": 3},
        "evaluation": {
            "starts": 4,
            "horizon": 5,
            "window_seed": 11,
            "overflow": float(np.finfo(np.float32).max),
        },
    }
    rows = evaluate_model_rollout(
        "linear", "adm", ExactAnyStepModel(), dataset, split, 0, config
    )
    assert len(rows) == 5
    assert all(row["n_windows"] == 4 for row in rows)
    assert all(row["error_mean"] == 0.0 for row in rows)


def test_rnn_and_ensemble_use_deterministic_prediction_means():
    dataset = make_linear_episodes()
    split = dataset.trajectory_split(seed=9)
    config = {
        "device": "cpu",
        "model": {"max_backtrack": 3},
        "evaluation": {
            "starts": 4,
            "horizon": 5,
            "window_seed": 11,
            "overflow": float(np.finfo(np.float32).max),
        },
    }
    for kind, model in (
        ("rnn", ExactRNNModel()),
        ("ensemble", CenteredEnsembleModel()),
    ):
        rows = evaluate_model_rollout(
            "linear", kind, model, dataset, split, 0, config
        )
        assert all(row["error_mean"] == 0.0 for row in rows)
