import numpy as np
import torch

from admpo_repro.data import D4RLDataset
from admpo_repro.evaluation.figure2 import (
    _sample_prediction_step,
    build_test_windows,
    evaluate_model_rollout,
)


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
    def __init__(self):
        self.lengths = []

    def dyna_dist(self, obs: torch.Tensor, actions: torch.Tensor):
        self.lengths.append(actions.shape[1])
        mean = obs + actions.sum(dim=1)
        std = torch.zeros_like(mean)
        reward = torch.zeros((obs.shape[0], 1), dtype=obs.dtype)
        return mean, std, reward, reward


class ExactRNNModel:
    def __init__(self):
        self.lengths = []

    def dyna_dist(self, obs: torch.Tensor, actions: torch.Tensor):
        self.lengths.append(actions.shape[1])
        mean = obs[:, -1] + actions[:, -1]
        std = torch.zeros_like(mean)
        reward = torch.zeros((obs.shape[0], 1), dtype=obs.dtype)
        return mean, std, reward, reward


class ExactEnsembleModel:
    def dyna_dist(self, obs: torch.Tensor, action: torch.Tensor):
        center = obs + action
        means = center[None].repeat(5, 1, 1)
        stds = torch.zeros_like(means)
        rewards = torch.zeros((5, obs.shape[0], 1), dtype=obs.dtype)
        return means, stds, rewards, rewards


class UnitGaussianADM:
    def dyna_dist(self, obs: torch.Tensor, actions: torch.Tensor):
        mean = torch.zeros_like(obs)
        std = torch.ones_like(obs)
        reward = torch.zeros((obs.shape[0], 1), dtype=obs.dtype)
        return mean, std, reward, reward


def _config() -> dict:
    return {
        "device": "cpu",
        "model": {"max_backtrack": 3},
        "evaluation": {
            "starts": 4,
            "horizon": 5,
            "window_seed": 11,
            "rollout_seed": 17,
            "overflow": float(np.finfo(np.float32).max),
        },
    }


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


def test_adm_and_rnn_share_uniform_per_step_backtrack_sequence():
    dataset = make_linear_episodes()
    split = dataset.trajectory_split(seed=9)
    expected_rng = np.random.default_rng(17 + 11)
    expected = [int(expected_rng.integers(1, 4)) for _ in range(5)]
    assert len(set(expected)) > 1
    for kind, model in (("adm", ExactAnyStepModel()), ("rnn", ExactRNNModel())):
        rows = evaluate_model_rollout(
            "linear", kind, model, dataset, split, 0, _config()
        )
        assert model.lengths == expected
        assert [row["backtrack_k"] for row in rows] == expected
        assert all(row["n_windows"] == 4 for row in rows)
        assert all(row["error_mean"] == 0.0 for row in rows)


def test_ensemble_samples_elites_but_remains_exact_when_elites_agree():
    dataset = make_linear_episodes()
    split = dataset.trajectory_split(seed=9)
    rows = evaluate_model_rollout(
        "linear", "ensemble", ExactEnsembleModel(), dataset, split, 0, _config()
    )
    assert all(row["error_mean"] == 0.0 for row in rows)


def test_prediction_uses_gaussian_sample_not_mean():
    observation_history = np.zeros((2, 1, 3), dtype=np.float32)
    action_history = np.empty((2, 0, 1), dtype=np.float32)
    action = np.zeros((2, 1), dtype=np.float32)
    noise_seed = 123
    expected = np.random.default_rng(noise_seed).normal(size=(2, 3)).astype(
        np.float32
    )
    sampled, k = _sample_prediction_step(
        "adm",
        UnitGaussianADM(),
        observation_history,
        action_history,
        action,
        "cpu",
        1,
        np.random.default_rng(7),
        np.random.default_rng(noise_seed),
        np.random.default_rng(9),
    )
    assert k == 1
    np.testing.assert_allclose(sampled, expected)
    assert not np.allclose(sampled, 0.0)
