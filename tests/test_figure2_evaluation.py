import numpy as np
import pytest
import torch
from pathlib import Path

from admpo_repro.data import D4RLDataset
from admpo_repro.evaluation.figure2 import (
    _sample_prediction_step,
    aggregate_repeat_rows,
    build_test_windows,
    evaluate_model_rollout,
    load_completed_repeat_rows,
    write_figure2_rows,
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
            "rollout_repeats": 2,
            "inference_chunk": 4096,
            "overflow": float(np.finfo(np.float32).max),
        },
    }


def test_test_windows_are_fixed_and_wholly_held_out():
    dataset = make_linear_episodes()
    split = dataset.trajectory_split(seed=9)
    first = build_test_windows(dataset, split, history=3, horizon=5, count=4, seed=11)
    second = build_test_windows(dataset, split, history=3, horizon=5, count=4, seed=11)
    np.testing.assert_array_equal(first.starts, second.starts)
    assert np.unique(first.starts).size == first.starts.size
    valid = set(split.valid_starts("test", 7).tolist())
    assert set(first.starts.tolist()).issubset(valid)
    np.testing.assert_array_equal(
        first.future_actions, np.ones((4, 5, 1), dtype=np.float32)
    )


def test_fixed_windows_fail_instead_of_sampling_with_replacement():
    dataset = make_linear_episodes()
    split = dataset.trajectory_split(seed=9)
    available = split.valid_starts("test", 7).size
    with pytest.raises(RuntimeError, match="fewer than"):
        build_test_windows(
            dataset, split, history=3, horizon=5, count=available + 1, seed=11
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


def test_chunking_keeps_one_shared_k_for_the_whole_window_batch():
    dataset = make_linear_episodes()
    split = dataset.trajectory_split(seed=9)
    config = _config()
    config["evaluation"]["inference_chunk"] = 2
    model = ExactAnyStepModel()
    rows = evaluate_model_rollout("linear", "adm", model, dataset, split, 0, config)
    ks = [int(row["backtrack_k"]) for row in rows]
    assert model.lengths == [value for k in ks for value in (k, k)]
    assert all(row["error_mean"] == 0.0 for row in rows)


def test_different_repeats_use_different_reproducible_k_streams():
    dataset = make_linear_episodes()
    split = dataset.trajectory_split(seed=9)
    first = evaluate_model_rollout(
        "linear", "adm", ExactAnyStepModel(), dataset, split, 0, _config(), repeat=0
    )
    second = evaluate_model_rollout(
        "linear", "adm", ExactAnyStepModel(), dataset, split, 0, _config(), repeat=1
    )
    first_ks = [row["backtrack_k"] for row in first]
    second_ks = [row["backtrack_k"] for row in second]
    assert first_ks != second_ks
    repeated = evaluate_model_rollout(
        "linear", "adm", ExactAnyStepModel(), dataset, split, 0, _config(), repeat=1
    )
    assert second_ks == [row["backtrack_k"] for row in repeated]


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


def test_repeat_aggregation_pools_within_seed_without_creating_extra_seeds():
    base = {
        "task": "linear",
        "seed": 0,
        "model": "adm",
        "rollout_length": 1,
        "backtrack_k": 1,
        "error_std": 0.0,
        "error_sem": 0.0,
        "n_windows": 4,
        "n_samples": 4,
        "rollout_repeats": 2,
        "repeat_mean_std": "",
        "repeat_mean_sem": "",
    }
    rows = [
        {**base, "repeat": 0, "error_mean": 1.0},
        {**base, "repeat": 1, "error_mean": 3.0},
    ]
    aggregated = aggregate_repeat_rows(rows, rollout_repeats=2)
    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["repeat"] == -1
    assert row["error_mean"] == 2.0
    assert row["error_std"] == 1.0
    assert row["n_windows"] == 4
    assert row["n_samples"] == 8
    assert row["rollout_repeats"] == 2


def test_repeat_csv_resume_keeps_only_complete_groups(tmp_path: Path):
    base = {
        "task": "linear",
        "seed": 0,
        "model": "adm",
        "rollout_length": 1,
        "backtrack_k": 1,
        "error_mean": 1.0,
        "error_std": 0.0,
        "error_sem": 0.0,
        "n_windows": 4,
        "n_samples": 4,
        "rollout_repeats": 2,
        "repeat_mean_std": "",
        "repeat_mean_sem": "",
    }
    path = tmp_path / "repeat.csv"
    write_figure2_rows(
        [{**base, "repeat": 0}, {**base, "repeat": 1}], path
    )
    complete = load_completed_repeat_rows(path, horizon=1, n_windows=4, rollout_repeats=2)
    assert {(row["model"], int(row["repeat"])) for row in complete} == {
        ("adm", 0),
        ("adm", 1),
    }
    assert load_completed_repeat_rows(
        path, horizon=2, n_windows=4, rollout_repeats=2
    ) == []
