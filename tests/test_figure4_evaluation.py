from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from admpo_repro.data import D4RLDataset, TrajectorySplit
from admpo_repro.evaluation import figure4


def _dataset() -> tuple[D4RLDataset, TrajectorySplit]:
    observations = np.arange(24, dtype=np.float32).reshape(12, 2)
    actions = np.full((12, 2), 0.25, dtype=np.float32)
    boundaries = np.zeros((12, 1), dtype=np.float32)
    boundaries[[3, 7, 11]] = 1.0
    dataset = D4RLDataset(
        task="hopper-medium-replay-v2",
        observations=observations,
        actions=actions,
        rewards=np.zeros((12, 1), dtype=np.float32),
        next_observations=observations + 0.25,
        terminals=np.zeros((12, 1), dtype=np.float32),
        timeouts=boundaries,
        max_episode_steps=4,
    )
    split = TrajectorySplit(
        episode_ranges=((0, 4), (4, 8), (8, 12)),
        train_episodes=(0,),
        validation_episodes=(1,),
        test_episodes=(2,),
        seed=202405,
        ratios=(0.8, 0.1, 0.1),
    )
    return dataset, split


def test_rollout_batch_is_held_out_and_dataset_actions_are_contiguous() -> None:
    dataset, split = _dataset()
    batch = figure4.build_rollout_batch(
        dataset, split, starts=1, history=3, horizon=2, seed=7
    )
    assert batch.window_starts.tolist() == [8]
    np.testing.assert_array_equal(batch.observations[0], dataset.observations[8:11])
    np.testing.assert_array_equal(batch.previous_actions[0], dataset.actions[8:10])
    np.testing.assert_array_equal(batch.dataset_actions[0], dataset.actions[10:12])


def test_population_disagreement_and_gaussian_component_sampling() -> None:
    statistics = figure4.PredictionStatistics(
        mean_prediction=np.zeros((3, 2), dtype=np.float32),
        uncertainty_std=np.zeros(3, dtype=np.float32),
        uncertainty_var=np.zeros(3, dtype=np.float32),
        total_uncertainty_std=np.zeros(3, dtype=np.float32),
        total_uncertainty_var=np.zeros(3, dtype=np.float32),
        component_means=np.stack(
            (
                np.zeros((3, 2), dtype=np.float32),
                np.full((3, 2), 10.0, dtype=np.float32),
            )
        ),
        component_stds=np.zeros((2, 3, 2), dtype=np.float32),
    )
    sampled, choices = figure4.sample_next_state(
        statistics, np.random.default_rng(123)
    )
    assert choices.shape == (3,)
    assert sampled.shape == (3, 2)
    expected = statistics.component_means[choices, np.arange(3)]
    np.testing.assert_allclose(sampled, expected, atol=1e-5)


@dataclass
class _Space:
    low: np.ndarray
    high: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape


class _Environment:
    action_space = _Space(
        low=np.full(2, -1.0, dtype=np.float32),
        high=np.full(2, 1.0, dtype=np.float32),
    )


class _Oracle:
    def __init__(self, env: object) -> None:
        del env

    def batch_step(self, observations: np.ndarray, actions: np.ndarray):
        next_observations = observations + actions
        rewards = np.zeros((len(observations), 1), dtype=np.float32)
        dones = np.zeros((len(observations), 1), dtype=bool)
        return next_observations, rewards, dones


def test_trajectory_error_accumulates_but_local_error_stays_one_step(
    monkeypatch,
) -> None:
    dataset, split = _dataset()

    def fake_statistics(kind, model, obs_hist, act_hist, action, device):
        del kind, model, act_hist, device
        mean = obs_hist[:, -1] + action + 1.0
        component_means = mean[None]
        zeros = np.zeros(len(mean), dtype=np.float32)
        return figure4.PredictionStatistics(
            mean_prediction=mean,
            uncertainty_std=zeros,
            uncertainty_var=zeros,
            total_uncertainty_std=zeros,
            total_uncertainty_var=zeros,
            component_means=component_means,
            component_stds=np.zeros_like(component_means),
        )

    def deterministic_sample(statistics, rng):
        del rng
        return statistics.component_means[0].copy(), np.zeros(1, dtype=np.int64)

    monkeypatch.setattr(figure4, "MujocoOracle", _Oracle)
    monkeypatch.setattr(figure4, "prediction_statistics", fake_statistics)
    monkeypatch.setattr(figure4, "sample_next_state", deterministic_sample)
    monkeypatch.setattr(
        figure4,
        "termination",
        lambda task, observations: np.zeros(len(observations), dtype=bool),
    )
    config = {
        "device": "cpu",
        "model": {"max_backtrack": 2},
        "evaluation": {
            "starts": 1,
            "rollout_horizon": 2,
            "points_per_policy": 2,
            "sampling_seed": 202405,
            "max_collection_attempts": 1,
        },
    }
    rows = figure4.collect_rollout_points(
        dataset.task,
        "adm",
        object(),
        "dataset",
        None,
        dataset,
        split,
        _Environment(),
        0,
        config,
    )
    assert [row["rollout_step"] for row in rows] == [1, 2]
    assert np.isclose(rows[0]["trajectory_error_rmse"], 1.0)
    assert np.isclose(rows[1]["trajectory_error_rmse"], 2.0)
    assert np.isclose(rows[0]["local_error_rmse"], 1.0)
    assert np.isclose(rows[1]["local_error_rmse"], 1.0)
