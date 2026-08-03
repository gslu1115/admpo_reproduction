import numpy as np

from admpo_repro.data import D4RLDataset


def make_dataset() -> D4RLDataset:
    n = 8
    terminals = np.zeros((n, 1), dtype=np.float32)
    timeouts = np.zeros((n, 1), dtype=np.float32)
    timeouts[2] = 1
    terminals[6] = 1
    return D4RLDataset(
        task="test",
        observations=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        actions=np.zeros((n, 1), dtype=np.float32),
        rewards=np.zeros((n, 1), dtype=np.float32),
        next_observations=np.arange(2, (n + 1) * 2, dtype=np.float32).reshape(n, 2),
        terminals=terminals,
        timeouts=timeouts,
        max_episode_steps=3,
    )


def test_sequences_never_cross_episode_boundary():
    dataset = make_dataset()
    starts = dataset.valid_starts(3)
    # A terminal/timeout on the final transition is valid; crossing one is not.
    assert starts.tolist() == [0, 3, 4]
    sequences = dataset.sequences(starts, 3)
    assert sequences["observations"].shape == (3, 3, 2)


def test_statistics_replace_zero_standard_deviation():
    dataset = make_dataset()
    _, _, _, action_std = dataset.statistics()
    assert np.all(action_std == 1.0)


def test_trajectory_split_is_8_1_1_and_has_no_episode_overlap():
    episode_length = 2
    episode_count = 10
    n = episode_length * episode_count
    timeouts = np.zeros((n, 1), dtype=np.float32)
    timeouts[episode_length - 1 :: episode_length] = 1
    dataset = D4RLDataset(
        task="split-test",
        observations=np.arange(n, dtype=np.float32).reshape(n, 1),
        actions=np.ones((n, 1), dtype=np.float32),
        rewards=np.zeros((n, 1), dtype=np.float32),
        next_observations=np.arange(1, n + 1, dtype=np.float32).reshape(n, 1),
        terminals=np.zeros((n, 1), dtype=np.float32),
        timeouts=timeouts,
        max_episode_steps=episode_length,
    )
    split = dataset.trajectory_split(seed=17)
    assert len(split.train_episodes) == 8
    assert len(split.validation_episodes) == 1
    assert len(split.test_episodes) == 1
    partitions = [
        set(split.episode_ids(name))
        for name in ("train", "validation", "test")
    ]
    assert not (partitions[0] & partitions[1])
    assert not (partitions[0] & partitions[2])
    assert not (partitions[1] & partitions[2])
    assert set.union(*partitions) == set(range(episode_count))
    for name in ("train", "validation", "test"):
        starts = split.valid_starts(name, episode_length)
        assert all(index % episode_length == 0 for index in starts)


def test_training_statistics_can_exclude_validation_and_test():
    dataset = make_dataset()
    indices = np.asarray([0, 1, 2], dtype=np.int64)
    obs_mu, _, _, _ = dataset.statistics(indices)
    np.testing.assert_allclose(obs_mu, dataset.observations[indices].mean(axis=0))
