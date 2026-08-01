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
