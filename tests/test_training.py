import numpy as np

from admpo_repro.dynamics.training import (
    _ensemble_member_improved,
    _ensemble_steps_per_epoch,
)


def test_ensemble_first_finite_validation_is_an_improvement() -> None:
    assert _ensemble_member_improved(0.1, float("inf"), 0.01)
    assert _ensemble_member_improved(0.98, 1.0, 0.01)
    assert not _ensemble_member_improved(0.995, 1.0, 0.01)
    assert not _ensemble_member_improved(float("nan"), 1.0, 0.01)
    assert not _ensemble_member_improved(float("inf"), 1.0, 0.01)


def test_ensemble_full_phase_steps_use_training_pool_size() -> None:
    train_pool = np.arange(1_025, dtype=np.int64)

    assert _ensemble_steps_per_epoch(None, train_pool, 256) == 4
    assert _ensemble_steps_per_epoch(None, train_pool[:100], 256) == 1
    assert _ensemble_steps_per_epoch(7, train_pool, 256) == 7
