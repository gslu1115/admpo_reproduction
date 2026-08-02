from admpo_repro.dynamics.training import _ensemble_member_improved


def test_ensemble_first_finite_validation_is_an_improvement() -> None:
    assert _ensemble_member_improved(0.1, float("inf"), 0.01)
    assert _ensemble_member_improved(0.98, 1.0, 0.01)
    assert not _ensemble_member_improved(0.995, 1.0, 0.01)
    assert not _ensemble_member_improved(float("nan"), 1.0, 0.01)
    assert not _ensemble_member_improved(float("inf"), 1.0, 0.01)
