from admpo_repro.config import load_config
from admpo_repro.runner import _auto_worker_counts, _figure4_worker_count


def test_figure2_formal_scope_is_the_corrected_three_seed_protocol() -> None:
    figure2 = load_config("figure2", "full")
    assert figure2["tasks"] == [
        "hopper-medium-replay-v2",
        "walker2d-medium-replay-v2",
    ]
    assert figure2["seeds"] == [0, 1, 2]
    assert figure2["split"] == {
        "train": 0.8,
        "validation": 0.1,
        "test": 0.1,
        "seed": 202405,
    }
    assert figure2["evaluation"]["horizon"] == 100
    assert figure2["evaluation"]["starts"] == 100


def test_pilot_is_one_formal_task_and_worker_limit_is_weighted() -> None:
    pilot = load_config("figure2", "pilot")
    assert pilot["tasks"] == ["hopper-medium-replay-v2"]
    assert pilot["seeds"] == [0]
    full = load_config("figure2", "full")
    assert _auto_worker_counts(full, 3) == (3, 3)


def test_figure4_uses_the_agreed_local_two_seed_protocol() -> None:
    figure4 = load_config("figure4", "full")
    assert figure4["tasks"] == ["hopper-medium-replay-v2"]
    assert figure4["seeds"] == [0, 1]
    assert figure4["policy"]["epochs"] == 5000
    assert figure4["evaluation"]["policies"] == ["random", "learned", "dataset"]
    assert figure4["evaluation"]["rollout_horizon"] == 5
    assert _figure4_worker_count(figure4, 0, 2) == 2
