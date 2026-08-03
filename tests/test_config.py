from admpo_repro.config import load_config
from admpo_repro.runner import _auto_worker_counts, _result_name, _scope


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


def test_legacy_figure4_deadline_scope_remains_isolated() -> None:
    figure4 = load_config("figure4", "deadline48h")
    assert figure4["tasks"] == ["hopper-medium-replay-v2"]
    assert figure4["seeds"] == [0, 1]
    assert figure4["policy"]["epochs"] == 1250
    assert _scope("deadline48h") == "deadline48h"
    assert _result_name("deadline48h", "task-v2", 1).startswith("deadline48h_")
