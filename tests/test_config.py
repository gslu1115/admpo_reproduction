from admpo_repro.config import load_config
from admpo_repro.runner import _result_name, _scope


def test_deadline48h_scope_and_overrides_are_isolated() -> None:
    figure2 = load_config("figure2", "deadline48h")
    assert figure2["tasks"] == [
        "hopper-medium-replay-v2",
        "walker2d-medium-replay-v2",
    ]
    assert figure2["seeds"] == [0, 1, 2]
    assert figure2["training"]["max_epochs"] == 120
    assert figure2["bc"]["epochs"] == 100

    figure4 = load_config("figure4", "deadline48h")
    assert figure4["tasks"] == ["hopper-medium-replay-v2"]
    assert figure4["seeds"] == [0, 1]
    assert figure4["policy"]["epochs"] == 1250
    assert figure4["policy"]["scheduler_epochs"] == 5000

    assert _scope("deadline48h") == "deadline48h"
    assert _result_name("deadline48h", "task-v2", 1).startswith("deadline48h_")
