import random

import numpy as np
import torch

from admpo_repro.runtime import capture_rng_state, restore_rng_state, seed_everything


def test_rng_restore_accepts_numpy_backed_torch_state() -> None:
    seed_everything(17)
    state = capture_rng_state()
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    state["torch"] = state["torch"].cpu().numpy()
    state.pop("cuda", None)
    restore_rng_state(state)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
