from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def configure_mujoco_environment() -> None:
    """Configure mujoco-py for the current process without touching shell files."""
    mujoco_bin = Path.home() / ".mujoco" / "mujoco210" / "bin"
    candidates = [str(mujoco_bin), "/usr/lib/wsl/lib"]
    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(current + candidates))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng_state() -> dict:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints are normally loaded onto the training device.  In that case
    # PyTorch also maps RNG tensors to CUDA, while set_rng_state requires a CPU
    # ByteTensor.  Accept NumPy-backed legacy/checkpoint representations too.
    torch_state = torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu")
    torch.set_rng_state(torch_state)
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [
            torch.as_tensor(item, dtype=torch.uint8, device="cpu")
            for item in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
