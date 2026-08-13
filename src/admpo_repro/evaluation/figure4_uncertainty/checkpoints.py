from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from admpo_repro.config import load_config
from admpo_repro.data import D4RLDataset
from admpo_repro.dynamics.models import ADMDynamics
from admpo_repro.dynamics.training import load_dynamics_checkpoint
from admpo_repro.policies.bc import BCPolicy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(*arrays: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _resolve(root: Path, configured: str) -> Path:
    path = (root / configured).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint is missing: {path}")
    return path


def checkpoint_paths(config: dict[str, Any], root: Path, seed: int) -> dict[str, Path]:
    key = str(int(seed))
    checkpoints = config["checkpoints"]
    paths = {
        "adm": _resolve(root, checkpoints["adm"][key]),
        "learned_sac": _resolve(root, checkpoints["learned_sac"][key]),
        "ensemble": _resolve(root, checkpoints["ensemble"][key]),
        "bc": _resolve(root, checkpoints["bc"]),
    }
    for name in ("adm", "learned_sac", "ensemble"):
        match = re.search(r"seed[-_](\d+)", paths[name].as_posix())
        if match is None or int(match.group(1)) != int(seed):
            raise RuntimeError(f"{name} checkpoint does not match model seed {seed}: {paths[name]}")
    return paths


def build_preflight_manifest(
    config: dict[str, Any], root: Path, dataset: D4RLDataset
) -> dict[str, Any]:
    configured_seeds = sorted(int(key) for key in config["checkpoints"]["adm"])
    if configured_seeds != [0, 1, 2]:
        raise RuntimeError(f"the main experiment requires checkpoint seeds [0,1,2], got {configured_seeds}")
    manifest: dict[str, Any] = {
        "environment": {
            "env_name": dataset.task,
            "dataset_path": str(dataset.source_path.resolve()) if dataset.source_path else None,
            "dataset_sha256": sha256_file(dataset.source_path.resolve()) if dataset.source_path else None,
            "transitions": dataset.size,
            "observation_dim": dataset.obs_dim,
            "action_dim": dataset.action_dim,
        },
        "adm": {},
        "ensemble": {},
    }
    bc_seen: Path | None = None
    for seed in configured_seeds:
        paths = checkpoint_paths(config, root, seed)
        bc_seen = paths["bc"]
        adm_state = torch.load(paths["adm"], map_location="cpu")
        required_stats = ("obs_mu", "obs_std", "act_mu", "act_std")
        if any(key not in adm_state for key in required_stats):
            raise RuntimeError(f"ADM seed {seed} checkpoint lacks normalization parameters")
        actor_state = torch.load(paths["learned_sac"], map_location="cpu")
        if "actor" not in actor_state:
            raise RuntimeError(f"SAC seed {seed} checkpoint has no actor state")
        ensemble_state = torch.load(paths["ensemble"], map_location="cpu")
        if ensemble_state.get("kind") != "ensemble":
            raise RuntimeError(f"seed {seed} Figure 2 checkpoint is not an ensemble")
        if int(ensemble_state.get("seed", -1)) != seed or ensemble_state.get("task") != dataset.task:
            raise RuntimeError(f"seed/task mismatch in {paths['ensemble']}")
        model_state = ensemble_state.get("model", {})
        first_weight = model_state.get("model.layers.0.weight")
        elite_tensor = model_state.get("model.elite_indices")
        if first_weight is None or int(first_weight.shape[0]) != 7:
            raise RuntimeError(f"Ensemble seed {seed} does not contain seven members")
        if elite_tensor is None:
            raise RuntimeError(f"Ensemble seed {seed} has no saved elite indices")
        elite_ids = [int(value) for value in elite_tensor.tolist()]
        if len(elite_ids) != 5 or len(set(elite_ids)) != 5:
            raise RuntimeError(f"invalid elite set for seed {seed}: {elite_ids}")
        saved_elites = [int(value) for value in ensemble_state.get("elites", elite_ids)]
        if saved_elites != elite_ids:
            raise RuntimeError(f"elite metadata/state mismatch for seed {seed}")
        for key in required_stats:
            if key not in model_state:
                raise RuntimeError(f"Ensemble seed {seed} lacks checkpoint normalization {key}")
        manifest["adm"][str(seed)] = {
            "dynamics": str(paths["adm"]),
            "dynamics_sha256": sha256_file(paths["adm"]),
            "actor": str(paths["learned_sac"]),
            "actor_sha256": sha256_file(paths["learned_sac"]),
            "normalization_source": "dyna checkpoint state_dict",
            "normalization_sha256": _array_sha256(*(adm_state[key] for key in required_stats)),
            "model": "ADMDynamics(hidden=200,rnn_layers=3,residual_blocks=4,max_backtrack=5)",
        }
        manifest["ensemble"][str(seed)] = {
            "checkpoint": str(paths["ensemble"]),
            "checkpoint_sha256": sha256_file(paths["ensemble"]),
            "members": 7,
            "elite_ids": elite_ids,
            "normalization_source": "Figure 2 checkpoint model state_dict",
            "normalization_sha256": _array_sha256(*(model_state[key] for key in required_stats)),
        }
    if bc_seen is None:
        raise RuntimeError("BC checkpoint resolution unexpectedly failed")
    bc_state = torch.load(bc_seen, map_location="cpu")
    if bc_state.get("task") != dataset.task or bc_state.get("observation_normalization") != "none":
        raise RuntimeError("BC checkpoint task/normalization does not match the fixed protocol")
    if bc_state.get("identity", {}).get("dataset", {}).get("sha256") != manifest["environment"]["dataset_sha256"]:
        raise RuntimeError("BC checkpoint was trained against a different dataset artifact")
    manifest["bc"] = {
        "checkpoint": str(bc_seen),
        "checkpoint_sha256": sha256_file(bc_seen),
        "deterministic": True,
        "seed": int(bc_state["seed"]),
        "hidden_dims": [int(value) for value in bc_state["hidden_dims"]],
        "observation_normalization": bc_state["observation_normalization"],
        "best_validation_mse": float(bc_state["best_validation_mse"]),
    }
    static_path = root / "vendor" / "ADMPO" / "components" / "static_fns" / "hopper.py"
    if not static_path.is_file():
        raise FileNotFoundError(static_path)
    manifest["termination"] = {
        "implementation": str(static_path.resolve()),
        "sha256": sha256_file(static_path.resolve()),
        "role": "sampled model state controls rollout survival",
    }
    manifest["true_dynamics_oracle"] = {
        "implementation": "admpo_repro.evaluation.oracle.MujocoOracle.batch_step",
        "root_x": 0.0,
        "gym_done_role": "diagnostic_only",
        "time_limit_truncation": "ignored; each oracle query is one step after reset",
    }
    return manifest


def load_adm(dataset: D4RLDataset, path: Path, device: str) -> ADMDynamics:
    state = torch.load(path, map_location=device)
    converted = {key.replace(".layer_norm.", ".norm."): value for key, value in state.items()}
    model = ADMDynamics(
        dataset.obs_dim,
        dataset.action_dim,
        hidden_dim=200,
        rnn_layers=3,
        residual_blocks=4,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(converted, strict=True)
    # The normalization tensors are part of the checkpoint and must not be overwritten.
    model.eval()
    return model


def load_ensemble(dataset: D4RLDataset, path: Path, seed: int, device: str):
    config = load_config("figure2", "full", [seed])
    config["device"] = device
    model = load_dynamics_checkpoint("ensemble", dataset, config, path)
    model.eval()
    return model


def load_sac_actor(
    dataset: D4RLDataset,
    path: Path,
    root: Path,
    hidden_dims: list[int],
    device: str,
) -> torch.nn.Module:
    """Load only the official SAC actor needed by evaluation.

    Constructing the full ADMPO agent also creates critics, dynamics, optimizers,
    and replay-training state.  None of those objects participate in Figure 4.
    """
    vendor = root / "vendor" / "ADMPO"
    if not (vendor / "components" / "actor.py").is_file():
        raise FileNotFoundError(
            "vendor/ADMPO is missing; run git submodule update --init --recursive"
        )
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from components.actor import ProbActor

    actor = ProbActor(
        (dataset.obs_dim,),
        [int(value) for value in hidden_dims],
        dataset.action_dim,
    ).to(device)
    state = torch.load(path, map_location=device)
    actor.load_state_dict(state["actor"], strict=True)
    actor.eval()
    return actor


def load_fixed_bc(dataset: D4RLDataset, path: Path, device: str) -> BCPolicy:
    state = torch.load(path, map_location=device)
    policy = BCPolicy(
        dataset.obs_dim,
        np.asarray(state["action_low"], dtype=np.float32),
        np.asarray(state["action_high"], dtype=np.float32),
        state["hidden_dims"],
    ).to(device)
    policy.load_state_dict(state["model"], strict=True)
    policy.eval()
    return policy
