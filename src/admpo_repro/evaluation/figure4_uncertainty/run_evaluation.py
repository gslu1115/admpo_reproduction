from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from admpo_repro.config import ROOT, load_config
from admpo_repro.data import load_cached_d4rl_dataset
from admpo_repro.evaluation.oracle import MujocoOracle, audit_oracle_rmse
from admpo_repro.runtime import configure_mujoco_environment, seed_everything

from .adapters import ADMEvaluatorAdapter, EnsembleEvaluatorAdapter
from .checkpoints import (
    build_preflight_manifest,
    checkpoint_paths,
    load_adm,
    load_ensemble,
    load_fixed_bc,
    load_sac_actor,
)
from .outputs import (
    sanity_check_batch,
    scalar_slices,
    write_final_figure,
)
from .policies import (
    DeterministicBCPolicy,
    RandomUniformPolicy,
    SACPolicy,
    random_action_table,
)
from .protocol import SeedManager, sample_initial_windows
from .rollout import evaluate_native_rollout


def _default_output(root: Path, config: dict[str, Any]) -> Path:
    run_id = f"hopper_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return root / config["output"]["namespace"] / run_id


def _prepare_output(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite a non-empty evaluation directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_env(task: str):
    import mujoco_py  # noqa: F401  # Load the compiled extension before D4RL probes it.
    import d4rl  # noqa: F401  # Environment registration after MuJoCo paths are configured.
    import gym

    return gym.make(task)


def _vendor_termination(root: Path):
    vendor = root / "vendor" / "ADMPO"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from components.static_fns.hopper import StaticFns

    return StaticFns.termination_fn


def _validate_protocol(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    required = {
        "dynamics_types": ["adm", "ensemble"],
        "policies": ["random", "bc", "learned_sac"],
        "max_backtrack": 5,
        "endpoint_semantics": "after_h_model_transitions_then_one_step_probe",
        "rollout_state": "selected_predictor_gaussian_sample",
        "error": "five_predictor_mixture_expected_squared_error_raw_state_sum",
        "uncertainty": "paper_eq4_total_variance_l1",
        "replacement_or_top_up": False,
    }
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise RuntimeError(f"hard protocol constraint {key!r} must equal {expected!r}")
    if [int(value) for value in protocol.get("horizons", [])] != [5]:
        raise RuntimeError("hard protocol constraint 'horizons' must equal [5]")
    source_sampling = protocol.get("adm_source_sampling")
    if source_sampling != "vendor_batch_shared":
        raise RuntimeError("Figure 4 requires vendor_batch_shared ADM source sampling")
    if config["task"] != "hopper-medium-replay-v2":
        raise RuntimeError("Figure 4 must use the exact Figure 2 Hopper task")
    learned_cfg = config["policies"]["learned_sac"]
    if learned_cfg.get("sampling") != "actor_mean_then_tanh":
        raise RuntimeError(
            "Figure 4 requires deterministic learned-SAC actor-mean actions"
        )
    if learned_cfg.get("stochastic") is not False:
        raise RuntimeError(
            "Figure 4 requires learned-SAC stochastic=false"
        )


def _oracle_report(dataset, env, config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["oracle_audit"]
    audit = audit_oracle_rmse(
        dataset,
        env,
        points=int(cfg["samples"]),
        seed=int(config["random_seeds"]["initial_windows"]),
    )
    report = {
        "samples": audit.samples,
        "excluded_clipped": audit.excluded_clipped,
        "median_rmse": audit.median_rmse,
        "p99_rmse": audit.p99_rmse,
        "maximum_rmse": audit.maximum_rmse,
        "median_tolerance": float(cfg["median_rmse_tolerance"]),
        "p99_tolerance": float(cfg["p99_rmse_tolerance"]),
        "passed": (
            audit.median_rmse <= float(cfg["median_rmse_tolerance"])
            and audit.p99_rmse <= float(cfg["p99_rmse_tolerance"])
        ),
        "state_restoration_limitation": (
            "Hopper observations omit root x; root x is restored as 0. Dataset audit "
            "excludes observations with clipped qvel because they are not invertible."
        ),
    }
    if not report["passed"]:
        raise RuntimeError(f"MuJoCo one-step oracle audit failed: {report}")
    return report


def _ensure_mujoco_loader_environment(
    reexec_module: str = "admpo_repro.evaluation.figure4_uncertainty.run_evaluation",
    reexec_args: list[str] | None = None,
) -> None:
    """Re-exec once so native dlopen sees MuJoCo libraries at process start."""
    required = [str(Path.home() / ".mujoco" / "mujoco210" / "bin"), "/usr/lib/wsl/lib"]
    current = [value for value in os.environ.get("LD_LIBRARY_PATH", "").split(":") if value]
    if all(value in current for value in required):
        return
    if os.environ.get("ADMPO_FIGURE4_MUJOCO_REEXEC") == "1":
        raise RuntimeError("MuJoCo loader paths are still missing after the guarded re-exec")
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys([*current, *required]))
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    environment["ADMPO_FIGURE4_MUJOCO_REEXEC"] = "1"
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-m",
            reexec_module,
            *(sys.argv[1:] if reexec_args is None else reexec_args),
        ],
        environment,
    )


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    root = Path(config["root"]).resolve()
    _validate_protocol(config)
    seed_manager = SeedManager(config["random_seeds"])
    seed_everything(seed_manager.seed("initial_windows", "global_torch_state"))
    dataset = load_cached_d4rl_dataset(config["task"])
    split_cfg = config["dataset_split"]
    split = dataset.trajectory_split(
        train_ratio=float(split_cfg["train"]),
        validation_ratio=float(split_cfg["validation"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )
    windows = sample_initial_windows(
        dataset,
        split,
        max_backtrack=int(config["protocol"]["max_backtrack"]),
        count=int(config["evaluation"]["initial_windows"]),
        seed=seed_manager.seed("initial_windows", dataset.task),
    )
    build_preflight_manifest(config, root, dataset)

    env = _make_env(dataset.task)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    if not np.allclose(action_low, -action_high):
        raise RuntimeError("vendor SAC action mapping assumes symmetric action bounds")
    oracle = MujocoOracle(env)
    _oracle_report(dataset, env, config)
    termination_fn = _vendor_termination(root)

    bc_path = checkpoint_paths(config, root, 0)["bc"]
    bc_model = load_fixed_bc(dataset, bc_path, config["device"])
    if not np.array_equal(bc_model.action_low.detach().cpu().numpy(), action_low):
        raise RuntimeError("fixed BC action lower bounds do not match the environment")
    if not np.array_equal(bc_model.action_high.detach().cpu().numpy(), action_high):
        raise RuntimeError("fixed BC action upper bounds do not match the environment")

    scalar_results = []
    model_seeds = [int(value) for value in config["model_seeds"]]
    max_step = 6
    completed_groups = 0
    total_groups = (
        len(model_seeds)
        * len(config["protocol"]["dynamics_types"])
        * len(config["protocol"]["policies"])
    )

    for model_seed in model_seeds:
        seed_everything(model_seed)
        paths = checkpoint_paths(config, root, model_seed)
        adm_model = load_adm(dataset, paths["adm"], config["device"])
        ensemble_model = load_ensemble(dataset, paths["ensemble"], model_seed, config["device"])
        sac_actor = load_sac_actor(
            dataset,
            paths["learned_sac"],
            root,
            config["policies"]["learned_sac"]["hidden_dims"],
            config["device"],
        )
        random_table = random_action_table(
            seed_manager.rng("random_policy", model_seed),
            windows.count,
            max_step,
            action_low,
            action_high,
        )
        learned_policy = SACPolicy(
            sac_actor,
            action_low,
            action_high,
        )
        policies = {
            "random": RandomUniformPolicy(random_table, action_low, action_high),
            "bc": DeterministicBCPolicy(bc_model, action_low, action_high),
            "learned_sac": learned_policy,
        }
        adapters = {
            "adm": ADMEvaluatorAdapter(
                adm_model,
                dataset.obs_dim,
                config["device"],
                int(config["evaluation"]["inference_chunk"]),
            ),
            "ensemble": EnsembleEvaluatorAdapter(
                ensemble_model,
                dataset.obs_dim,
                config["device"],
                int(config["evaluation"]["inference_chunk"]),
            ),
        }
        for dynamics_type in config["protocol"]["dynamics_types"]:
            for policy_name in config["protocol"]["policies"]:
                source_stream = "adm_backtrack" if dynamics_type == "adm" else "ensemble_member"
                gaussian_stream = "adm_gaussian" if dynamics_type == "adm" else "ensemble_gaussian"
                batch = evaluate_native_rollout(
                    adapter=adapters[dynamics_type],
                    policy=policies[policy_name],
                    oracle=oracle,
                    termination_fn=termination_fn,
                    initial_windows=windows,
                    horizons=config["protocol"]["horizons"],
                    source_rng=seed_manager.rng(source_stream, model_seed, policy_name),
                    gaussian_rng=seed_manager.rng(gaussian_stream, model_seed, policy_name),
                    oracle_chunk=int(config["evaluation"]["oracle_chunk"]),
                    model_seed=model_seed,
                )
                reported_source_sampling = (
                    config["protocol"]["adm_source_sampling"]
                    if dynamics_type == "adm"
                    else "per_trajectory_elite"
                )
                sanity = sanity_check_batch(batch, source_sampling=reported_source_sampling)
                scalar_results.extend(scalar_slices(batch))
                completed_groups += 1
                progress = {
                    "status": "running",
                    "completed_groups": completed_groups,
                    "total_groups": total_groups,
                    "latest": {
                        "dynamics_type": dynamics_type,
                        "model_seed": model_seed,
                        "policy": policy_name,
                        "valid_counts": sanity["valid_counts"],
                    },
                }
                print(json.dumps(progress, ensure_ascii=False), flush=True)
        del adm_model, ensemble_model, sac_actor, adapters, policies
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    figure = write_final_figure(scalar_results, config, output_dir)
    summary = {
        "status": "complete",
        "phase": config["phase"],
        "task": dataset.task,
        "figure": figure,
        "sanity_checks_passed": True,
    }
    env.close()
    return summary


def run_phase(
    *,
    reexec_module: str = "admpo_repro",
    reexec_args: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Execute the fixed Figure 4 main evaluation protocol."""
    _ensure_mujoco_loader_environment(reexec_module, reexec_args)
    configure_mujoco_environment()
    config = load_config("figure4_uncertainty", "full")
    output_dir = _prepare_output(_default_output(ROOT, config))
    return output_dir, run(config, output_dir)
