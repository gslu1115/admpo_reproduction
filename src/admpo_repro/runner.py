from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from admpo_repro.config import load_config
from admpo_repro.data import load_d4rl_dataset
from admpo_repro.dynamics.training import (
    build_dynamics,
    load_dynamics_checkpoint,
    train_ensemble,
    train_sequence_model,
)
from admpo_repro.evaluation.figure2 import evaluate_model_rollout, write_figure2_rows
from admpo_repro.evaluation.figure4 import (
    collect_dataset_diagnostic,
    collect_rollout_points,
    write_figure4_rows,
)
from admpo_repro.evaluation.oracle import audit_oracle
from admpo_repro.manifest import environment_manifest, sha256, write_manifest
from admpo_repro.plotting import plot_figure2, plot_figure4
from admpo_repro.policies.admpo import load_learned_policy, train_admpo
from admpo_repro.policies.bc import load_bc, train_bc
from admpo_repro.runtime import seed_everything


def _scope(phase: str) -> str:
    if phase in ("smoke", "deadline48h"):
        return phase
    return "full"


def _result_prefix(phase: str) -> str:
    return f"{phase}_" if phase in ("smoke", "deadline48h") else ""


def _run_dir(root: Path, phase: str, task: str, seed: int) -> Path:
    return root / "runs" / _scope(phase) / task / f"seed-{seed}"


def _result_name(phase: str, task: str, seed: int) -> str:
    return f"{_result_prefix(phase)}{task}_seed-{seed}.csv"


def _train_figure2_models(
    task: str, seed: int, config: dict, resume: bool
) -> tuple[object, object, dict[str, Path]]:
    root = Path(config["root"])
    seed_everything(seed)
    dataset, env = load_d4rl_dataset(task, seed)
    run_dir = _run_dir(root, config["phase"], task, seed)
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {kind: model_dir / f"{kind}.pt" for kind in ("adm", "ensemble", "rnn")}
    bc_path = model_dir / "bc.pt"
    if not (resume and bc_path.exists()):
        train_bc(dataset, env, config, seed, bc_path, resume)
    for kind in ("adm", "rnn"):
        if resume and paths[kind].exists():
            continue
        model = build_dynamics(kind, dataset, config, config["device"])
        train_sequence_model(kind, model, dataset, config, seed, paths[kind], resume)
        del model
        torch.cuda.empty_cache()
    if not (resume and paths["ensemble"].exists()):
        model = build_dynamics("ensemble", dataset, config, config["device"])
        train_ensemble(model, dataset, config, seed, paths["ensemble"], resume)
        del model
        torch.cuda.empty_cache()
    paths["bc"] = bc_path
    return dataset, env, paths


def run_figure2(config: dict, resume: bool) -> None:
    root = Path(config["root"])
    manifest = environment_manifest(root)
    manifest.update({
        "experiment": "figure2",
        "phase": config["phase"],
        "seeds": config["seeds"],
        "tasks": config["tasks"],
        "config_sha256": sha256(root / "configs" / "figure2.yaml"),
        "datasets": {},
        "oracle_audits": {},
        "runs": [],
    })
    for task in config["tasks"]:
        for seed in config["seeds"]:
            run_started = datetime.now(timezone.utc).isoformat()
            run_clock = time.perf_counter()
            dataset, env, paths = _train_figure2_models(task, seed, config, resume)
            if seed == config["seeds"][0]:
                audit = audit_oracle(dataset, env, 1000 if config["phase"] != "smoke" else 32, seed)
                manifest["oracle_audits"][task] = audit.__dict__ | {"passed": audit.passed}
                if not audit.passed:
                    raise RuntimeError(f"MuJoCo oracle audit failed for {task}: {audit}")
                if dataset.source_path:
                    manifest["datasets"][task] = {"path": str(dataset.source_path), "sha256": sha256(dataset.source_path), "size": dataset.source_path.stat().st_size}
            policy = load_bc(dataset, env, config, paths["bc"])
            rows = []
            for kind in ("adm", "ensemble", "rnn"):
                model = load_dynamics_checkpoint(kind, dataset, config, paths[kind])
                rows.extend(evaluate_model_rollout(task, kind, model, policy, dataset, env, seed, config))
                del model
                torch.cuda.empty_cache()
            output = root / "results" / "figure2" / "per_seed" / _result_name(config["phase"], task, seed)
            write_figure2_rows(rows, output)
            env.close()
            manifest["runs"].append(
                {
                    "task": task,
                    "seed": seed,
                    "started_at": run_started,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": time.perf_counter() - run_clock,
                }
            )
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(root / "results" / "manifests" / f"figure2-{config['phase']}.json", manifest)
    plot_figure2(root, _result_prefix(config["phase"]))


def run_figure4(config: dict, resume: bool) -> None:
    root = Path(config["root"])
    fig2_config = load_config("figure2", config["phase"], config["seeds"])
    # Figure 4 needs both replay tasks even when Figure 2 pilot is task-restricted.
    fig2_config["tasks"] = config["tasks"]
    manifest = environment_manifest(root)
    manifest.update({
        "experiment": "figure4",
        "phase": config["phase"],
        "seeds": config["seeds"],
        "tasks": config["tasks"],
        "config_sha256": sha256(root / "configs" / "figure4.yaml"),
        "runs": [],
    })
    for task in config["tasks"]:
        for seed in config["seeds"]:
            run_started = datetime.now(timezone.utc).isoformat()
            run_clock = time.perf_counter()
            dataset, env, paths = _train_figure2_models(task, seed, fig2_config, resume)
            policy_dir = _run_dir(root, config["phase"], task, seed) / "admpo"
            final_policy = policy_dir / "final.pt"
            if not (resume and final_policy.exists()):
                final_policy = train_admpo(
                    dataset, env, config, seed, paths["adm"], policy_dir, resume
                )
            learned = load_learned_policy(dataset, env, config, final_policy)
            behavior_model = load_bc(dataset, env, fig2_config, paths["bc"])
            behavior = behavior_model.act
            rows, diagnostics = [], []
            for kind in ("adm", "ensemble"):
                model = load_dynamics_checkpoint(kind, dataset, fig2_config, paths[kind])
                for policy_name, policy in (
                    ("random", None), ("learned", learned), ("behavior", behavior)
                ):
                    rows.extend(
                        collect_rollout_points(
                            task, kind, model, policy_name, policy,
                            dataset, env, seed, config,
                        )
                    )
                diagnostics.extend(
                    collect_dataset_diagnostic(task, kind, model, dataset, env, seed, config)
                )
                del model
                torch.cuda.empty_cache()
            output = root / "results" / "figure4" / "points" / _result_name(config["phase"], task, seed)
            diagnostic_output = root / "results" / "figure4" / "diagnostics" / _result_name(config["phase"], task, seed)
            write_figure4_rows(rows, output)
            write_figure4_rows(diagnostics, diagnostic_output)
            env.close()
            manifest["runs"].append(
                {
                    "task": task,
                    "seed": seed,
                    "started_at": run_started,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": time.perf_counter() - run_clock,
                }
            )
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(root / "results" / "manifests" / f"figure4-{config['phase']}.json", manifest)
    plot_figure4(root, _result_prefix(config["phase"]))


def run_experiment(experiment: str, phase: str, seeds: list[int], resume: bool) -> None:
    config = load_config(experiment, phase, seeds)
    if experiment == "figure2":
        run_figure2(config, resume)
    elif experiment == "figure4":
        run_figure4(config, resume)
    else:
        raise ValueError(experiment)


def check_environment(phase: str = "smoke") -> dict:
    config = load_config("figure2", phase, [0])
    root = Path(config["root"])
    report = environment_manifest(root)
    report["tasks"] = {}
    for task in config["tasks"]:
        dataset, env = load_d4rl_dataset(task, 0)
        audit = audit_oracle(dataset, env, 32 if phase == "smoke" else 1000, 0)
        report["tasks"][task] = {
            "dataset_size": dataset.size,
            "observation_shape": list(dataset.observations.shape),
            "action_shape": list(dataset.actions.shape),
            "oracle": audit.__dict__ | {"passed": audit.passed},
        }
        env.close()
    output = root / "results" / "manifests" / "environment-check.json"
    write_manifest(output, report)
    return report


def plot_experiment(experiment: str, phase: str = "full") -> tuple[Path, Path]:
    root = Path(load_config(experiment, phase)["root"])
    prefix = _result_prefix(phase)
    return plot_figure2(root, prefix) if experiment == "figure2" else plot_figure4(root, prefix)
