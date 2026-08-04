from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from admpo_repro.config import load_config
from admpo_repro.data import load_cached_d4rl_dataset, load_d4rl_dataset
from admpo_repro.dynamics.training import (
    build_dynamics,
    load_dynamics_checkpoint,
    train_ensemble,
    train_sequence_model,
)
from admpo_repro.evaluation.figure2 import (
    aggregate_repeat_rows,
    build_test_windows,
    evaluate_model_rollout,
    load_completed_repeat_rows,
    write_figure2_rows,
)
from admpo_repro.evaluation.figure4 import (
    collect_dataset_diagnostic,
    collect_rollout_points,
    write_figure4_rows,
)
from admpo_repro.manifest import environment_manifest, sha256, write_manifest
from admpo_repro.plotting import plot_figure2, plot_figure4
from admpo_repro.policies.admpo import load_learned_policy, train_admpo
from admpo_repro.runtime import seed_everything


FIGURE2_KINDS = ("adm", "rnn", "ensemble")


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


def _figure2_scope(config: dict) -> str:
    # A pilot is a formal job and is therefore reusable by the full run.
    return "smoke" if config["phase"] == "smoke" else "formal"


def _figure2_model_dir(root: Path, config: dict, task: str, seed: int) -> Path:
    return (
        root
        / "runs"
        / "figure2"
        / config["artifact_namespace"]
        / _figure2_scope(config)
        / task
        / f"seed-{seed}"
        / "models"
    )


def _figure2_result_dir(root: Path, config: dict) -> Path:
    path = root / "results" / "figure2" / config["artifact_namespace"]
    evaluation_namespace = config.get("evaluation_namespace")
    if evaluation_namespace:
        path /= evaluation_namespace
    return path / config["phase"]


def _figure2_split(dataset, config: dict):
    split_cfg = config["split"]
    return dataset.trajectory_split(
        train_ratio=float(split_cfg["train"]),
        validation_ratio=float(split_cfg["validation"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )


def _figure2_paths(root: Path, config: dict, task: str, seed: int) -> dict[str, Path]:
    model_dir = _figure2_model_dir(root, config, task, seed)
    return {kind: model_dir / f"{kind}.pt" for kind in FIGURE2_KINDS}


def _train_figure2_job(
    config: dict, task: str, seed: int, kind: str, resume: bool
) -> dict:
    """Process-safe model-level worker used by the RTX 5090 scheduler."""
    started_at = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    seed_everything(seed)
    torch.set_num_threads(int(config.get("parallel", {}).get("cpu_threads_per_worker", 4)))
    dataset = load_cached_d4rl_dataset(task)
    split = _figure2_split(dataset, config)
    path = _figure2_paths(Path(config["root"]), config, task, seed)[kind]
    if resume and path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("split") != split.to_dict():
            raise RuntimeError(f"existing checkpoint {path} uses a different trajectory split")
        return {
            "task": task,
            "seed": seed,
            "model": kind,
            "status": "reused",
            "checkpoint": str(path),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.perf_counter() - clock,
            "peak_gpu_memory_gb": 0.0,
        }
    if torch.cuda.is_available() and str(config["device"]).startswith("cuda"):
        torch.cuda.set_device(torch.device(config["device"]))
        torch.cuda.reset_peak_memory_stats(config["device"])
    model = build_dynamics(
        kind,
        dataset,
        config,
        config["device"],
        statistics_indices=split.indices("train"),
    )
    if kind == "ensemble":
        train_ensemble(model, dataset, config, seed, path, resume, split)
    else:
        train_sequence_model(kind, model, dataset, config, seed, path, resume, split)
    peak = 0.0
    if torch.cuda.is_available() and str(config["device"]).startswith("cuda"):
        peak = torch.cuda.max_memory_allocated(config["device"]) / 1024**3
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "task": task,
        "seed": seed,
        "model": kind,
        "status": "trained",
        "checkpoint": str(path),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - clock,
        "peak_gpu_memory_gb": peak,
    }


def _auto_worker_counts(config: dict, requested_workers: int) -> tuple[int, int]:
    if config["phase"] == "pilot" or (
        config["phase"] == "smoke" and requested_workers <= 0
    ):
        return 1, 1
    parallel = config.get("parallel", {})
    if requested_workers > 0:
        sequence_workers = requested_workers
    else:
        total_gb = 0.0
        if torch.cuda.is_available() and str(config["device"]).startswith("cuda"):
            total_gb = torch.cuda.get_device_properties(config["device"]).total_memory / 1024**3
        if total_gb >= 28:
            sequence_workers = int(parallel.get("sequence_workers_32gb", 4))
        elif total_gb >= 14:
            sequence_workers = 2
        else:
            sequence_workers = 1
    sequence_workers = max(1, min(sequence_workers, int(parallel.get("max_workers", 4))))
    ensemble_workers = max(
        1,
        min(
            sequence_workers,
            int(parallel.get("ensemble_workers_32gb", 2)),
        ),
    )
    return sequence_workers, ensemble_workers


def _run_training_stage(jobs: list[tuple], workers: int) -> list[dict]:
    if workers == 1:
        results = []
        for job in jobs:
            result = _train_figure2_job(*job)
            print(json.dumps({"event": "figure2_job_finished", **result}, ensure_ascii=False), flush=True)
            results.append(result)
        return results
    results: list[dict] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as executor:
        futures = {executor.submit(_train_figure2_job, *job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(json.dumps({"event": "figure2_job_finished", **result}, ensure_ascii=False), flush=True)
            results.append(result)
    return results


def run_figure2(config: dict, resume: bool, workers: int = 0) -> None:
    root = Path(config["root"])
    manifest = environment_manifest(root)
    manifest.update(
        {
            "experiment": "figure2",
            "protocol": config["protocol"],
            "artifact_namespace": config["artifact_namespace"],
            "evaluation_namespace": config.get("evaluation_namespace"),
            "phase": config["phase"],
            "seeds": config["seeds"],
            "tasks": config["tasks"],
            "config_sha256": sha256(root / "configs" / "figure2.yaml"),
            "datasets": {},
            "splits": {},
            "evaluation_windows": {},
            "training_jobs": [],
            "evaluations": [],
        }
    )
    manifest["rollout_semantics"] = {
        "adm_rnn_backtrack": (
            "one scalar k sampled uniformly from [1,m] per repeat and batched "
            "rollout step; ADM and RNN share the same k stream"
        ),
        "ensemble_member": (
            "one elite sampled independently per trajectory and rollout step"
        ),
        "state_prediction": "Gaussian sample mean + standard_normal * std",
        "rollout_seed": int(config["evaluation"]["rollout_seed"]),
        "fixed_window_count": int(config["evaluation"]["starts"]),
        "rollout_repeats": int(config["evaluation"]["rollout_repeats"]),
        "inference_chunk": int(config["evaluation"]["inference_chunk"]),
        "plot_statistical_unit": "training seed; repeats are pooled within seed",
    }
    sequence_workers, ensemble_workers = _auto_worker_counts(config, workers)
    manifest["parallel"] = {
        "requested_workers": workers,
        "sequence_workers": sequence_workers,
        "ensemble_workers": ensemble_workers,
    }
    sequence_jobs = [
        (config, task, seed, kind, resume)
        for task in config["tasks"]
        for seed in config["seeds"]
        for kind in ("adm", "rnn")
    ]
    ensemble_jobs = [
        (config, task, seed, "ensemble", resume)
        for task in config["tasks"]
        for seed in config["seeds"]
    ]
    manifest["training_jobs"].extend(
        _run_training_stage(sequence_jobs, sequence_workers)
    )
    manifest["training_jobs"].extend(
        _run_training_stage(ensemble_jobs, ensemble_workers)
    )

    result_dir = _figure2_result_dir(root, config)
    for task in config["tasks"]:
        dataset = load_cached_d4rl_dataset(task)
        split = _figure2_split(dataset, config)
        windows = build_test_windows(
            dataset,
            split,
            int(config["model"]["max_backtrack"]),
            int(config["evaluation"]["horizon"]),
            int(config["evaluation"]["starts"]),
            int(config["evaluation"]["window_seed"]),
        )
        if dataset.source_path is not None:
            manifest["datasets"][task] = {
                "path": str(dataset.source_path),
                "sha256": sha256(dataset.source_path),
                "size": dataset.source_path.stat().st_size,
            }
        manifest["splits"][task] = split.to_dict()
        manifest["evaluation_windows"][task] = {
            "count": int(windows.starts.size),
            "candidate_count": int(windows.candidate_count),
            "sampled_without_replacement": True,
            "horizon": int(config["evaluation"]["horizon"]),
            "rollout_repeats": int(config["evaluation"]["rollout_repeats"]),
            "start_indices": windows.starts.tolist(),
            "start_indices_sha256": hashlib.sha256(windows.starts.tobytes()).hexdigest(),
        }
        for seed in config["seeds"]:
            evaluation_clock = time.perf_counter()
            rollout_repeats = int(config["evaluation"]["rollout_repeats"])
            repeat_output = (
                result_dir / "per_repeat" / f"{task}_seed-{seed}.csv"
            )
            repeat_rows = (
                load_completed_repeat_rows(
                    repeat_output,
                    int(config["evaluation"]["horizon"]),
                    int(windows.starts.size),
                    rollout_repeats,
                )
                if resume
                else []
            )
            completed = {
                (str(row["model"]), int(row["repeat"])) for row in repeat_rows
            }
            resumed_repeats = len(completed)
            paths = _figure2_paths(root, config, task, seed)
            for kind in FIGURE2_KINDS:
                missing_repeats = [
                    repeat
                    for repeat in range(rollout_repeats)
                    if (kind, repeat) not in completed
                ]
                if not missing_repeats:
                    continue
                model = load_dynamics_checkpoint(kind, dataset, config, paths[kind])
                for repeat in missing_repeats:
                    repeat_clock = time.perf_counter()
                    new_rows = evaluate_model_rollout(
                        task,
                        kind,
                        model,
                        dataset,
                        split,
                        seed,
                        config,
                        windows,
                        repeat=repeat,
                    )
                    repeat_rows.extend(new_rows)
                    write_figure2_rows(repeat_rows, repeat_output)
                    completed.add((kind, repeat))
                    print(
                        json.dumps(
                            {
                                "event": "figure2_evaluation_repeat_finished",
                                "task": task,
                                "seed": seed,
                                "model": kind,
                                "repeat_completed": repeat + 1,
                                "repeats_total": rollout_repeats,
                                "windows": int(windows.starts.size),
                                "duration_seconds": time.perf_counter() - repeat_clock,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rows = aggregate_repeat_rows(repeat_rows, rollout_repeats)
            output = result_dir / "per_seed" / f"{task}_seed-{seed}.csv"
            write_figure2_rows(rows, output)
            manifest["evaluations"].append(
                {
                    "task": task,
                    "seed": seed,
                    "output": str(output),
                    "repeat_output": str(repeat_output),
                    "rollout_repeats": rollout_repeats,
                    "windows_per_repeat": int(windows.starts.size),
                    "stochastic_rollouts": int(windows.starts.size) * rollout_repeats,
                    "resumed_model_repeats": resumed_repeats,
                    "duration_seconds": time.perf_counter() - evaluation_clock,
                }
            )
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_name = f"figure2-{config['artifact_namespace']}"
    if config.get("evaluation_namespace"):
        manifest_name += f"-{config['evaluation_namespace']}"
    manifest_path = (
        root
        / "results"
        / "manifests"
        / f"{manifest_name}-{config['phase']}.json"
    )
    write_manifest(manifest_path, manifest)
    plot_figure2(root, result_dir=result_dir)


def _figure4_scope(config: dict) -> str:
    return "smoke" if config["phase"] == "smoke" else "formal"


def _figure4_run_dir(root: Path, config: dict, task: str, seed: int) -> Path:
    return (
        root
        / "runs"
        / "figure4"
        / config["artifact_namespace"]
        / _figure4_scope(config)
        / task
        / f"seed-{seed}"
    )


def _figure4_result_dir(root: Path, config: dict) -> Path:
    return (
        root
        / "results"
        / "figure4"
        / config["artifact_namespace"]
        / config["phase"]
    )


def _prepare_figure4_models(
    task: str, seed: int, fig2_config: dict
) -> tuple[object, object, object, dict[str, Path]]:
    """Load, but never retrain, corrected Figure 2 dynamics checkpoints."""

    root = Path(fig2_config["root"])
    seed_everything(seed)
    dataset, env = load_d4rl_dataset(task, seed)
    split = _figure2_split(dataset, fig2_config)
    paths = _figure2_paths(root, fig2_config, task, seed)
    expected_split = split.to_dict()
    for kind in ("adm", "ensemble"):
        path = paths[kind]
        if not path.exists():
            raise FileNotFoundError(
                f"Figure 4 requires the corrected Figure 2 checkpoint: {path}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("split") != expected_split:
            raise RuntimeError(
                f"refusing Figure 4 checkpoint with a different trajectory split: {path}"
            )
    return dataset, env, split, paths


def _run_figure4_job(config: dict, task: str, seed: int, resume: bool) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    seed_everything(seed)
    torch.set_num_threads(4)
    fig2_config = load_config("figure2", "full", [seed])
    dataset, env, split, paths = _prepare_figure4_models(
        task, seed, fig2_config
    )
    try:
        root = Path(config["root"])
        policy_dir = _figure4_run_dir(root, config, task, seed) / "admpo"
        final_policy = policy_dir / "final.pt"
        if not (resume and final_policy.exists()):
            final_policy = train_admpo(
                dataset, env, config, seed, paths["adm"], policy_dir, resume
            )
        learned = load_learned_policy(dataset, env, config, final_policy)
        rows, diagnostics = [], []
        for kind in ("adm", "ensemble"):
            model = load_dynamics_checkpoint(
                kind, dataset, fig2_config, paths[kind]
            )
            for policy_name, policy in (
                ("random", None),
                ("learned", learned),
                ("dataset", None),
            ):
                rows.extend(
                    collect_rollout_points(
                        task,
                        kind,
                        model,
                        policy_name,
                        policy,
                        dataset,
                        split,
                        env,
                        seed,
                        config,
                    )
                )
            diagnostics.extend(
                collect_dataset_diagnostic(
                    task,
                    kind,
                    model,
                    dataset,
                    split,
                    env,
                    seed,
                    config,
                )
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result_dir = _figure4_result_dir(root, config)
        output = result_dir / "points" / f"{task}_seed-{seed}.csv"
        diagnostic_output = (
            result_dir / "diagnostics" / f"{task}_seed-{seed}.csv"
        )
        write_figure4_rows(rows, output)
        write_figure4_rows(diagnostics, diagnostic_output)
        return {
            "task": task,
            "seed": seed,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.perf_counter() - clock,
            "points": len(rows),
            "diagnostic_points": len(diagnostics),
            "policy_checkpoint": str(final_policy),
            "dynamics_checkpoints": {
                kind: {"path": str(paths[kind]), "sha256": sha256(paths[kind])}
                for kind in ("adm", "ensemble")
            },
            "output": str(output),
            "diagnostic_output": str(diagnostic_output),
        }
    finally:
        env.close()


def _figure4_worker_count(config: dict, requested_workers: int, jobs: int) -> int:
    if config["phase"] in ("smoke", "pilot"):
        return 1
    configured = int(config.get("parallel", {}).get("max_workers", 2))
    workers = requested_workers if requested_workers > 0 else configured
    return max(1, min(workers, configured, jobs))


def _run_figure4_stage(jobs: list[tuple], workers: int) -> list[dict]:
    if workers == 1:
        results = []
        for job in jobs:
            result = _run_figure4_job(*job)
            print(
                json.dumps(
                    {"event": "figure4_job_finished", **result},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            results.append(result)
        return results
    context = multiprocessing.get_context("spawn")
    results: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as executor:
        futures = {executor.submit(_run_figure4_job, *job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(
                json.dumps(
                    {"event": "figure4_job_finished", **result},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            results.append(result)
    return results


def run_figure4(config: dict, resume: bool, workers: int = 0) -> None:
    root = Path(config["root"])
    jobs = [
        (config, task, seed, resume)
        for task in config["tasks"]
        for seed in config["seeds"]
    ]
    worker_count = _figure4_worker_count(config, workers, len(jobs))
    manifest = environment_manifest(root)
    manifest.update(
        {
            "experiment": "figure4",
            "protocol": config["protocol"],
            "artifact_namespace": config["artifact_namespace"],
            "phase": config["phase"],
            "seeds": config["seeds"],
            "tasks": config["tasks"],
            "config_sha256": sha256(root / "configs" / "figure4.yaml"),
            "primary_metric": {
                "error": "cumulative trajectory raw-state RMSE",
                "uncertainty": "RMS disagreement of predictive means",
            },
            "action_sources": {
                "random": "uniform legal action sampled independently per step",
                "learned": "deterministic ADMPO policy action at current model state",
                "dataset": "open-loop contiguous held-out D4RL action sequence",
            },
            "secondary_metrics": [
                "local one-step RMSE versus MuJoCo reconstructed at the model state",
                "trajectory and local MSE versus predictive-mean variance",
                "aleatoric-plus-epistemic total uncertainty",
                "within-policy and within-rollout-step centered correlations",
            ],
            "parallel_workers": worker_count,
            "runs": [],
        }
    )
    manifest["runs"] = _run_figure4_stage(jobs, worker_count)
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = (
        root
        / "results"
        / "manifests"
        / f"figure4-{config['artifact_namespace']}-{config['phase']}.json"
    )
    write_manifest(manifest_path, manifest)
    plot_figure4(root, result_dir=_figure4_result_dir(root, config))


def run_experiment(
    experiment: str,
    phase: str,
    seeds: list[int] | None,
    resume: bool,
    workers: int = 0,
) -> None:
    config = load_config(experiment, phase, seeds)
    if experiment == "figure2":
        run_figure2(config, resume, workers)
    elif experiment == "figure4":
        run_figure4(config, resume, workers)
    else:
        raise ValueError(experiment)


def check_environment(phase: str = "smoke") -> dict:
    config = load_config("figure2", phase, [0])
    root = Path(config["root"])
    report = environment_manifest(root)
    report["protocol"] = config["protocol"]
    report["tasks"] = {}
    for task in config["tasks"]:
        dataset = load_cached_d4rl_dataset(task)
        split = _figure2_split(dataset, config)
        report["tasks"][task] = {
            "dataset_size": dataset.size,
            "observation_shape": list(dataset.observations.shape),
            "action_shape": list(dataset.actions.shape),
            "source_path": str(dataset.source_path),
            "sha256": sha256(dataset.source_path) if dataset.source_path else None,
            "split": split.to_dict(),
        }
    output = root / "results" / "manifests" / "environment-check.json"
    write_manifest(output, report)
    return report


def plot_experiment(experiment: str, phase: str = "full") -> tuple[Path, Path]:
    config = load_config(experiment, phase)
    root = Path(config["root"])
    if experiment == "figure2":
        return plot_figure2(root, result_dir=_figure2_result_dir(root, config))
    return plot_figure4(root, result_dir=_figure4_result_dir(root, config))
