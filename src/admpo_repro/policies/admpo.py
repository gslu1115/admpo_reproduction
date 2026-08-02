from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from admpo_repro.data import D4RLDataset
from admpo_repro.runtime import capture_rng_state, restore_rng_state


def _vendor_path(root: Path) -> Path:
    path = root / "vendor" / "ADMPO"
    if not (path / "agent" / "admpo.py").exists():
        raise FileNotFoundError(
            "vendor/ADMPO is missing; run git submodule update --init --recursive"
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def _raw_dataset(dataset: D4RLDataset) -> dict[str, np.ndarray]:
    return {
        "observations": dataset.observations,
        "actions": dataset.actions,
        "rewards": dataset.rewards.reshape(-1),
        "next_observations": dataset.next_observations,
        "terminals": dataset.terminals.reshape(-1),
        "timeouts": dataset.timeouts.reshape(-1),
    }


def _build_agent(dataset: D4RLDataset, env: object, config: dict):
    root = Path(config["root"])
    _vendor_path(root)
    from agent.admpo import ADMPOAgent
    from components.static_fns import STATICFUNC

    cfg = config["policy"]
    static_fn = STATICFUNC[dataset.task.split("-")[0].lower()]
    agent = ADMPOAgent(
        obs_shape=env.observation_space.shape,
        hidden_dims=cfg["hidden_dims"],
        action_dim=dataset.action_dim,
        action_space=env.action_space,
        static_fn=static_fn,
        max_arm_step=int(config["model"]["max_backtrack"]),
        arm_hidden_dim=int(config["model"]["hidden_dim"]),
        actor_freq=1,
        actor_lr=float(cfg["actor_lr"]),
        critic_lr=float(cfg["critic_lr"]),
        model_lr=float(cfg["model_lr"]),
        tau=float(cfg["tau"]),
        gamma=float(cfg["gamma"]),
        alpha=float(cfg["alpha"][dataset.task]),
        auto_alpha=False,
        alpha_lr=1e-4,
        target_entropy=None,
        penalty_coef=float(cfg["penalty"]),
        deterministic_backup=False,
        q_clip=None,
        device=config["device"],
    )
    return agent


def _build_buffers(dataset: D4RLDataset, config: dict):
    _vendor_path(Path(config["root"]))
    from buffer.buffer import ReplayBuffer
    from buffer.buffer4seqsamp import ReplayBufferForSeqSampling

    real = ReplayBufferForSeqSampling(max(1_000_000, dataset.size), (dataset.obs_dim,), dataset.action_dim)
    real.load_dataset(_raw_dataset(dataset), dataset.max_episode_steps)
    cfg = config["policy"]
    capacity = int(cfg["rollout_batch_size"] * cfg["rollout_horizon"] * cfg["model_retain_epochs"])
    model = ReplayBuffer(capacity, (dataset.obs_dim,), dataset.action_dim)
    return real, model


def _convert_adm_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace(".norm.", ".layer_norm."): value for key, value in state.items()}


def _agent_state(agent) -> dict:
    state = {
        "actor": agent.actor.state_dict(),
        "critic1": agent.critic1.state_dict(),
        "critic2": agent.critic2.state_dict(),
        "critic1_target": agent.critic1_trgt.state_dict(),
        "critic2_target": agent.critic2_trgt.state_dict(),
        "dynamics": agent.dynamics.state_dict(),
        "actor_optimizer": agent.actor_optim.state_dict(),
        "critic1_optimizer": agent.critic1_optim.state_dict(),
        "critic2_optimizer": agent.critic2_optim.state_dict(),
        "dynamics_optimizer": agent.dyna_optim.state_dict(),
        "alpha": agent._alpha,
        "critic_count": agent.critic_cnt,
    }
    if getattr(agent, "_auto_alpha", False):
        state.update(
            {
                "log_alpha": agent._log_alpha,
                "alpha_optimizer": agent._alpha_optim.state_dict(),
            }
        )
    return state


def _restore_agent(agent, state: dict) -> None:
    agent.actor.load_state_dict(state["actor"])
    agent.critic1.load_state_dict(state["critic1"])
    agent.critic2.load_state_dict(state["critic2"])
    agent.critic1_trgt.load_state_dict(state["critic1_target"])
    agent.critic2_trgt.load_state_dict(state["critic2_target"])
    agent.dynamics.load_state_dict(state["dynamics"])
    agent.actor_optim.load_state_dict(state["actor_optimizer"])
    agent.critic1_optim.load_state_dict(state["critic1_optimizer"])
    agent.critic2_optim.load_state_dict(state["critic2_optimizer"])
    agent.dyna_optim.load_state_dict(state["dynamics_optimizer"])
    agent._alpha = state["alpha"]
    agent.critic_cnt = state["critic_count"]
    if "log_alpha" in state:
        agent._log_alpha = state["log_alpha"]
        agent._alpha_optim.load_state_dict(state["alpha_optimizer"])


def _buffer_state(buffer) -> dict:
    return {
        "size": buffer.size,
        "cnt": buffer.cnt,
        "memory": {key: value[: buffer.size].copy() for key, value in buffer.memory.items()},
    }


def _restore_buffer(buffer, state: dict) -> None:
    size = int(state["size"])
    for key, value in state["memory"].items():
        buffer.memory[key][:size] = value
    buffer.size = size
    buffer.cnt = int(state["cnt"])


def _save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def _rollout_in_chunks(agent, real_buffer, model_buffer, batch_size: int, chunk: int, horizon: int, history: int) -> None:
    remaining = batch_size
    while remaining:
        current = min(remaining, chunk)
        initial = real_buffer.sample_nstep(current, history - 1)
        transitions = agent.rollout(initial, horizon)
        if len(transitions["s"]):
            model_buffer.store_batch(**transitions)
        remaining -= current


def _evaluate(agent, env: object, episodes: int) -> tuple[float, float]:
    rewards = []
    for _ in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        total = 0.0
        while not done:
            action, _ = agent.act(obs, deterministic=True)
            result = env.step(action)
            if len(result) == 4:
                obs, reward, done, _ = result
            else:
                obs, reward, terminated, truncated, _ = result
                done = terminated or truncated
            total += float(reward)
        rewards.append(total)
    mean_reward = float(np.mean(rewards))
    score = float(env.get_normalized_score(mean_reward) * 100) if hasattr(env, "get_normalized_score") else float("nan")
    return mean_reward, score


def train_admpo(
    dataset: D4RLDataset,
    env: object,
    config: dict,
    seed: int,
    adm_checkpoint: Path,
    output_dir: Path,
    resume: bool,
) -> Path:
    cfg = config["policy"]
    device = config["device"]
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = _build_agent(dataset, env, config)
    real_buffer, model_buffer = _build_buffers(dataset, config)
    adm_state = torch.load(adm_checkpoint, map_location=device)["model"]
    agent.dynamics.load_state_dict(_convert_adm_state(adm_state), strict=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        agent.actor_optim, int(cfg.get("scheduler_epochs", cfg["epochs"]))
    )
    records: list[dict] = []
    start_epoch = 0
    checkpoints = sorted(output_dir.glob("epoch-*.pt"))
    if resume and checkpoints:
        state = torch.load(checkpoints[-1], map_location=device)
        _restore_agent(agent, state["agent"])
        _restore_buffer(model_buffer, state["model_buffer"])
        scheduler.load_state_dict(state["scheduler"])
        records = state["records"]
        start_epoch = int(state["epoch"]) + 1
        restore_rng_state(state["rng"])
    history = int(config["model"]["max_backtrack"])
    num_steps = start_epoch * int(cfg["steps_per_epoch"])
    for epoch in range(start_epoch, int(cfg["epochs"])):
        losses = []
        for _ in range(int(cfg["steps_per_epoch"])):
            if num_steps % int(cfg["rollout_frequency"]) == 0:
                _rollout_in_chunks(
                    agent,
                    real_buffer,
                    model_buffer,
                    int(cfg["rollout_batch_size"]),
                    int(cfg["rollout_chunk_size"]),
                    int(cfg["rollout_horizon"]),
                    history,
                )
            real_count = int(int(cfg["batch_size"]) * float(cfg["real_ratio"]))
            fake_count = int(cfg["batch_size"]) - real_count
            real_batch = real_buffer.sample(real_count)
            fake_batch = model_buffer.sample(fake_count)
            batch = {
                key: np.concatenate((real_batch[key], fake_batch[key]), axis=0)
                for key in real_batch
                if key != "timeout"
            }
            info = agent.learn(**batch)
            losses.append(info)
            num_steps += 1
        scheduler.step()
        record = {
            "epoch": epoch,
            "actor_loss": float(np.nanmean([x["loss"]["actor"] for x in losses if x["loss"]["actor"] is not None])),
            "critic1_loss": float(np.mean([x["loss"]["critic1"] for x in losses])),
            "critic2_loss": float(np.mean([x["loss"]["critic2"] for x in losses])),
            "alpha": float(losses[-1]["alpha"]),
        }
        if epoch % int(cfg["eval_interval"]) == 0 or epoch + 1 == int(cfg["epochs"]):
            reward, score = _evaluate(agent, env, int(cfg["eval_episodes"]))
            record.update({"eval_reward": reward, "eval_score": score})
        records.append(record)
        if (epoch + 1) % int(cfg["checkpoint_interval"]) == 0 or epoch + 1 == int(cfg["epochs"]):
            path = output_dir / f"epoch-{epoch + 1:06d}.pt"
            _save(
                {
                    "epoch": epoch,
                    "agent": _agent_state(agent),
                    "model_buffer": _buffer_state(model_buffer),
                    "scheduler": scheduler.state_dict(),
                    "records": records,
                    "rng": capture_rng_state(),
                    "task": dataset.task,
                    "seed": seed,
                },
                path,
            )
            old = sorted(output_dir.glob("epoch-*.pt"))[:-2]
            for old_path in old:
                old_path.unlink()
            (output_dir / "training.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    final = output_dir / "final.pt"
    _save(
        {
            "actor": agent.actor.state_dict(),
            "agent": _agent_state(agent),
            "records": records,
            "task": dataset.task,
            "seed": seed,
        },
        final,
    )
    return final


def load_learned_policy(
    dataset: D4RLDataset,
    env: object,
    config: dict,
    checkpoint: Path,
) -> Callable[[np.ndarray], np.ndarray]:
    agent = _build_agent(dataset, env, config)
    payload = torch.load(checkpoint, map_location=config["device"])
    agent.actor.load_state_dict(payload["actor"])
    agent.eval()

    def act(observations: np.ndarray) -> np.ndarray:
        action, _ = agent.act(observations, deterministic=True)
        return action

    return act
