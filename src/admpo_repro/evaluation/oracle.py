from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from admpo_repro.data import D4RLDataset


@dataclass
class OracleAudit:
    samples: int
    excluded_clipped: int
    median_mse: float
    p99_mse: float
    maximum_mse: float

    @property
    def passed(self) -> bool:
        return self.median_mse <= 1e-8 and self.p99_mse <= 1e-5


@dataclass
class OracleRMSEAudit:
    """One-step state reconstruction audit in RMSE units."""

    samples: int
    excluded_clipped: int
    median_rmse: float
    p99_rmse: float
    maximum_rmse: float

    def passed(self, tolerance: float) -> bool:
        return self.median_rmse <= tolerance and self.p99_rmse <= tolerance


class MujocoOracle:
    """MuJoCo oracle: one-step queries and continuous control trajectories.

    ``step`` / ``batch_step`` reconstruct the simulator state from a D4RL
    observation and perform a single transition (used for one-step
    diagnostics and audits).  ``state_from_observation`` + ``step_states``
    instead carry the *full* MuJoCo state (qpos/qvel/act/udd_state) between
    transitions, so a multi-step real trajectory is exactly
    ``s_{i+1} = T(s_i, a_i)`` from a common initial observation, and the
    environment is never re-seeded or re-observed between steps.
    """

    def __init__(self, env: object) -> None:
        self.env = env
        self.base = getattr(env, "unwrapped", env)

    def set_state_from_observation(self, observation: np.ndarray) -> None:
        observation = np.asarray(observation, dtype=np.float64).reshape(-1)
        nq = int(self.base.model.nq)
        nv = int(self.base.model.nv)
        if observation.size == nq + nv - 1:
            observation = np.concatenate((np.zeros(1, dtype=observation.dtype), observation))
        if observation.size != nq + nv:
            raise ValueError(
                f"observation has {observation.size} values, expected {nq + nv} or {nq + nv - 1}"
            )
        self.env.reset()
        if hasattr(self.env, "_elapsed_steps"):
            self.env._elapsed_steps = 0
        self.base.set_state(observation[:nq], observation[nq:])

    def step(self, observation: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        self.set_state_from_observation(observation)
        result = self.env.step(np.asarray(action, dtype=np.float32))
        if len(result) == 4:
            next_obs, reward, done, _ = result
        else:
            next_obs, reward, terminated, truncated, _ = result
            done = terminated or truncated
        return np.asarray(next_obs, dtype=np.float32), float(reward), bool(done)

    def batch_step(
        self, observations: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_obs = np.empty_like(observations, dtype=np.float32)
        rewards = np.empty((len(observations), 1), dtype=np.float32)
        dones = np.zeros((len(observations), 1), dtype=bool)
        for i, (obs, action) in enumerate(zip(observations, actions)):
            try:
                next_obs[i], rewards[i, 0], dones[i, 0] = self.step(obs, action)
            except Exception:
                next_obs[i] = np.nan
                rewards[i, 0] = np.nan
                dones[i, 0] = True
        return next_obs, rewards, dones

    def state_from_observation(self, observation: np.ndarray):
        """Decode a D4RL observation into a full MuJoCo SimState.

        ``sim.get_state()`` is used as a template so actuator and user-data
        fields keep their reset values; only qpos/qvel are replaced from the
        observation.  This state is the common start ``s_0`` of a continuous
        control trajectory.
        """
        observation = np.asarray(observation, dtype=np.float64).reshape(-1)
        nq = int(self.base.model.nq)
        nv = int(self.base.model.nv)
        if observation.size == nq + nv - 1:
            observation = np.concatenate((np.zeros(1, dtype=observation.dtype), observation))
        if observation.size != nq + nv:
            raise ValueError(
                f"observation has {observation.size} values, expected {nq + nv} or {nq + nv - 1}"
            )
        state = self.base.sim.get_state()
        state.qpos[:] = observation[:nq]
        state.qvel[:] = observation[nq:]
        return state

    def step_states(
        self, states: list, actions: np.ndarray
    ) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        """Step a batch of full simulator states continuously.

        Each ``state`` is the full MuJoCo state carried from the previous
        transition, so the environment is restored to exactly the state a
        continuous run would have at that point and then stepped with the
        given action.  The returned next states are again full simulator
        states (``sim.get_state()``), never observations, so clipped D4RL
        velocity observations cannot corrupt the control trajectory.
        """

        actions = np.asarray(actions, dtype=np.float32)
        batch = len(actions)
        obs_dim = int(self.base.observation_space.shape[0])
        # gym requires env.reset() before env.step(); the reset itself is
        # immediately overwritten by the carried full simulator state below.
        self.env.reset()
        next_observations = np.empty((batch, obs_dim), dtype=np.float32)
        rewards = np.empty((batch, 1), dtype=np.float32)
        dones = np.zeros((batch, 1), dtype=bool)
        next_states: list = []
        for i, (state, action) in enumerate(zip(states, actions)):
            try:
                self.base.sim.set_state(state)
                result = self.env.step(action)
            except Exception:
                next_observations[i] = np.nan
                rewards[i, 0] = np.nan
                dones[i, 0] = True
                next_states.append(None)
                continue
            if len(result) == 4:
                next_obs, reward, done, _ = result
            else:
                next_obs, reward, terminated, truncated, _ = result
                done = terminated or truncated
            next_observations[i] = np.asarray(next_obs, dtype=np.float32)
            rewards[i, 0] = float(reward)
            dones[i, 0] = bool(done)
            next_states.append(self.base.sim.get_state())
        return next_states, next_observations, rewards, dones


def termination(task: str, next_obs: np.ndarray) -> np.ndarray:
    finite = np.isfinite(next_obs).all(axis=-1)
    bounded = (np.abs(next_obs[:, 1:]) < 100).all(axis=-1)
    if task.startswith("hopper"):
        healthy = finite & bounded & (next_obs[:, 0] > 0.7) & (np.abs(next_obs[:, 1]) < 0.2)
    elif task.startswith("walker2d"):
        healthy = finite & (next_obs[:, 0] > 0.8) & (next_obs[:, 0] < 2.0)
        healthy &= (next_obs[:, 1] > -1.0) & (next_obs[:, 1] < 1.0)
    else:
        healthy = finite
    return ~healthy


def audit_oracle(
    dataset: D4RLDataset, env: object, samples: int = 1000, seed: int = 0
) -> OracleAudit:
    rng = np.random.default_rng(seed)
    pool = np.where(~dataset.boundaries)[0]
    # Gym's Hopper/Walker observations clip qvel to [-10, 10]. A clipped
    # observation no longer contains the full simulator state and therefore
    # cannot be used to audit exact state reconstruction.
    qvel_start = int(getattr(env, "unwrapped", env).model.nq) - 1
    observable = (np.abs(dataset.observations[pool, qvel_start:]) < 9.999).all(axis=-1)
    excluded_clipped = int((~observable).sum())
    pool = pool[observable]
    indexes = rng.choice(pool, size=min(samples, pool.size), replace=False)
    oracle = MujocoOracle(env)
    predicted, _, _ = oracle.batch_step(dataset.observations[indexes], dataset.actions[indexes])
    errors = np.mean(np.square(predicted - dataset.next_observations[indexes]), axis=-1)
    errors = errors[np.isfinite(errors)]
    return OracleAudit(
        samples=int(errors.size),
        excluded_clipped=excluded_clipped,
        median_mse=float(np.median(errors)),
        p99_mse=float(np.quantile(errors, 0.99)),
        maximum_mse=float(np.max(errors)),
    )


def audit_oracle_rmse(
    dataset: D4RLDataset, env: object, points: int = 5000, seed: int = 0
) -> OracleRMSEAudit:
    """One-step reconstruction audit in per-state-dimension RMSE units.

    Reconstructs the simulator state from held-out D4RL observations, steps
    the oracle with the recorded action, and compares with the recorded next
    observation.  Clipped velocity observations (qvel at +/-10) are excluded
    because they no longer contain the full simulator state.
    """

    rng = np.random.default_rng(seed)
    pool = np.where(~dataset.boundaries)[0]
    qvel_start = int(getattr(env, "unwrapped", env).model.nq) - 1
    observable = (np.abs(dataset.observations[pool, qvel_start:]) < 9.999).all(axis=-1)
    excluded_clipped = int((~observable).sum())
    pool = pool[observable]
    indexes = rng.choice(pool, size=min(points, pool.size), replace=False)
    oracle = MujocoOracle(env)
    predicted, _, _ = oracle.batch_step(
        dataset.observations[indexes], dataset.actions[indexes]
    )
    mse = np.mean(
        np.square(predicted - dataset.next_observations[indexes]), axis=-1
    )
    rmse = np.sqrt(mse)
    rmse = rmse[np.isfinite(rmse)]
    return OracleRMSEAudit(
        samples=int(rmse.size),
        excluded_clipped=excluded_clipped,
        median_rmse=float(np.median(rmse)),
        p99_rmse=float(np.quantile(rmse, 0.99)),
        maximum_rmse=float(np.max(rmse)),
    )
