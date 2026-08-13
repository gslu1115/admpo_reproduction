from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapters import PredictionBatch
from .policies import RolloutPolicy
from .protocol import InitialWindows


@dataclass
class EndpointBatch:
    dynamics_type: str
    model_seed: int
    policy: str
    horizons: np.ndarray
    source_ids: np.ndarray
    initial_window_ids: np.ndarray
    current_transition_ids: np.ndarray
    trajectory_ids: np.ndarray
    rollout_source_schedule: np.ndarray
    sampled_next_state_schedule: np.ndarray
    valid_endpoint: np.ndarray
    terminated: np.ndarray
    terminated_before_endpoint: np.ndarray
    terminal_step: np.ndarray
    oracle_done: np.ndarray
    chosen_source: np.ndarray
    uncertainty: np.ndarray
    model_error: np.ndarray
    endpoint_state: np.ndarray
    action: np.ndarray
    all_predictor_means: np.ndarray
    all_predictor_logvars: np.ndarray
    selected_mean_next_state: np.ndarray
    selected_logvar: np.ndarray
    gaussian_epsilon: np.ndarray
    sampled_next_state: np.ndarray
    true_next_state: np.ndarray

    def validate(self) -> None:
        expected_schedule = (
            int(self.horizons[-1]) + 1,
            self.initial_window_ids.size,
        )
        if self.rollout_source_schedule.shape != expected_schedule:
            raise RuntimeError("rollout source schedule is not aligned to steps/windows")
        obs_dim = self.endpoint_state.shape[-1]
        expected_vector_schedule = (*expected_schedule, obs_dim)
        if self.sampled_next_state_schedule.shape != expected_vector_schedule:
            raise RuntimeError("sampled-next-state schedule is not aligned to steps/windows")
        used_sources = self.rollout_source_schedule[self.rollout_source_schedule >= 0]
        if not np.isin(used_sources, self.source_ids).all():
            raise RuntimeError("rollout source schedule contains an unknown source")
        expected = (self.horizons.size, self.initial_window_ids.size)
        for name in (
            "valid_endpoint",
            "terminated",
            "terminated_before_endpoint",
            "oracle_done",
            "chosen_source",
            "uncertainty",
            "model_error",
        ):
            if getattr(self, name).shape[:2] != expected:
                raise RuntimeError(f"endpoint field {name} is not aligned to horizons/windows")
        if not np.array_equal(self.horizons, np.asarray(sorted(set(self.horizons.tolist())))):
            raise RuntimeError("endpoint horizons must be unique and sorted")
        valid = self.valid_endpoint
        if np.any(self.terminated_before_endpoint & valid):
            raise RuntimeError("a trajectory terminated before an endpoint but was marked valid")
        if not np.isfinite(self.uncertainty[valid]).all():
            raise RuntimeError("non-finite uncertainty entered valid endpoints")
        if not np.isfinite(self.model_error[valid]).all():
            raise RuntimeError("non-finite model error entered valid endpoints")
        active_steps = self.rollout_source_schedule >= 0
        if not np.isfinite(self.sampled_next_state_schedule[active_steps]).all():
            raise RuntimeError("non-finite sampled rollout state was recorded")
        for values in (
            self.endpoint_state,
            self.action,
            self.all_predictor_means,
            self.all_predictor_logvars,
            self.selected_mean_next_state,
            self.selected_logvar,
            self.gaussian_epsilon,
            self.sampled_next_state,
            self.true_next_state,
        ):
            if not np.isfinite(values[valid]).all():
                raise RuntimeError("non-finite vector entered valid endpoints")


def _oracle_batch_step(oracle, states: np.ndarray, actions: np.ndarray, chunk_size: int):
    next_states = np.empty_like(states, dtype=np.float32)
    dones = np.zeros(states.shape[0], dtype=bool)
    for start in range(0, states.shape[0], chunk_size):
        stop = min(start + chunk_size, states.shape[0])
        next_obs, _, done = oracle.batch_step(states[start:stop], actions[start:stop])
        next_states[start:stop] = next_obs
        dones[start:stop] = done[:, 0]
    if not np.isfinite(next_states).all():
        raise RuntimeError("MuJoCo oracle returned a non-finite one-step next state")
    return next_states, dones


def _allocate(horizons: np.ndarray, windows: InitialWindows, action_dim: int):
    h_count, count = horizons.size, windows.count
    obs_dim = windows.history_states.shape[-1]
    scalar_float = lambda: np.full((h_count, count), np.nan, dtype=np.float32)
    vector = lambda width: np.full((h_count, count, width), np.nan, dtype=np.float32)
    return {
        "valid_endpoint": np.zeros((h_count, count), dtype=bool),
        "terminated": np.zeros((h_count, count), dtype=bool),
        "terminated_before_endpoint": np.zeros((h_count, count), dtype=bool),
        "oracle_done": np.zeros((h_count, count), dtype=bool),
        "chosen_source": np.full((h_count, count), -1, dtype=np.int64),
        "uncertainty": scalar_float(),
        "model_error": scalar_float(),
        "endpoint_state": vector(obs_dim),
        "action": vector(action_dim),
        "all_predictor_means": np.full(
            (h_count, count, 5, obs_dim), np.nan, dtype=np.float32
        ),
        "all_predictor_logvars": np.full(
            (h_count, count, 5, obs_dim), np.nan, dtype=np.float32
        ),
        "selected_mean_next_state": vector(obs_dim),
        "selected_logvar": vector(obs_dim),
        "gaussian_epsilon": vector(obs_dim),
        "sampled_next_state": vector(obs_dim),
        "true_next_state": vector(obs_dim),
    }


def evaluate_native_rollout(
    *,
    adapter,
    policy: RolloutPolicy,
    oracle,
    termination_fn,
    initial_windows: InitialWindows,
    horizons: list[int] | tuple[int, ...],
    source_rng: np.random.Generator,
    gaussian_rng: np.random.Generator,
    oracle_chunk: int,
    model_seed: int,
) -> EndpointBatch:
    """Roll out once and probe only the explicitly requested states ``s_h``.

    The initial history ends at s_0. Transitions t=0,...,h-1 move the selected
    Gaussian model to s_h. At s_h the same policy supplies a_h and the endpoint
    probe compares the five-component predictive mixture with MuJoCo
    F(s_h,a_h). The selected Gaussian sample from that probe is also the
    continuation state for later requested steps. A terminal probe transition
    remains a valid endpoint and stops all later endpoints.
    """
    endpoint_horizons = np.asarray([int(value) for value in horizons], dtype=np.int64)
    if not np.array_equal(endpoint_horizons, np.asarray([5], dtype=np.int64)):
        raise ValueError("the Figure 4 main protocol evaluates only h=5")
    if initial_windows.history_states.shape[1] != 5:
        raise RuntimeError("the ADM main protocol requires five initial states")
    if initial_windows.history_actions.shape[1] != 4:
        raise RuntimeError("the ADM main protocol requires four preceding actions")

    count = initial_windows.count
    action_dim = initial_windows.history_actions.shape[-1]
    obs_dim = initial_windows.history_states.shape[-1]
    arrays = _allocate(endpoint_horizons, initial_windows, action_dim)
    rollout_steps = int(endpoint_horizons[-1]) + 1
    if adapter.kind == "adm":
        shared_schedule = source_rng.integers(0, 5, size=rollout_steps).astype(np.int64)
        source_position_schedule = np.repeat(shared_schedule[:, None], count, axis=1)
    elif adapter.kind == "ensemble":
        source_position_schedule = source_rng.integers(
            0, 5, size=(rollout_steps, count)
        ).astype(np.int64)
    else:
        raise ValueError(f"unknown dynamics adapter kind: {adapter.kind}")
    gaussian_epsilon_schedule = gaussian_rng.normal(
        size=(rollout_steps, count, initial_windows.history_states.shape[-1])
    ).astype(np.float32)
    rollout_source_schedule = np.full((rollout_steps, count), -1, dtype=np.int64)
    sampled_next_state_schedule = np.full(
        (rollout_steps, count, obs_dim), np.nan, dtype=np.float32
    )
    horizon_index = {int(value): index for index, value in enumerate(endpoint_horizons)}
    states = initial_windows.history_states.copy()
    previous_actions = initial_windows.history_actions.copy()
    active = np.ones(count, dtype=bool)
    terminal_step = np.full(count, -1, dtype=np.int64)

    for rollout_step in range(int(endpoint_horizons[-1]) + 1):
        active_ids = np.flatnonzero(active)
        if active_ids.size == 0:
            break
        current_state = states[active_ids, -1].copy()
        current_action = policy.act(current_state, active_ids, rollout_step)
        if current_action.shape != (active_ids.size, action_dim):
            raise RuntimeError("policy returned an incorrectly shaped action batch")
        action_history = np.concatenate(
            (previous_actions[active_ids], current_action[:, None, :]), axis=1
        )
        prediction: PredictionBatch = adapter.predict_all_sources(
            states[active_ids], action_history
        )
        uncertainty = prediction.paper_eq4_uncertainty_l1()

        selected_positions = source_position_schedule[rollout_step, active_ids]
        row = np.arange(active_ids.size, dtype=np.int64)
        selected_mean = prediction.means[selected_positions, row]
        selected_logvar = prediction.logvars[selected_positions, row]
        selected_source = prediction.source_ids[selected_positions]
        rollout_source_schedule[rollout_step, active_ids] = selected_source
        epsilon = gaussian_epsilon_schedule[rollout_step, active_ids]
        sampled_next = (
            selected_mean + np.exp(0.5 * selected_logvar) * epsilon
        )
        if not np.isfinite(sampled_next).all():
            raise RuntimeError("Gaussian rollout propagation produced a non-finite state")
        sampled_next_state_schedule[rollout_step, active_ids] = sampled_next

        terminated = np.asarray(
            termination_fn(current_state, current_action, sampled_next), dtype=bool
        ).reshape(-1)
        if terminated.shape != (active_ids.size,):
            raise RuntimeError("vendor termination function returned an invalid shape")

        if rollout_step in horizon_index:
            h_index = horizon_index[rollout_step]
            true_next, oracle_done = _oracle_batch_step(
                oracle, current_state, current_action, int(oracle_chunk)
            )
            model_error = prediction.mixture_expected_squared_error(true_next)
            arrays["valid_endpoint"][h_index, active_ids] = True
            arrays["terminated"][h_index, active_ids] = terminated
            arrays["oracle_done"][h_index, active_ids] = oracle_done
            arrays["chosen_source"][h_index, active_ids] = selected_source
            arrays["uncertainty"][h_index, active_ids] = uncertainty
            arrays["model_error"][h_index, active_ids] = model_error
            arrays["endpoint_state"][h_index, active_ids] = current_state
            arrays["action"][h_index, active_ids] = current_action
            arrays["all_predictor_means"][h_index, active_ids] = np.moveaxis(
                prediction.means, 0, 1
            )
            arrays["all_predictor_logvars"][h_index, active_ids] = np.moveaxis(
                prediction.logvars, 0, 1
            )
            arrays["selected_mean_next_state"][h_index, active_ids] = selected_mean
            arrays["selected_logvar"][h_index, active_ids] = selected_logvar
            arrays["gaussian_epsilon"][h_index, active_ids] = epsilon
            arrays["sampled_next_state"][h_index, active_ids] = sampled_next
            arrays["true_next_state"][h_index, active_ids] = true_next

        newly_terminated = active_ids[terminated]
        terminal_step[newly_terminated] = rollout_step + 1
        active[newly_terminated] = False
        if rollout_step == int(endpoint_horizons[-1]):
            break
        survivors = active_ids[~terminated]
        local_survivors = np.flatnonzero(~terminated)
        if survivors.size:
            states[survivors] = np.concatenate(
                (states[survivors, 1:], sampled_next[local_survivors, None, :]), axis=1
            )
            previous_actions[survivors] = action_history[local_survivors, -4:]

    for h_index, horizon in enumerate(endpoint_horizons):
        before = (terminal_step >= 0) & (terminal_step <= int(horizon))
        arrays["terminated_before_endpoint"][h_index] = before & ~arrays["valid_endpoint"][h_index]

    batch = EndpointBatch(
        dynamics_type=adapter.kind,
        model_seed=int(model_seed),
        policy=policy.name,
        horizons=endpoint_horizons,
        source_ids=(
            np.arange(1, 6, dtype=np.int64)
            if adapter.kind == "adm"
            else np.asarray(adapter.elite_ids, dtype=np.int64).copy()
        ),
        initial_window_ids=initial_windows.initial_window_ids.copy(),
        current_transition_ids=initial_windows.current_transition_ids.copy(),
        trajectory_ids=initial_windows.trajectory_ids.copy(),
        rollout_source_schedule=rollout_source_schedule,
        sampled_next_state_schedule=sampled_next_state_schedule,
        terminal_step=terminal_step,
        **arrays,
    )
    batch.validate()
    return batch
