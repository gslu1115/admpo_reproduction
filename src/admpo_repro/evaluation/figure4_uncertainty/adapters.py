from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PredictionBatch:
    """Raw-state Gaussian predictions from five epistemic sources."""

    means: np.ndarray
    logvars: np.ndarray
    source_ids: np.ndarray

    def validate(self) -> None:
        if self.means.ndim != 3 or self.logvars.shape != self.means.shape:
            raise RuntimeError("predictor means/logvars must have shape [source,batch,state]")
        if self.means.shape[0] != 5 or self.source_ids.shape != (5,):
            raise RuntimeError("the main protocol requires exactly five predictor sources")
        if not np.isfinite(self.means).all() or not np.isfinite(self.logvars).all():
            raise RuntimeError("non-finite model prediction entered the evaluation")

    def paper_eq4_uncertainty_l1(self) -> np.ndarray:
        """Literal paper Eq. 4: L1 norm of epistemic plus aleatoric variance."""
        self.validate()
        means = self.means.astype(np.float64)
        logvars = self.logvars.astype(np.float64)
        epistemic_variance = means.var(axis=0, ddof=0)
        aleatoric_variance = np.exp(logvars).mean(axis=0)
        total_variance = epistemic_variance + aleatoric_variance
        if not np.isfinite(total_variance).all() or np.any(total_variance < 0.0):
            raise RuntimeError("invalid total predictive variance in paper Eq. 4")
        return total_variance.sum(axis=-1).astype(np.float32)

    def mixture_expected_squared_error(self, true_next_states: np.ndarray) -> np.ndarray:
        """Expected squared raw-state error of the five-component Gaussian mixture.

        The source expectation matches Eq. 4: average over the five predictors,
        then sum the per-state-dimension squared quantities. For each source,
        ``E[||S' - s'||_2^2]`` equals squared mean error plus aleatoric variance.
        """
        self.validate()
        truth = np.asarray(true_next_states, dtype=np.float64)
        expected_shape = self.means.shape[1:]
        if truth.shape != expected_shape:
            raise ValueError(
                "true next states must have shape "
                f"{expected_shape}, got {truth.shape}"
            )
        if not np.isfinite(truth).all():
            raise RuntimeError("non-finite true next state entered model-error evaluation")
        means = self.means.astype(np.float64)
        variances = np.exp(self.logvars.astype(np.float64))
        per_source_squared_error = np.square(means - truth[None, ...]) + variances
        expected_squared_error = per_source_squared_error.mean(axis=0).sum(axis=-1)
        if not np.isfinite(expected_squared_error).all() or np.any(
            expected_squared_error < 0.0
        ):
            raise RuntimeError("invalid Gaussian-mixture expected squared error")
        return expected_squared_error.astype(np.float32)


class ADMEvaluatorAdapter:
    """Evaluate the five official any-step backtracking predictors."""

    kind = "adm"

    def __init__(self, model: torch.nn.Module, obs_dim: int, device: str, chunk_size: int):
        self.model = model
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        self.model.eval()

    @torch.no_grad()
    def predict_all_sources(self, states: np.ndarray, actions: np.ndarray) -> PredictionBatch:
        if states.ndim != 3 or actions.ndim != 3 or states.shape[0] != actions.shape[0]:
            raise ValueError("states/actions must be aligned batched histories")
        if states.shape[1] != actions.shape[1] or states.shape[1] != 5:
            raise RuntimeError("ADM evaluation requires five states and five actions")
        batch = states.shape[0]
        means = np.empty((5, batch, self.obs_dim), dtype=np.float32)
        logvars = np.empty_like(means)
        for k in range(1, 6):
            for start in range(0, batch, self.chunk_size):
                stop = min(start + self.chunk_size, batch)
                anchor = torch.as_tensor(states[start:stop, -k], device=self.device)
                action_sequence = torch.as_tensor(actions[start:stop, -k:], device=self.device)
                delta_mean, full_logvar = self.model.forward(anchor, action_sequence)
                means[k - 1, start:stop] = (
                    anchor + delta_mean[:, : self.obs_dim]
                ).cpu().numpy()
                logvars[k - 1, start:stop] = full_logvar[:, : self.obs_dim].cpu().numpy()
        result = PredictionBatch(means, logvars, np.arange(1, 6, dtype=np.int64))
        result.validate()
        return result


class EnsembleEvaluatorAdapter:
    """Evaluate the five fixed Figure 2 elite ensemble members."""

    kind = "ensemble"

    def __init__(self, model: torch.nn.Module, obs_dim: int, device: str, chunk_size: int):
        self.model = model
        self.obs_dim = int(obs_dim)
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        self.model.eval()
        elite = self.model.model.elite_indices.detach().cpu().numpy().astype(np.int64)
        if elite.shape != (5,) or np.unique(elite).size != 5:
            raise RuntimeError(f"expected five fixed unique elites, got {elite.tolist()}")
        if int(self.model.model.size) != 7:
            raise RuntimeError(f"expected seven ensemble members, got {self.model.model.size}")
        self.elite_ids = elite

    @torch.no_grad()
    def predict_all_sources(self, states: np.ndarray, actions: np.ndarray) -> PredictionBatch:
        if states.ndim != 3 or actions.ndim != 3 or states.shape[0] != actions.shape[0]:
            raise ValueError("states/actions must be aligned batched histories")
        batch = states.shape[0]
        means = np.empty((5, batch, self.obs_dim), dtype=np.float32)
        logvars = np.empty_like(means)
        elite_tensor = torch.as_tensor(self.elite_ids, dtype=torch.long, device=self.device)
        for start in range(0, batch, self.chunk_size):
            stop = min(start + self.chunk_size, batch)
            current = torch.as_tensor(states[start:stop, -1], device=self.device)
            action = torch.as_tensor(actions[start:stop, -1], device=self.device)
            full_mean, full_logvar = self.model.forward(current, action)
            elite_mean = full_mean.index_select(0, elite_tensor)[..., : self.obs_dim]
            elite_logvar = full_logvar.index_select(0, elite_tensor)[..., : self.obs_dim]
            means[:, start:stop] = (current[None] + elite_mean).cpu().numpy()
            logvars[:, start:stop] = elite_logvar.cpu().numpy()
        result = PredictionBatch(means, logvars, self.elite_ids.copy())
        result.validate()
        return result
