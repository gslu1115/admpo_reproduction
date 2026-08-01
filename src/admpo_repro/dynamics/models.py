from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def soft_clamp(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    x = upper - F.softplus(upper - x)
    return lower + F.softplus(x - lower)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = Swish()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(self.activation(self.linear(x))))


class SequenceBackbone(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        output_dim: int,
        hidden_dim: int = 200,
        rnn_layers: int = 3,
        residual_blocks: int = 4,
        dropout: float = 0.1,
        repeat_initial_obs: bool = True,
    ) -> None:
        super().__init__()
        self.repeat_initial_obs = repeat_initial_obs
        self.rnn_layer = nn.GRU(
            input_size=obs_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )
        layers: list[nn.Module] = [ResidualBlock(hidden_dim, dropout) for _ in range(residual_blocks)]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.out_layer = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self.rnn_layer.flatten_parameters()
        if self.repeat_initial_obs:
            obs = obs[:, None].expand(-1, actions.shape[1], -1)
        rnn_input = torch.cat((obs, actions), dim=-1)
        output, _ = self.rnn_layer(rnn_input)
        return self.out_layer(output[:, -1])


class _SequenceDynamics(nn.Module):
    repeat_initial_obs: bool

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 200,
        rnn_layers: int = 3,
        residual_blocks: int = 4,
        dropout: float = 0.1,
        repeat_initial_obs: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.repeat_initial_obs = repeat_initial_obs
        self.model = SequenceBackbone(
            obs_dim,
            action_dim,
            2 * (obs_dim + 1),
            hidden_dim,
            rnn_layers,
            residual_blocks,
            dropout,
            repeat_initial_obs,
        )
        self.obs_mu = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_std = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
        self.act_mu = nn.Parameter(torch.zeros(action_dim), requires_grad=False)
        self.act_std = nn.Parameter(torch.ones(action_dim), requires_grad=False)
        self.max_logvar = nn.Parameter(torch.full((obs_dim + 1,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((obs_dim + 1,), -10.0))

    def set_statistics(
        self,
        obs_mu: np.ndarray,
        obs_std: np.ndarray,
        act_mu: np.ndarray,
        act_std: np.ndarray,
    ) -> None:
        device = self.obs_mu.device
        self.obs_mu.data.copy_(torch.as_tensor(obs_mu, device=device))
        self.obs_std.data.copy_(torch.as_tensor(obs_std, device=device))
        self.act_mu.data.copy_(torch.as_tensor(act_mu, device=device))
        self.act_std.data.copy_(torch.as_tensor(act_std, device=device))

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_obs = (obs - self.obs_mu) / self.obs_std
        normalized_actions = (actions - self.act_mu) / self.act_std
        output = self.model(normalized_obs, normalized_actions)
        mean, logvar = torch.chunk(output, 2, dim=-1)
        return mean, soft_clamp(logvar, self.min_logvar, self.max_logvar)

    @torch.no_grad()
    def dyna_dist(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.forward(obs, actions)
        base = obs if self.repeat_initial_obs else obs[:, -1]
        mean = mean.clone()
        mean[:, :-1] += base
        std = torch.sqrt(torch.exp(logvar)).clamp_min(1e-6)
        return mean[:, :-1], std[:, :-1], mean[:, -1:], std[:, -1:]


class ADMDynamics(_SequenceDynamics):
    """Original Any-step Dynamics Model."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, repeat_initial_obs=True)


class RNNDynamics(_SequenceDynamics):
    """Bootstrapping RNN using the full state-action history as input."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, repeat_initial_obs=False)


class EnsembleLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, size: int, weight_decay: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size, input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(size, 1, output_dim))
        nn.init.trunc_normal_(self.weight, std=1 / (2 * input_dim**0.5))
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = torch.einsum("bi,eio->ebo", x, self.weight)
        else:
            x = torch.einsum("ebi,eio->ebo", x, self.weight)
        return x + self.bias

    def decay_loss(self) -> torch.Tensor:
        return 0.5 * self.weight_decay * self.weight.square().sum()


class EnsembleModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int] = (200, 200, 200, 200),
        size: int = 7,
        elites: int = 5,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)
        decays = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]
        dims = [input_dim, *hidden_dims]
        self.layers = nn.ModuleList(
            [EnsembleLinear(i, o, size, d) for i, o, d in zip(dims[:-1], dims[1:], decays[:-1])]
        )
        self.output_layer = EnsembleLinear(dims[-1], 2 * output_dim, size, decays[-1])
        self.activation = Swish()
        self.size = size
        self.num_elites = elites
        self.register_buffer("elite_indices", torch.arange(elites, dtype=torch.long))
        self.max_logvar = nn.Parameter(torch.full((output_dim,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((output_dim,), -10.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = self.activation(layer(x))
        mean, logvar = torch.chunk(self.output_layer(x), 2, dim=-1)
        return mean, soft_clamp(logvar, self.min_logvar, self.max_logvar)

    def decay_loss(self) -> torch.Tensor:
        return sum((layer.decay_loss() for layer in self.layers), self.output_layer.decay_loss())


class EnsembleDynamics(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int] = (200, 200, 200, 200),
        size: int = 7,
        elites: int = 5,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model = EnsembleModel(obs_dim + action_dim, obs_dim + 1, hidden_dims, size, elites)
        self.obs_mu = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_std = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
        self.act_mu = nn.Parameter(torch.zeros(action_dim), requires_grad=False)
        self.act_std = nn.Parameter(torch.ones(action_dim), requires_grad=False)

    def set_statistics(
        self,
        obs_mu: np.ndarray,
        obs_std: np.ndarray,
        act_mu: np.ndarray,
        act_std: np.ndarray,
    ) -> None:
        device = self.obs_mu.device
        self.obs_mu.data.copy_(torch.as_tensor(obs_mu, device=device))
        self.obs_std.data.copy_(torch.as_tensor(obs_std, device=device))
        self.act_mu.data.copy_(torch.as_tensor(act_mu, device=device))
        self.act_std.data.copy_(torch.as_tensor(act_std, device=device))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat(((obs - self.obs_mu) / self.obs_std, (action - self.act_mu) / self.act_std), dim=-1)
        return self.model(x)

    @torch.no_grad()
    def dyna_dist(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.forward(obs, action)
        mean = mean.clone()
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar)).clamp_min(1e-6)
        elite = self.model.elite_indices
        return mean[elite, ..., :-1], std[elite, ..., :-1], mean[elite, ..., -1:], std[elite, ..., -1:]
