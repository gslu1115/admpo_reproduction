import numpy as np
import torch

from admpo_repro.dynamics import ADMDynamics, EnsembleDynamics, RNNDynamics
from admpo_repro.evaluation.figure4 import prediction_statistics


def test_adm_and_rnn_output_shapes():
    batch, obs_dim, action_dim, length = 4, 3, 2, 5
    adm = ADMDynamics(obs_dim, action_dim, hidden_dim=8, rnn_layers=1, residual_blocks=1, dropout=0)
    rnn = RNNDynamics(obs_dim, action_dim, hidden_dim=8, rnn_layers=1, residual_blocks=1, dropout=0)
    obs = torch.zeros(batch, obs_dim)
    obs_seq = torch.zeros(batch, length, obs_dim)
    actions = torch.zeros(batch, length, action_dim)
    for model, model_obs in ((adm, obs), (rnn, obs_seq)):
        mean, logvar = model(model_obs, actions)
        assert mean.shape == (batch, obs_dim + 1)
        assert logvar.shape == mean.shape


def test_ensemble_shape_and_elites():
    model = EnsembleDynamics(3, 2, hidden_dims=(8, 8, 8, 8), size=7, elites=5)
    means, stds, rewards, reward_stds = model.dyna_dist(torch.zeros(4, 3), torch.zeros(4, 2))
    assert means.shape == (5, 4, 3)
    assert stds.shape == means.shape
    assert rewards.shape == (5, 4, 1)
    assert reward_stds.shape == rewards.shape
    assert not torch.equal(model.model.layers[0].weight[0], model.model.layers[0].weight[1])


def test_uncertainty_formula_matches_manual_variance():
    torch.manual_seed(0)
    model = EnsembleDynamics(3, 2, hidden_dims=(8, 8, 8, 8), size=7, elites=5)
    obs_hist = np.zeros((4, 5, 3), dtype=np.float32)
    act_hist = np.zeros((4, 4, 2), dtype=np.float32)
    action = np.zeros((4, 2), dtype=np.float32)
    _, uncertainty, total, _ = prediction_statistics(
        "ensemble", model, obs_hist, act_hist, action, "cpu"
    )
    means, stds, _, _ = model.dyna_dist(torch.zeros(4, 3), torch.zeros(4, 2))
    expected = torch.sqrt(means.var(dim=0).mean(dim=-1)).detach().numpy()
    expected_total = torch.sqrt((means.var(dim=0) + stds.square().mean(dim=0)).mean(dim=-1)).detach().numpy()
    np.testing.assert_allclose(uncertainty, expected, rtol=1e-6)
    np.testing.assert_allclose(total, expected_total, rtol=1e-6)
