from typing import Tuple

import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import GPyTorchPosterior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel

from currybo.surrogate_models import SimpleGP


@pytest.fixture
def training_data():
    train_X = torch.rand(10, 3)
    train_W = torch.rand(10, 2)
    train_Y = torch.rand(10, 1)
    return train_X, train_W, train_Y


@pytest.fixture
def simple_gp(training_data):
    train_X, train_W, train_Y = training_data
    return SimpleGP(
        train_X=train_X,
        train_W=train_W,
        train_Y=train_Y,
        kernel=RBFKernel,
        likelihood=GaussianLikelihood,
        normalize_inputs=True,
        standardize_outcomes=True
    )


def test_init(simple_gp: SimpleGP, training_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    train_X, train_W, train_Y = training_data
    train_inputs = torch.cat((train_X, train_W), dim=-1)

    assert simple_gp.train_inputs[0].shape == train_inputs.shape
    assert torch.allclose(simple_gp.train_inputs[0], train_inputs)
    assert simple_gp.train_targets.shape == train_Y.squeeze().shape
    assert torch.allclose(simple_gp.train_targets, Standardize(m=1)(train_Y)[0].squeeze())
    assert isinstance(simple_gp.covar_module, ScaleKernel)
    assert isinstance(simple_gp.covar_module.base_kernel, RBFKernel)
    assert isinstance(simple_gp.likelihood, GaussianLikelihood)
    assert isinstance(simple_gp.input_transform, Normalize)
    assert isinstance(simple_gp.outcome_transform, Standardize)


def test_fit(simple_gp):
    assert not simple_gp.trained
    simple_gp.fit()
    assert simple_gp.trained


def test_posterior(simple_gp):
    X = torch.rand(5, 3)
    W = torch.rand(5, 2)
    posterior = simple_gp.posterior(X, W)
    assert isinstance(posterior, GPyTorchPosterior)
    assert posterior.mean.shape == (5, 1)
    assert posterior.variance.shape == (5, 1)
