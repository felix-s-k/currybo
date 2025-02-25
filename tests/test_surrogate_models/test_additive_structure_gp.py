import pytest
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import GPyTorchPosterior

from currybo.surrogate_models import AdditiveStructureGP


@pytest.fixture
def training_data():
    train_X = torch.rand(10, 3)
    train_W = torch.rand(10, 2)
    train_Y = torch.rand(10, 1)
    return train_X, train_W, train_Y


@pytest.fixture
def additive_gp(training_data):
    train_X, train_W, train_Y = training_data

    x_kernel = [RBFKernel]
    x_kernel_kwargs = [{'lengthscale': 1.0}]
    w_kernel = [RBFKernel]
    w_kernel_kwargs = [{'lengthscale': 1.0}]
    likelihood = GaussianLikelihood

    return AdditiveStructureGP(
        train_X=train_X,
        train_W=train_W,
        train_Y=train_Y,
        x_kernels=x_kernel,
        x_kernel_kwargs=x_kernel_kwargs,
        w_kernels=w_kernel,
        w_kernel_kwargs=w_kernel_kwargs,
        likelihood=likelihood,
        normalize_inputs=True,
        standardize_outcomes=True
    )


def test_init(additive_gp: AdditiveStructureGP):

    assert isinstance(additive_gp, AdditiveStructureGP)
    assert isinstance(additive_gp.likelihood, GaussianLikelihood)
    assert isinstance(additive_gp.covar_module, ScaleKernel)
    assert isinstance(additive_gp.input_transform, Normalize)
    assert isinstance(additive_gp.outcome_transform, Standardize)


def test_fit(additive_gp: AdditiveStructureGP):
    assert not additive_gp.trained
    additive_gp.fit()
    assert additive_gp.trained


def test_posterior(additive_gp: AdditiveStructureGP):
    X = torch.rand(5, 3)
    W = torch.rand(5, 2)
    posterior = additive_gp.posterior(X, W)
    assert isinstance(posterior, GPyTorchPosterior)
    assert posterior.mean.shape == (5, 1)
    assert posterior.variance.shape == (5, 1)
