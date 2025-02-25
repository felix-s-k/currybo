from typing import Optional

import pytest
import random
import torch
from torch import Tensor
from botorch.posteriors.gpytorch import GPyTorchPosterior

from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal

from currybo.acquisition_strategies import (
    BaseMCAcquisitionStrategy, SequentialAcquisition,
    QuantitativeImprovement, UncertaintyUtility,
)
from currybo.surrogate_models import BaseSurrogate, SimpleGP
from currybo.aggregation_functions import BaseAggregation, Mean


# Mock input tensors for testing
@pytest.fixture
def mock_x_bounds():
    return torch.tensor([[0.0, 0.0], [1.0, 1.0]])


@pytest.fixture
def mock_w_options():
    return torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])


# Mock BaseSurrogate class for testing
class MockSurrogate(BaseSurrogate):
    def __init__(self):
        self.trained = True

    def fit(self, X_train: Tensor, Y_train: Tensor):
        # Mock fitting process
        self.trained = True

    def posterior(self, X: Tensor, W: Tensor):
        mean = X + W
        covar = torch.eye(mean.shape[-1])
        mvn = MultivariateNormal(mean, covar)
        return GPyTorchPosterior(mvn)


# Mock BaseAggregation class for testing
class MockAggregation(BaseAggregation):
    def forward(self, Y: Tensor, X: Optional[Tensor] = None, W: Optional[Tensor] = None) -> Tensor:
        return Y.mean(dim=-1)


@pytest.fixture
def mock_model():
    return MockSurrogate()


@pytest.fixture
def mock_aggregation_function():
    return MockAggregation()


# Test SequentialAcquisition initialization
def test_sequential_acquisition_init(mock_x_bounds, mock_w_options, mock_aggregation_function):
    acquisition = SequentialAcquisition(
        x_bounds=mock_x_bounds,
        w_options=mock_w_options,
        aggregation_function=mock_aggregation_function,
        x_utility= QuantitativeImprovement,
        x_utility_kwargs = {"best_f": 0},
        w_utility= UncertaintyUtility,
        w_utility_kwargs = None
    )
    assert acquisition.x_bounds is mock_x_bounds
    assert acquisition.w_options is mock_w_options
    assert acquisition.aggregation_function is mock_aggregation_function
    assert acquisition.maximization is True


# Test get_recommendation method
def test_get_recommendation():

    NUM_FUNCTIONS = 3
    BATCH_SIZE = 1
    NUM_OBJECTIVES = 1

    x_bounds = torch.rand((2, random.randint(1, 5)))
    mask = x_bounds[0, :] > x_bounds[1, :]
    x_bounds[0, mask], x_bounds[1, mask] = x_bounds[1, mask], x_bounds[0, mask]
    w_options = torch.rand((NUM_FUNCTIONS, random.randint(1, 5))) #one objective

    seq_acq = SequentialAcquisition(
        x_bounds=x_bounds,
        w_options=w_options,
        aggregation_function=Mean(),
        x_utility= QuantitativeImprovement,
        x_utility_kwargs = {"best_f": 0},
        w_utility= UncertaintyUtility,
        w_utility_kwargs = None
    )

    assert isinstance(seq_acq, BaseMCAcquisitionStrategy)

    observations_x = x_bounds[0] + (x_bounds[1] - x_bounds[0]) * torch.rand(random.randint(1, 5), x_bounds.shape[1])
    observations_y = torch.rand((observations_x.shape[0], NUM_OBJECTIVES))
    observations_w = w_options[torch.randint(0, w_options.shape[0], (observations_x.shape[0],))]

    surrogate_type = SimpleGP
    surrogate_kwargs = {"kernel": MaternKernel, "likelihood": GaussianLikelihood}

    surrogate = surrogate_type(train_X = observations_x, train_W = observations_w, train_Y = observations_y, **surrogate_kwargs)

    surrogate.fit()

    next_x, next_w_idx = seq_acq.get_recommendation(surrogate, BATCH_SIZE) # batch size 1

    assert isinstance(next_x, torch.Tensor)
    assert isinstance(next_w_idx, torch.Tensor)
    assert next_x.dim() == 2
    assert next_w_idx.dim() == 1

    assert next_x.shape[0] == BATCH_SIZE
    assert next_x.shape[-1] == observations_x.shape[-1]

    assert next_w_idx.shape[0] == BATCH_SIZE


# Test get_final_recommendation method
def test_get_final_recommendation():
    
    NUM_FUNCTIONS = 3
    NUM_OBJECTIVES = 1

    x_bounds = torch.rand((2, random.randint(1, 5)))
    mask = x_bounds[0, :] > x_bounds[1, :]
    x_bounds[0, mask], x_bounds[1, mask] = x_bounds[1, mask], x_bounds[0, mask]
    w_options = torch.rand((NUM_FUNCTIONS, random.randint(1, 5))) #one objective

    seq_acq = SequentialAcquisition(
        x_bounds=x_bounds,
        w_options=w_options,
        aggregation_function=Mean(),
        x_utility= QuantitativeImprovement,
        x_utility_kwargs = {"best_f": 0},
        w_utility= UncertaintyUtility,
        w_utility_kwargs = None
    )

    observations_x = x_bounds[0] + (x_bounds[1] - x_bounds[0]) * torch.rand(random.randint(1, 5), x_bounds.shape[1])
    observations_y = torch.rand((observations_x.shape[0], NUM_OBJECTIVES))
    observations_w = w_options[torch.randint(0, w_options.shape[0], (observations_x.shape[0],))]

    surrogate_type = SimpleGP
    surrogate_kwargs = {"kernel": MaternKernel, "likelihood": GaussianLikelihood}

    surrogate = surrogate_type(train_X = observations_x, train_W = observations_w, train_Y = observations_y, **surrogate_kwargs)

    surrogate.fit()

    final_x = seq_acq.get_final_recommendation(surrogate)

    assert isinstance(final_x, Tensor)
