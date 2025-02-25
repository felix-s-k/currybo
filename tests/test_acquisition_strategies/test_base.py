from typing import Tuple, Type

import pytest
import torch
from torch import Tensor
from botorch.posteriors.torch import TorchPosterior
from botorch.acquisition.monte_carlo import SampleReductionProtocol
from botorch.sampling import MCSampler

from currybo.acquisition_strategies import BaseMCAcquisitionStrategy
from currybo.aggregation_functions import BaseAggregation
from currybo.surrogate_models import BaseSurrogate


class MockMCAcquisitionStrategy(BaseMCAcquisitionStrategy):
    def __init__(
        self,
        x_bounds: Tensor,
        x_options: Tensor,
        w_options: Tensor,
        aggregation_function: BaseAggregation,
        sample_reduction: SampleReductionProtocol = torch.mean,
        q_reduction: SampleReductionProtocol = torch.amax,
        sampler_type: Type[MCSampler] = MCSampler,
        num_mc_samples: int = 512,
        maximization: bool = True,
        **kwargs
    ):
        super().__init__(
            x_bounds,
            x_options,
            w_options,
            aggregation_function,
            sample_reduction,
            q_reduction,
            sampler_type,
            num_mc_samples,
            maximization,
            **kwargs
        )

    def get_recommendation(
        self, model: BaseSurrogate, q: int = 1, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        return torch.rand(q, model.train_X.shape[-1]), torch.randint(0, model.train_W.shape[-1], (q,))

    def get_final_recommendation(self, model: BaseSurrogate, **kwargs) -> Tensor:
        return torch.rand(model.train_X.shape[-1])


@pytest.fixture
def mock_aggregation_function():
    class MockAggregationFunction(BaseAggregation):
        def forward(self, Y: Tensor, X: Tensor = None, W: Tensor = None) -> Tensor:
            return Y.mean(dim=-1)

    return MockAggregationFunction()


@pytest.fixture
def mock_surrogate_model():
    class MockSurrogateModel(BaseSurrogate):
        def __init__(self):
            self.train_X = torch.rand(10, 2)  # Example tensor shape
            self.train_W = torch.randint(0, 2, (10,))  # Example tensor shape
        
        def fit():  # Empty, because this is mock
            pass

        def posterior(self, X: Tensor, W: Tensor) -> TorchPosterior:  # Empty, because this is mock
            pass

        def condition_on_observations(self, X: Tensor, Y: Tensor, W: Tensor) -> BaseSurrogate:
            pass

    return MockSurrogateModel()


@pytest.fixture
def mock_mc_acquisition_strategy(mock_aggregation_function):
    x_bounds = torch.tensor([[0.0, 1.0]])
    w_options = torch.tensor([[0.0, 1.0]])
    return MockMCAcquisitionStrategy(
        x_bounds=x_bounds,
        x_options=None,
        w_options=w_options,
        aggregation_function=mock_aggregation_function,
    )


def test_mock_mc_acquisition_strategy_init(mock_aggregation_function):
    x_bounds = torch.tensor([[0.0, 1.0]])
    w_options = torch.tensor([[0.0, 1.0]])
    strategy = MockMCAcquisitionStrategy(
        x_bounds=x_bounds,
        x_options=None,
        w_options=w_options,
        aggregation_function=mock_aggregation_function,
    )

    assert torch.allclose(strategy.x_bounds, x_bounds)
    assert torch.allclose(strategy.w_options, w_options)
    assert strategy.aggregation_function == mock_aggregation_function
    assert strategy.sample_reduction == torch.mean
    assert strategy.q_reduction == torch.amax
    assert strategy.sampler_type == MCSampler
    assert strategy.num_mc_samples == 512
    assert strategy.maximization


def test_mock_mc_acquisition_strategy_get_recommendation(
    mock_mc_acquisition_strategy, mock_surrogate_model
):
    recommendation = mock_mc_acquisition_strategy.get_recommendation(mock_surrogate_model)
    assert isinstance(recommendation, tuple)
    assert isinstance(recommendation[0], Tensor)
    assert isinstance(recommendation[1], Tensor)


def test_mock_mc_acquisition_strategy_get_final_recommendation(
    mock_mc_acquisition_strategy, mock_surrogate_model
):
    final_recommendation = mock_mc_acquisition_strategy.get_final_recommendation(mock_surrogate_model)
    assert isinstance(final_recommendation, Tensor)
