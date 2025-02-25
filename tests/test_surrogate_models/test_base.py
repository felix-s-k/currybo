from typing import Optional

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors.torch import TorchPosterior

from currybo.surrogate_models import BaseSurrogate


class MockSurrogate(BaseSurrogate, SingleTaskGP):
    def __init__(
        self,
        train_X: torch.Tensor,
        train_W: torch.Tensor,
        train_Y: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs
    ):
        self.train_X = train_X
        self.train_W = train_W
        self.train_Y = train_Y
        self.input_transform = input_transform
        self.outcome_transform = outcome_transform

    def fit(self) -> None:
        self._trained = True

    def posterior(self, X: torch.Tensor, W: Optional[torch.Tensor] = None) -> TorchPosterior:
        return TorchPosterior(X)  # Mock implementation for testing

    def condition_on_observations(self, X: torch.Tensor, Y: torch.Tensor, W: Optional[torch.Tensor] = None) -> BaseSurrogate:
        # Return a new instance of the surrogate for testing
        return MockSurrogate(X, W, Y)


# Test the initialization of a mock subclass of BaseSurrogate
def test_mock_surrogate_initialization():
    train_X = torch.randn(5, 3)
    train_W = torch.randn(5, 2)
    train_Y = torch.randn(5, 1)

    surrogate = MockSurrogate(train_X, train_W, train_Y)
    assert isinstance(surrogate, BaseSurrogate)
    assert surrogate.train_X.shape == (5, 3)
    assert surrogate.train_W.shape == (5, 2)
    assert surrogate.train_Y.shape == (5, 1)


# Test the fit method in MockSurrogate
def test_fit_method():
    train_X = torch.randn(5, 3)
    train_W = torch.randn(5, 2)
    train_Y = torch.randn(5, 1)

    surrogate = MockSurrogate(train_X, train_W, train_Y)

    # Check that the model is not trained initially
    assert surrogate.trained is False

    # Call fit and verify that the model is now trained
    surrogate.fit()
    assert surrogate.trained is True


# Test the posterior method in MockSurrogate
def test_posterior_method():
    train_X = torch.randn(5, 3)
    train_W = torch.randn(5, 2)
    train_Y = torch.randn(5, 1)

    surrogate = MockSurrogate(train_X, train_W, train_Y)

    # Call the posterior method and check that it returns a TorchPosterior object
    posterior = surrogate.posterior(torch.randn(3, 3))
    assert isinstance(posterior, TorchPosterior)


# Test condition_on_observations method
def test_condition_on_observations():
    train_X = torch.randn(5, 3)
    train_W = torch.randn(5, 2)
    train_Y = torch.randn(5, 1)

    surrogate = MockSurrogate(train_X, train_W, train_Y)

    # Call condition_on_observations and ensure it returns a new surrogate model
    X_new = torch.randn(3, 3)
    Y_new = torch.randn(3, 1)
    updated_surrogate = surrogate.condition_on_observations(X_new, Y_new)

    assert isinstance(updated_surrogate, MockSurrogate)
    assert updated_surrogate.train_X.shape == (3, 3)
    assert updated_surrogate.train_Y.shape == (3, 1)


# Test trained property setter and getter directly
def test_trained_property_setter_getter():
    train_X = torch.randn(5, 3)
    train_W = torch.randn(5, 2)
    train_Y = torch.randn(5, 1)

    surrogate = MockSurrogate(train_X, train_W, train_Y)

    # Set the trained property directly
    surrogate.trained = True
    assert surrogate.trained is True

    surrogate.trained = False
    assert surrogate.trained is False
