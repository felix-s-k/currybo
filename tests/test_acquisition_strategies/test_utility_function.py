import pytest
import torch
from currybo.acquisition_strategies.utility_function import (
    Random,
    SimpleRegret,
    UncertaintyUtility,
    QuantileUtility,
    QuantitativeImprovement,
    QualitativeImprovement
)


# Mock input samples for testing
@pytest.fixture
def mock_samples():
    return torch.arange(9).float().reshape(3, 3) + 1


# Test for Random utility function
def test_random(mock_samples):
    utility = Random()
    result = utility(mock_samples)
    assert result.shape == mock_samples.shape


# Test for SimpleRegret utility function
def test_simple_regret_maximize(mock_samples):
    utility = SimpleRegret(maximize=True)
    result = utility(mock_samples)
    assert torch.allclose(result, mock_samples)


def test_simple_regret_minimize(mock_samples):
    utility = SimpleRegret(maximize=False)
    result = utility(mock_samples)
    assert torch.allclose(result, -mock_samples)


# Test for UncertaintyUtility function
def test_uncertainty_utility(mock_samples):
    utility = UncertaintyUtility()
    result = utility(mock_samples)
    expected = torch.abs(mock_samples - mock_samples.mean(dim=0, keepdim=True))
    assert torch.allclose(result, expected)


# Test for QuantileUtility function
def test_quantile_utility_maximize(mock_samples):
    beta = 0.5
    utility = QuantileUtility(beta=beta, maximize=True)
    result = utility(mock_samples)
    mean = mock_samples.mean(dim=0, keepdim=True)
    expected = mean + beta * torch.abs(mock_samples - mean)
    assert torch.allclose(result, expected)


def test_quantile_utility_minimize(mock_samples):
    beta = 0.5
    utility = QuantileUtility(beta=beta, maximize=False)
    result = utility(mock_samples)
    mean = mock_samples.mean(dim=0, keepdim=True)
    expected = -mean + beta * torch.abs(mock_samples - mean)
    assert torch.allclose(result, expected)


# Test for QuantitativeImprovement function
def test_quantitative_improvement_maximize(mock_samples):
    best_f = 4.0
    utility = QuantitativeImprovement(best_f=best_f, maximize=True)
    result = utility(mock_samples)
    expected = (mock_samples - best_f).clamp(min=0)
    assert torch.allclose(result, expected)


def test_quantitative_improvement_minimize(mock_samples):
    best_f = 4.0
    utility = QuantitativeImprovement(best_f=best_f, maximize=False)
    result = utility(mock_samples)
    expected = (best_f - mock_samples).clamp(min=0)
    assert torch.allclose(result, expected)


# Test for QualitativeImprovement function
def test_qualitative_improvement_maximize(mock_samples):
    best_f = 4.0
    tau = 0.01
    utility = QualitativeImprovement(best_f=best_f, tau=tau, maximize=True)
    result = utility(mock_samples)
    expected = torch.sigmoid((mock_samples - best_f) / tau)
    assert torch.allclose(result, expected)


def test_qualitative_improvement_minimize(mock_samples):
    best_f = 4.0
    tau = 0.01
    utility = QualitativeImprovement(best_f=best_f, tau=tau, maximize=False)
    result = utility(mock_samples)
    expected = torch.sigmoid((best_f - mock_samples) / tau)
    assert torch.allclose(result, expected)
