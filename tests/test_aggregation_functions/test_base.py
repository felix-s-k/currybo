import pytest
import torch
from abc import ABCMeta, abstractmethod
from typing import Optional
from torch import Tensor

from currybo.aggregation_functions import BaseAggregation


class ConcreteAggregation(BaseAggregation):
    def forward(self, Y: Tensor, X: Optional[Tensor] = None, W: Optional[Tensor] = None) -> Tensor:
        return torch.mean(Y, dim=-1)


@pytest.fixture
def sample_tensors():
    return {
        "5d_tensor": torch.rand(2, 3, 4, 1, 5),
        "2d_tensor": torch.rand(6, 5)
    }


@pytest.fixture
def aggregation():
    return ConcreteAggregation()


def test_align_y_shape_5d(aggregation, sample_tensors):
    Y = sample_tensors["5d_tensor"]
    Y_aligned = aggregation._align_y_shape(Y)
    assert Y_aligned.shape == (2, 3, 4, 5)


def test_align_y_shape_2d(aggregation, sample_tensors):
    Y = sample_tensors["2d_tensor"]
    Y_aligned = aggregation._align_y_shape(Y)
    assert Y_aligned.shape == (6, 5)


def test_align_y_shape_invalid_dim(aggregation):
    Y = torch.rand(6, 5, 4)
    with pytest.raises(NotImplementedError):
        aggregation._align_y_shape(Y)

    Y = torch.rand(6)
    with pytest.raises(ValueError):
        aggregation._align_y_shape(Y)


def test_call_method(aggregation, sample_tensors):
    Y = sample_tensors["5d_tensor"]
    result = aggregation(Y)
    assert result.shape == (2, 3, 4)

    Y = sample_tensors["2d_tensor"]
    result = aggregation(Y)
    assert result.shape == (6,)


def test_forward_method(aggregation, sample_tensors):
    Y = sample_tensors["2d_tensor"]
    result = aggregation.forward(Y)
    assert torch.allclose(result, torch.mean(Y, dim=-1))
