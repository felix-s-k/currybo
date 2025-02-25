import pytest
import torch
from currybo.aggregation_functions import Mean


@pytest.fixture
def sample_tensors():
    return {
        "5d_tensor": torch.rand(2, 3, 4, 1, 5),
        "2d_tensor": torch.rand(6, 5),
        "aligned_5d_tensor": torch.rand(2, 3, 4, 5),
    }


@pytest.fixture
def mean_aggregation():
    return Mean()


def test_align_y_shape_5d(mean_aggregation, sample_tensors):
    Y = sample_tensors["5d_tensor"]
    Y_aligned = mean_aggregation._align_y_shape(Y)
    assert Y_aligned.shape == (2, 3, 4, 5)


def test_align_y_shape_2d(mean_aggregation, sample_tensors):
    Y = sample_tensors["2d_tensor"]
    Y_aligned = mean_aggregation._align_y_shape(Y)
    assert Y_aligned.shape == (6, 5)


def test_align_y_shape_invalid_dim(mean_aggregation):
    Y = torch.rand(6, 5, 4)
    with pytest.raises(NotImplementedError):
        mean_aggregation._align_y_shape(Y)

    Y = torch.rand(6)
    with pytest.raises(ValueError):
        mean_aggregation._align_y_shape(Y)


def test_call_method(mean_aggregation, sample_tensors):
    Y = sample_tensors["5d_tensor"]
    result = mean_aggregation(Y)
    assert result.shape == (2, 3, 4)

    Y = sample_tensors["2d_tensor"]
    result = mean_aggregation(Y)
    assert result.shape == (6,)


def test_forward_method(mean_aggregation, sample_tensors):
    Y = sample_tensors["aligned_5d_tensor"]
    expected_result = Y.mean(dim=-1)
    result = mean_aggregation.forward(Y)
    assert torch.allclose(result, expected_result)


def test_forward_method_2d(mean_aggregation, sample_tensors):
    Y = sample_tensors["2d_tensor"]
    expected_result = Y.mean(dim=-1)
    result = mean_aggregation.forward(Y)
    assert torch.allclose(result, expected_result)
