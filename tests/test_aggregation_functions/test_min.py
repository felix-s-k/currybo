import pytest
import torch
from currybo.aggregation_functions import Min


@pytest.fixture
def sample_tensors():
    return {
        "5d_tensor": torch.rand(2, 3, 4, 1, 5),
        "2d_tensor": torch.rand(6, 5),
        "aligned_5d_tensor": torch.rand(2, 3, 4, 5),
    }


@pytest.fixture
def min_aggregation():
    return Min()


def test_align_y_shape_5d(min_aggregation, sample_tensors):
    Y = sample_tensors["5d_tensor"]
    Y_aligned = min_aggregation._align_y_shape(Y)
    assert Y_aligned.shape == (2, 3, 4, 5)


def test_align_y_shape_2d(min_aggregation, sample_tensors):
    Y = sample_tensors["2d_tensor"]
    Y_aligned = min_aggregation._align_y_shape(Y)
    assert Y_aligned.shape == (6, 5)


def test_align_y_shape_invalid_dim(min_aggregation):
    Y = torch.rand(6, 5, 4)
    with pytest.raises(NotImplementedError):
        min_aggregation._align_y_shape(Y)

    Y = torch.rand(6)
    with pytest.raises(ValueError):
        min_aggregation._align_y_shape(Y)


def test_call_method(min_aggregation, sample_tensors):
    Y = sample_tensors["5d_tensor"]
    result = min_aggregation(Y)
    assert result.shape == (2, 3, 4)

    Y = sample_tensors["2d_tensor"]
    result = min_aggregation(Y)
    assert result.shape == (6,)


def test_forward_method(min_aggregation, sample_tensors):
    Y = sample_tensors["aligned_5d_tensor"]
    expected_result = Y.min(dim=-1)[0]
    result = min_aggregation.forward(Y)
    assert torch.allclose(result, expected_result)


def test_forward_method_2d(min_aggregation, sample_tensors):
    Y = sample_tensors["2d_tensor"]
    expected_result = Y.min(dim=-1)[0]
    result = min_aggregation.forward(Y)
    assert torch.allclose(result, expected_result)
