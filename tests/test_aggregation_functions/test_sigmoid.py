import pytest
import torch
from currybo.aggregation_functions import Sigmoid


@pytest.fixture
def sample_tensors():
    return {
        "5d_tensor": torch.rand(2, 3, 4, 1, 5),
        "2d_tensor": torch.rand(6, 5),
        "aligned_5d_tensor": torch.rand(2, 3, 4, 5),
    }

@pytest.fixture
def sigmoid_aggregation():
    return Sigmoid(threshold=0.5, k=300)


def test_sigmoid_init():
    with pytest.raises(ValueError):
        Sigmoid(threshold=0.5, k=700)  # Exponential factor too high

    sigmoid = Sigmoid(threshold=0.5, k=300)
    assert sigmoid.threshold == 0.5
    assert sigmoid.k == 300


def test_align_y_shape_5d(sigmoid_aggregation, sample_tensors):
    Y = sample_tensors["5d_tensor"]
    Y_aligned = sigmoid_aggregation._align_y_shape(Y)
    assert Y_aligned.shape == (2, 3, 4, 5)


def test_align_y_shape_2d(sigmoid_aggregation, sample_tensors):
    Y = sample_tensors["2d_tensor"]
    Y_aligned = sigmoid_aggregation._align_y_shape(Y)
    assert Y_aligned.shape == (6, 5)


def test_align_y_shape_invalid_dim(sigmoid_aggregation):
    Y = torch.rand(6, 5, 4)
    with pytest.raises(NotImplementedError):
        sigmoid_aggregation._align_y_shape(Y)

    Y = torch.rand(6)
    with pytest.raises(ValueError):
        sigmoid_aggregation._align_y_shape(Y)


def test_call_method(sigmoid_aggregation, sample_tensors):
    Y = sample_tensors["5d_tensor"]
    result = sigmoid_aggregation(Y)
    assert result.shape == (2, 3, 4)

    Y = sample_tensors["2d_tensor"]
    result = sigmoid_aggregation(Y)
    assert result.shape == (6,)


def test_forward_method(sigmoid_aggregation, sample_tensors):
    Y = sample_tensors["aligned_5d_tensor"]
    expected_result = torch.sum(1 / (1 + torch.exp(-sigmoid_aggregation.k * (Y - sigmoid_aggregation.threshold))), dim=-1) / Y.shape[-1]
    result = sigmoid_aggregation.forward(Y)
    assert torch.allclose(result, expected_result)


def test_forward_method_2d(sigmoid_aggregation, sample_tensors):
    Y = sample_tensors["2d_tensor"]
    expected_result = torch.sum(1 / (1 + torch.exp(-sigmoid_aggregation.k * (Y - sigmoid_aggregation.threshold))), dim=-1) / Y.shape[-1]
    result = sigmoid_aggregation.forward(Y)
    assert torch.allclose(result, expected_result)
