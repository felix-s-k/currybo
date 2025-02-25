import pytest
import torch
from currybo.aggregation_functions import MSE


@pytest.fixture
def sample_tensors():
    return {
        "5d_tensor": torch.rand(2, 3, 4, 1, 5),
        "2d_tensor": torch.rand(6, 5),
        "aligned_5d_tensor": torch.rand(2, 3, 4, 5),
    }


@pytest.fixture
def mse_aggregation():
    return MSE(maximum=1.0)


def test_mse_init(mse_aggregation):
    assert mse_aggregation.maximum == 1.0


def test_align_y_shape_5d(mse_aggregation, sample_tensors):
    Y = sample_tensors["5d_tensor"]
    Y_aligned = mse_aggregation._align_y_shape(Y)
    assert Y_aligned.shape == (2, 3, 4, 5)


def test_align_y_shape_2d(mse_aggregation, sample_tensors):
    Y = sample_tensors["2d_tensor"]
    Y_aligned = mse_aggregation._align_y_shape(Y)
    assert Y_aligned.shape == (6, 5)


def test_align_y_shape_invalid_dim(mse_aggregation):
    Y = torch.rand(6, 5, 4)
    with pytest.raises(NotImplementedError):
        mse_aggregation._align_y_shape(Y)

    Y = torch.rand(6)
    with pytest.raises(ValueError):
        mse_aggregation._align_y_shape(Y)


def test_call_method(mse_aggregation, sample_tensors):
    Y = sample_tensors["5d_tensor"]
    result = mse_aggregation(Y)
    assert result.shape == (2, 3, 4)

    Y = sample_tensors["2d_tensor"]
    result = mse_aggregation(Y)
    assert result.shape == (6,)


def test_forward_method(mse_aggregation, sample_tensors):
    Y = sample_tensors["aligned_5d_tensor"]
    maximum = mse_aggregation.maximum
    expected_result = -torch.sum((Y - maximum)**2, dim=-1) / Y.shape[-1]
    result = mse_aggregation.forward(Y)
    assert torch.allclose(result, expected_result)


def test_forward_method_2d(mse_aggregation, sample_tensors):
    Y = sample_tensors["2d_tensor"]
    maximum = mse_aggregation.maximum
    expected_result = -torch.sum((Y - maximum)**2, dim=-1) / Y.shape[-1]
    result = mse_aggregation.forward(Y)
    assert torch.allclose(result, expected_result)
