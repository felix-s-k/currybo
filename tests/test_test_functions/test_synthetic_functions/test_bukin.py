import torch
import random
import numpy as np
from currybo.test_functions import (
    ParametrizedBukin,
    ParametrizedBaseTestProblem,
)


def test_evaluate_true_tensor():

    surface = ParametrizedBukin()

    assert isinstance(surface, ParametrizedBaseTestProblem)

    assert surface.dim == 2
    assert len(surface._bounds) == surface.dim
    for bound in surface._bounds:
        assert isinstance(bound, tuple) and len(bound) == 2

    dummy_X = torch.rand((random.randint(1, 5), surface.dim))
    dummy_output = surface.evaluate_true(dummy_X)

    assert isinstance(dummy_output, torch.Tensor)
    assert dummy_X.shape[0] == dummy_output.shape[0]
    assert dummy_output.dim() == 1


def test_evaluate_true_numpy():

    surface = ParametrizedBukin()

    assert isinstance(surface, ParametrizedBaseTestProblem)

    assert surface.dim == 2
    assert len(surface._bounds) == surface.dim
    for bound in surface._bounds:
        assert isinstance(bound, tuple) and len(bound) == 2

    dummy_X = torch.rand((random.randint(1, 5), surface.dim)).detach().numpy()
    dummy_output = surface.evaluate_true(dummy_X)

    assert isinstance(dummy_output, np.ndarray)
    assert dummy_X.shape[0] == dummy_output.shape[0]
    assert dummy_output.ndim == 1


def test_get_scalarization_factor():
    surface = ParametrizedBukin()

    min_val, max_val = surface.get_scalarization_factor()
    
    assert isinstance(min_val, float), "min_val should be a float"
    assert isinstance(max_val, float), "max_val should be a float"
    assert min_val <= max_val, "min_val should be less than max_val"
