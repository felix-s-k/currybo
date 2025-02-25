from typing import Tuple

import torch
from torch import Tensor
from ..parametrizable_function import ParametrizedBaseTestProblem
from scipy.optimize import shgo
from typing import Union
import numpy as np

class ParametrizedBeale(ParametrizedBaseTestProblem):
    """
    Parametrized version of the Beale test function.
    """
    _parameter_defaults = {
        "a": 1.5,
        "b": 2.25,
        "c": 2.625,
    }

    dim = 2
    _bounds = [(-4.5, 4.5), (-4.5, 4.5)]

    
    def evaluate_true(self, X: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:

        input_is_numpy = False
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            input_is_numpy = True
        
        a = self.a
        b = self.b
        c = self.c

        x1, x2 = X[..., 0], X[..., 1]
        part1 = (a - x1 + x1 * x2) ** 2
        part2 = (b - x1 + x1 * x2**2) ** 2
        part3 = (c - x1 + x1 * x2**3) ** 2

        if self.negate:
            result = -(part1 + part2 + part3)
        else:
            result = part1 + part2 + part3

        if self.max_val is not None and self.min_val is not None:
            result = (result - (self.min_val + self.max_val) / 2) / (self.max_val - self.min_val) * (self.upper - self.lower) + (self.upper + self.lower) / 2     

        if input_is_numpy:
            result = result.numpy()
        return result
    
    def get_scalarization_factor(self) -> Tuple[float, float]:
        min_result = shgo(func = self.evaluate_true, bounds = self._bounds)
        min_val = min_result.fun

        self.negate = not self.negate
        max_result = shgo(func = self.evaluate_true, bounds = self._bounds)
        self.negate = not self.negate
        max_val = -1 * max_result.fun

        return min_val, max_val
