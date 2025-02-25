from typing import Tuple

import torch
from torch import Tensor
from ..parametrizable_function import ParametrizedBaseTestProblem
from scipy.optimize import shgo
from typing import Union
import numpy as np

class ParametrizedDixonPrice(ParametrizedBaseTestProblem):
    """
    Parametrized version of a one-dimensional DixonPrice test function.
    """
    _parameter_defaults = {
        "a": 1.0,
        "b": 2.0,
    }

    dim = 1
    _bounds = [(-5.0, 5.0)]

    def evaluate_true(self, X: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        input_is_numpy = False
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            input_is_numpy = True

        d = self.dim
        part1 = (X[..., 0] - self.a) ** 2
        i = X.new(range(2, d + 1))
        part2 = torch.sum(i * (self.b * X[..., 1:] ** 2 - X[..., :-1]) ** 2, dim=-1)
        if self.negate:
            result = -(part1 + part2)
        else:
            result = part1 + part2

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