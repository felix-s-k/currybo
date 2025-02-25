from typing import Tuple

import torch
from torch import Tensor
from ..parametrizable_function import ParametrizedBaseTestProblem
from scipy.optimize import shgo
from typing import Union
import numpy as np

class ParametrizedThreeHumpCamel(ParametrizedBaseTestProblem):
    """
    Parametrized version of the ThreeHumpCamel test function.
    """
    _parameter_defaults = {
        "a": 2,
        "b": 1.05,
        "c": 6,
    }
    
    dim = 2
    _bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    def evaluate_true(self, X: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        input_is_numpy = False
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            input_is_numpy = True
            
        a = self.a
        b = self.b
        c = self.c

        x1, x2 = X[..., 0], X[..., 1]
        if self.negate:
            result = -(a * x1**2 - b * x1**4 + x1**6 / c + x1 * x2 + x2**2)
        else:
            result = a * x1**2 - b * x1**4 + x1**6 / c + x1 * x2 + x2**2

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