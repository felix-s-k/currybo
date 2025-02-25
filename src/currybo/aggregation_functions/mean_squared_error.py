from typing import Optional
from torch import Tensor
import torch

from .base import BaseAggregation


class MSE(BaseAggregation):
    """
    The mean squared error aggregation function is defined as the mean squared error \
        of the function values compared to a known function maximum.

    Args:
        maximum (float): The maximum value the function can take.
    """

    differentiable = True

    def __init__(self, maximum: float) -> None:
        self.maximum = maximum

    def forward(self, Y: Tensor, X: Optional[Tensor] = None, W: Optional[Tensor] = None) -> Tensor:
        """
        Compute the MSE aggregation function all outcomes of the objective evaluations Y.

        Args:
            Y (Tensor): A `... x r` tensor of objective evaluations.
            X (Optional[Tensor]): A `n x q x d` tensor of data points at which the objectives were evaluated.
            W (Optional[Tensor]): A `n x q x w` tensor of objective function parameters for each data point.

        Returns:
            Tensor: A `...`-dim tensor of the generality metric for each data point.
        """
        return -torch.sum((Y - self.maximum)**2, dim=-1) / Y.shape[-1]
