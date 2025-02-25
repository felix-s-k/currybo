from typing import Optional
from torch import Tensor

from .base import BaseAggregation


class Min(BaseAggregation):
    """The minimum aggregation function is defined as the minimum value of the function values."""

    differentiable = True

    def forward(self, Y: Tensor, X: Optional[Tensor] = None, W: Optional[Tensor] = None) -> Tensor:
        """
        Compute the Min aggregation function all outcomes of the objective evaluations Y.

        Args:
            Y (Tensor): A `... x r` tensor of objective evaluations.
            X (Optional[Tensor]): A `n x q x d` tensor of data points at which the objectives were evaluated.
            W (Optional[Tensor]): A `n x q x w` tensor of objective function parameters for each data point.

        Returns:
            Tensor: A `...`-dim tensor of the generality metric for each data point.
        """
        values, _ = Y.min(dim=-1)
        return values
