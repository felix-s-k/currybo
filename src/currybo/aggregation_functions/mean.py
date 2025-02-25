from typing import Optional
from torch import Tensor

from .base import BaseAggregation


class Mean(BaseAggregation):
    """The mean aggregation function is defined as the mean of the outcomes of the objectives evaluated over the set of solutions X."""

    differentiable = True

    def forward(self, Y: Tensor, X: Optional[Tensor] = None, W: Optional[Tensor] = None) -> Tensor:
        """
        Compute the mean aggregation function all outcomes of the objective evaluations Y.

        Args:
            Y (Tensor): A `... x r` tensor of objective evaluations.
            X (Optional[Tensor]): A `n x q x d` tensor of data points at which the objectives were evaluated.
            W (Optional[Tensor]): A `n x q x w` tensor of objective function parameters for each data point.

        Returns:
            Tensor: A `...`-dim tensor of the generality metric for each data point.
        """
        return Y.mean(dim=-1)
