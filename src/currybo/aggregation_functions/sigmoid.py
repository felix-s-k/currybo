from typing import Optional
from torch import Tensor
import torch

from .base import BaseAggregation


class Sigmoid(BaseAggregation):
    """
    The counts above threshold aggregation function is defined as the number of outcomes of the objectives evaluated over \
    the set of solutions X that are above a certain threshold. This is approximated by a steep sigmoid.

    Args:
        threshold (float): The threshold above which to count the outcomes.
        k (Optional[float]): exponential factor that defines steepness of sigmoid. \
            If the standard value is used, the sigmoid approximates a heaviside function (fraction above threshold).
    """

    differentiable = True

    def __init__(self, threshold: float, k: Optional[float] = 300.0) -> None:
        self.threshold = threshold
        self.k = k
        if k > 650:
            raise ValueError("This exponential factor will likely cause numerical errors. Please reduce it.")

    def forward(self, Y: Tensor, X: Optional[Tensor] = None, W: Optional[Tensor] = None) -> Tensor:
        """
        Compute the sigmoid aggregation function all outcomes of the objective evaluations Y.

        Args:
            Y (Tensor): A `... x r` tensor of objective evaluations.
            X (Optional[Tensor]): A `n x q x d` tensor of data points at which the objectives were evaluated.
            W (Optional[Tensor]): A `n x q x w` tensor of objective function parameters for each data point.

        Returns:
            Tensor: A `...`-dim tensor of the generality metric for each data point.
        """
        return torch.sum(1 / (1 + torch.exp(-self.k * (Y - self.threshold))), dim=-1) / Y.shape[-1]
