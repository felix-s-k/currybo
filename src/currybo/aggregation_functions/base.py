from abc import ABCMeta, abstractmethod
from typing import Optional
from torch import Tensor


class BaseAggregation(metaclass=ABCMeta):
    """
    Abstract base class for a metric that quantifies the generality of a given set of some solution X that was evaluated \
    over a set of objectives Y (each one parametrized by W).

    Attributes:
        differentiable: A boolean flag indicating whether the metric is differentiable (important for selecting the
                        optimizer for the acquisition function).
    """

    differentiable: bool = True  # TODO: Check if we still need this for MC acquisition functions

    def _align_y_shape(self, Y: Tensor) -> Tensor:
        """
        Aligns the shape of the tensor of objective evaluations Y.

        Args:
            Y (Tensor): A tensor of objective evaluations.
                    - `num_mc_samples` is the number of Monte Carlo samples.
                    - `n` is the number of data points at which the objectives were evaluated.
                    - `q` is the number of locations to evaluate jointly.
                    - `m` is the number of model outputs (usually 1), otherwise the mean cannot be computed.
                    - `r` is the number of objective functions evaluated.
               Possible shapes:
                    1) `num_mc_samples x n x q x m x r`
                    2) `n x r`

        Returns:
            Tensor: A homogenized tensor of objective evaluations.
                    1) `num_mc_samples x n x q x r`
                    2) `n x r`
        """
        if Y.dim() == 6:
            if Y.shape[3] != 1:
                raise ValueError("The `num_outputs` dimension of the tensor must be 1.")
            return Y.squeeze(dim=3)
        elif Y.dim() == 5:
            if Y.shape[3] != 1:
                raise ValueError("The `num_outputs` dimension of the tensor must be 1.")
            return Y.squeeze(dim=3)
        elif Y.dim() == 4:
            raise NotImplementedError("No support for 4-dimensional tensors currently.")
        elif Y.dim() == 3:
            raise NotImplementedError("No support for 3-dimensional tensors currently.")
        elif Y.dim() == 2:
            return Y
        else:
            raise ValueError("The tensor must have 2, 3, 4, 5, or 6 dimensions.")

    def __call__(self, Y: Tensor, X: Optional[Tensor] = None, W: Optional[Tensor] = None) -> Tensor:
        """
        Compute the generality metric for all outcomes of the objective evaluations Y.

        Args:
            Y (Tensor): A `... x r` tensor of objective evaluations. Possible shapes are defined in the `_align_y_shape` method.
            X (Optional[Tensor]): A `n x q x d` tensor of data points at which the objectives were evaluated.
            W (Optional[Tensor]): A `n x q x w` tensor of objective function parameters for each data point.

        Returns:
            A `...`-dim tensor of the generality metric for each data point.
        """
        Y = self._align_y_shape(Y)
        return self.forward(Y, X, W)

    @abstractmethod
    def forward(self, Y: Tensor, X: Optional[Tensor] = None, W: Optional[Tensor] = None) -> Tensor:
        """
        Compute the generality metric for all outcomes of the objective evaluations Y.

        Args:
            Y (Tensor): A `... x r` tensor of objective evaluations.

            X (Optional[Tensor]): A `n x q x d` tensor of data points at which the objectives were evaluated.
            W (Optional[Tensor]): A `n x q x w` tensor of objective function parameters for each data point.

        Returns:
            Tensor: A `...`-dim tensor of the generality metric for each data point, squeezed along the last dimension.
        """
        raise NotImplementedError("The `forward` method must be implemented in the derived class.")
