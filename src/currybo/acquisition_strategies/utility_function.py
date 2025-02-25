from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor
import numpy as np
import random


class BaseMCUtilityFunction(torch.nn.Module, metaclass=ABCMeta):
    """Abstract base class for "Conventional" Bayesian Optimization utility functions. This is a minimal implementation \
    of the utility function concept, computing the utility function values for samples from the posterior distribution."""

    deterministic: bool = True   # TODO: Check if this is still needed with the MCAcquisitionStrategy

    def __init__(self, maximize: bool = True, **kwargs):

        self.maximize = maximize

        for key, value in kwargs.items():
            setattr(self, key, value)

        super().__init__()

    @abstractmethod
    def forward(self, samples: Tensor) -> Tensor:
        """
        Compute the utility function value for a given posterior distribution.

        Args:
            samples: A `num_samples x ...' tensor of samples from the posterior distribution.

        Returns:
            A `num_samples x ...' tensor of the utility function values.
        """
        raise NotImplementedError


class Random(BaseMCUtilityFunction):
    """Minimal implementation of the Random baseline as a utility function."""

    def __init__(self, random_seed: int = 0, maximize: bool = True):
        super().__init__(maximize=maximize, random_seed=random_seed)

    def forward(self, samples: Tensor) -> Tensor:
        """Compute a random number as the utility function value. Includes the dummy term `0.0 * mean`to ensure that the \
        output tensor is part of the computational graph used for gradient-based optimization."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        return 0.0 * samples.mean(dim=0, keepdim=True) + torch.rand_like(samples)


class SimpleRegret(BaseMCUtilityFunction):
    """Minimal implementation of the "Simple Regret" / "Simple Reward" utility."""

    def forward(self, samples: Tensor) -> Tensor:
        """Return the function value itself as the utility function value."""
        if self.maximize:
            return samples
        else:
            return -samples


class UncertaintyUtility(BaseMCUtilityFunction):
    """Minimal implementation of a fully uncertainty-driven utility (corresponding to the posterior variance as the acquisition function)."""

    def forward(self, samples: Tensor) -> Tensor:
        """Compute the utility function value based on the deviation of the samples from the mean."""
        return torch.abs(samples - samples.mean(dim=0, keepdim=True))


class QuantileUtility(BaseMCUtilityFunction):
    """
    Minimal implementation of a quantile-based utility function, which corresponds to the Upper Confidence Bound \
    acquisition function.

    Args:
        beta: A float representing the trade-off parameter between the mean and the standard deviation.
        maximize: A boolean flag indicating whether the optimization problem is a maximization problem.
    """

    def __init__(self, beta: float = 0.5, maximize: bool = True, **kwargs):
        super().__init__(maximize=maximize, beta=torch.tensor(beta))

    def forward(self, samples: Tensor) -> Tensor:
        """Compute the utility function value based on the deviation of the samples from the mean."""
        mean = samples.mean(dim=0, keepdim=True)

        if self.maximize:
            return mean + self.beta * torch.abs(samples - mean)
        else:
            return -mean + self.beta * torch.abs(samples - mean)

        # ATTN: Double-check the signs to make sure that maximization / minimization is handled correctly.


class QuantitativeImprovement(BaseMCUtilityFunction):
    """
    Minimal implementation of the "Quantitative Improvement" utility function, which corresponds to the Expected \
    Improvement acquisition function.

    Args:
        best_f: The best function value observed so far.
        maximize: A boolean flag indicating whether the optimization problem is a maximization problem.
    """

    def __init__(self, maximize: bool = True, **kwargs):
        best_f = kwargs.get("best_f", None)
        super().__init__(maximize=maximize, best_f=torch.tensor(best_f))

    def forward(self, samples: Tensor) -> Tensor:
        """
        Computes the utility function value based on the deviation of the samples from the mean.
        """
        if self.maximize:
            return (samples - self.best_f).clamp(min=0)
        else:
            return (self.best_f - samples).clamp(min=0)

class QualitativeImprovement(BaseMCUtilityFunction):
    """Minimal implementation of the "Qualitative Improvement" utility function, which corresponds to the Probability of Improvement acquisition function."""

    def __init__(self, tau: float = 0.01, maximize: bool = True, **kwargs):
        best_f = kwargs.get("best_f", best_f)
        super().__init__(maximize=maximize, best_f=torch.tensor(best_f), tau=torch.tensor(tau))

    def forward(self, samples: Tensor) -> Tensor:
        """
        Computes the utility function value based on the deviation of the samples from the mean.
        """
        if self.maximize:
            return torch.sigmoid((samples - self.best_f) / self.tau)
        else:
            return torch.sigmoid((self.best_f - samples) / self.tau)