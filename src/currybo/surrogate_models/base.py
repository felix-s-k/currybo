from __future__ import annotations
from typing import Optional
from abc import ABCMeta, abstractmethod

import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors.torch import TorchPosterior


class BaseSurrogate(metaclass=ABCMeta):
    """
    Abstract base class for a surrogate model that can be used for "general" Bayesian Optimization.

    Args:
        train_X (torch.Tensor): A `n x d` tensor of training features, where `n` is the number of training points and `d` is the
                 number of features.
        train_W (torch.Tensor): A `n x w` tensor of objective function parameters for each training data point, where `n` is the number
                 of training points and `w` is the number of objective function parameters.
        train_Y (torch.Tensor): A `n x t` tensor of training observations, where `n` is the number of training points and `t` is the
                 number of outcomes.
        input_transform (Optional[InputTransform]): An input transform (botorch.models.transforms.input) to apply to the X values.
        outcome_transform (Optional[OutcomeTransform]): An outcome transform (botorch.models.transforms.outcome) to apply to Y values.

    """

    _trained = False

    @abstractmethod
    def __init__(
            self,
            train_X: torch.Tensor,
            train_W: torch.Tensor,
            train_Y: torch.Tensor,
            input_transform: Optional[InputTransform] = None,
            outcome_transform: Optional[OutcomeTransform] = None,
            **kwargs
    ):
        raise NotImplementedError

    @property
    def trained(self) -> bool:
        """Return whether the model is trained."""
        return self._trained

    @trained.setter
    def trained(self, value: bool) -> None:
        self._trained = value

    @abstractmethod
    def fit(self) -> None:
        """Fit the surrogate model to the training data."""
        pass

    @abstractmethod
    def posterior(self, X: torch.Tensor, W: Optional[torch.Tensor] = None) -> TorchPosterior:
        """
        Get the posterior distribution at a set of points.

        Args:
            X (torch.Tensor): A `m x d` tensor of points at which to evaluate the posterior.
            W (torch.Tensor): A `m x w` tensor of objective function parameters for each point in `X`.

        Returns:
            A TorchPosterior object representing the posterior distribution at the given points.
        """
        raise NotImplementedError

    @abstractmethod
    def condition_on_observations(
        self, X: torch.Tensor, Y: torch.Tensor, W: Optional[torch.Tensor] = None
    ) -> BaseSurrogate:
        """
        Update the surrogate model with new observations.

        Args:
            X (torch.Tensor): A `q x d` tensor of design points to condition on.
            Y (torch.Tensor): A `q x t` tensor of observed outcomes corresponding to `X`.
            W (torch.Tensor): A `q x w` tensor of objective function parameters for each point in `X`.

        Returns:
            A new surrogate model conditioned on the new observations.
        """
        raise NotImplementedError
