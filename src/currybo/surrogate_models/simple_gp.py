from typing import Type, Optional

import torch

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.fit import fit_gpytorch_mll
from botorch.optim.fit import (
    fit_gpytorch_mll_torch, fit_gpytorch_mll_scipy
)
from botorch.exceptions.errors import ModelFittingError

from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.constraints import Positive
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.mlls import ExactMarginalLogLikelihood

from currybo.surrogate_models.base import BaseSurrogate


class SimpleGP(BaseSurrogate, SingleTaskGP):
    """
    A simple GP model that uses a single kernel to model the entire input parameter space, i.e. input features and objective function parameters.

    The model is trained on a `n x (d + w)` tensor, where d is the  number of features and w is the number of objective function parameters.

    Args:
        train_X (torch.Tensor): A `n x d` tensor of training features
        train_W (torch.Tensor): A `n x w` tensor of objective function parameters for each training data point
        train_Y (torch.Tensor): A `n x t` tensor of training observations.
        kernel (Kernel): A kernel to use in the model.
        likelihood (Likelihood): A likelihood to use for the model.
        normalize_inputs (bool): True if the input data should be normalized (using a botorch InputTransform).
        standardize_outcomes (bool): True if the output data should be standardized (using a botorch OutcomeTransform).

    """

    def __init__(
            self,
            train_X: torch.Tensor,
            train_W: torch.Tensor,
            train_Y: torch.Tensor,
            kernel: Type[Kernel],
            likelihood: Type[Likelihood],
            normalize_inputs: bool = True,
            standardize_outcomes: bool = True,
    ):

        train_inputs = torch.cat((train_X, train_W), dim=-1)

        input_transform = Normalize(train_inputs.shape[-1]) if normalize_inputs else None
        outcome_transform = Standardize(m=1) if standardize_outcomes else None

        covar_module = ScaleKernel(
            kernel(
                ard_num_dims=train_inputs.shape[-1],
                lengthscale_constraint=Positive(),
            ),
            outputscale_prior=GammaPrior(1.1, 0.05),
        )

        SingleTaskGP.__init__(
            self,
            train_inputs,
            train_Y,
            likelihood=likelihood(),
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

    def fit(self) -> None:
        """Fit the surrogate model to the training data."""
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        try:
            fit_gpytorch_mll(mll)
        except (RuntimeError, ModelFittingError):
            try:
                fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch, optimizer_kwargs={})
            except (RuntimeError, ModelFittingError):
                fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_scipy, optimizer_kwargs={"method": "Nelder-Mead"})
        self.trained = True

    def posterior(self, X: torch.Tensor, W: Optional[torch.Tensor] = None) -> GPyTorchPosterior:
        """
        Get the posterior distribution at a set of points.

        Args:
            X (torch.Tensor): A `m x d` tensor of points at which to evaluate the posterior.
            W (torch.Tensor): A `m x w` tensor of objective function parameters for each point in `X`.

        Returns:
            A GPyTorchPosterior object representing the posterior distribution at the given points.
        """
        if W is None:
            test_inputs = X
        else:
            test_inputs = torch.cat((X, W), dim=-1)
        return SingleTaskGP.posterior(self, test_inputs)

    def condition_on_observations(
            self, X: torch.Tensor, Y: torch.Tensor, W: Optional[torch.Tensor] = None
    ) -> BaseSurrogate:
        """
        Return a new model that is conditioned on new observed data points (X, W; Y).

        Args:
            X (torch.Tensor): A `n x d` tensor of design points to condition on.
            Y (torch.Tensor): A `n x t` tensor of observed outcomes corresponding to `X`.
            W (torch.Tensor): A `n x w` tensor of objective function parameters for each point in `X`.
        """
        if W is None:
            new_inputs = X
        else:
            new_inputs = torch.cat((X, W), dim=-1)

        return SingleTaskGP.condition_on_observations(self, X=new_inputs, Y=Y)
        # ATTN: Is this function auto-differentiable? It uses a `deepcopy` operation somewhere in the process. Does this
        # ATTN: update all tensors in the computational graph? If not, it might be very tricky to actually use this
        # ATTN: method in any acquisition routine where gradients are beneficial (e.g. acqf optimization).
