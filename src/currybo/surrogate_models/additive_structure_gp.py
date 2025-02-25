from typing import Type, Optional, List, Dict, Any

import torch

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.fit import fit_gpytorch_mll

from gpytorch.kernels.kernel import Kernel
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood

from currybo.surrogate_models.base import BaseSurrogate


class AdditiveStructureGP(BaseSurrogate, SingleTaskGP):
    """
    A GP model that uses a sum of multiple Kernels to model the features and the objective function parameters separately.

    Args:
        train_X (torch.Tensor): A `n x d` tensor of training features
        train_W (torch.Tensor): A `n x w` tensor of objective function parameters for each training data point
        train_Y (torch.Tensor): A `n x t` tensor of training observations.
        x_kernel (Optional[Kernel]): A kernel to use for the features.
        w_kernel (Kernel): A kernel to use for the objective function parameters.
        likelihood (Likelihood): A likelihood to use for the model.
        normalize_inputs (bool): True if the input data should be normalized (using a botorch InputTransform).
        standardize_outcomes (bool): True if the output data should be standardized (using a botorch OutcomeTransform).

    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_W: torch.Tensor,
        train_Y: torch.Tensor,
        x_kernels: Optional[Type[Kernel]],
        x_kernel_kwargs: List[Dict[str, Any]],
        w_kernels: Type[Kernel],
        w_kernel_kwargs: List[Dict[str, Any]],
        likelihood: Type[Likelihood],
        normalize_inputs: Optional[bool] = True,
        standardize_outcomes: Optional[bool] = True
    ):

        train_inputs = torch.cat((train_X, train_W), dim=-1)

        input_transform = Normalize(train_inputs.shape[-1]) if normalize_inputs else None
        outcome_transform = Standardize(m=1) if standardize_outcomes else None

        # Preliminary transformation of the input data to set the _aug_batch_shape attribute
        with torch.no_grad():
            train_X_transformed = self.transform_inputs(train_inputs, input_transform=input_transform)
        self._set_dimensions(train_X=train_X_transformed, train_Y=train_Y)

        x_kernels_list = []

        for i, kernel in enumerate(x_kernels):

            x_kernels_list.append(kernel(**x_kernel_kwargs[i]))

        w_kernels_list = []

        for i, kernel in enumerate(w_kernels):

            w_kernels_list.append(kernel(**w_kernel_kwargs[i]))

        if x_kernels_list:
            x_kernel = x_kernels_list[0]
            for kernel in x_kernels_list[1:]:
                x_kernel += kernel

        if w_kernels_list:
            w_kernel = w_kernels_list[0]
            for kernel in w_kernels_list[1:]:
                w_kernel += kernel

        covar_module = ScaleKernel(
            x_kernel + w_kernel,
            batch_shape=self._aug_batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )

        SingleTaskGP.__init__(
            self,
            train_X=train_inputs,
            train_Y=train_Y,
            likelihood=likelihood(),
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform
        )

        self._subset_batch_dict = {
            "mean_module.raw_constant": -1,
            "covar_module.raw_outputscale": -1,
        }

    def fit(self) -> None:
        """Fit the surrogate model to the training data."""
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        fit_gpytorch_mll(mll)
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
