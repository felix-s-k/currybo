from abc import ABCMeta, abstractmethod
from typing import Tuple, Type, Optional, Callable

import torch
from torch import Tensor

from botorch.optim import optimize_acqf, optimize_acqf_discrete, optimize_acqf_mixed
from botorch.sampling import MCSampler, SobolQMCNormalSampler

from .utils import tensor_to_fixed_features_list
from .utility_function import BaseMCUtilityFunction, SimpleRegret
from ..aggregation_functions import BaseAggregation, Mean
from ..surrogate_models import BaseSurrogate


class MCAcquisitionFunction(torch.nn.Module):
    """
    Class for a simple a Monte Carlo acquisition function that can be used in combination with a custom utility function
    and aggregation function in the context of "general" Bayesian Optimization.

    The acquisition function is computed by

    1. computing the posterior distribution for each W option,
    2. sampling from the posterior distribution,
    3. aggregating the samples over all model outputs and W options,
    4. computing the utility function values,
    5. reducing the utility function values over the q and num_mc_samples dimension.

    Args:
        model: The surrogate model to use.
        utility: The utility function to use for the acquisition function.
        w_options: A tensor of shape `r x w` of possible objective function parameters, where `r` is the number of
                   objective functions that can be evaluated, and `w` is the number of parameters per objective function.
        aggregation_function: The aggregation metric to use for the acquisition strategy.
        sample_reduction: The reduction function to use for the samples.
        q_reduction: The reduction function to use for the q dimension.
        sampler: The MCSampler to use for sampling from the posterior distribution.
        num_mc_samples: The number of samples to draw from the posterior distribution.
        max_batch_size: The maximum batch size for optimization.
        **kwargs: Additional keyword arguments to pass to the acquisition function.
    """

    def __init__(
            self,
            model: Optional[BaseSurrogate] = None,
            utility: BaseMCUtilityFunction = None,
            w_options: Tensor = None,
            aggregation_function: BaseAggregation = Mean(),
            sample_reduction: Callable = torch.mean,
            q_reduction: Callable = torch.amax,
            sampler: MCSampler = None,
            num_mc_samples: int = 3,
            max_batch_size: int = 1024,
            **kwargs
    ):

        super().__init__()

        self.model = model
        self.utility = utility

        self.aggregation_function = aggregation_function

        if sampler:
            self.sampler = sampler
        else:
            self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_mc_samples]))

        self.sample_reduction = sample_reduction
        self.q_reduction = q_reduction

        self.max_batch_size = max_batch_size

        if w_options is None:
            raise ValueError("W Options have to be defined")
        self.w_options = w_options

        for key, value in kwargs.items():
            setattr(self, key, value)

    def optimize(
            self,
            q: int = 1,
            bounds: Optional[Tensor] = None,
            options: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Optimize the acquisition function over the input feature space X.

        Args:
            q: The number of points to return.
            bounds: The bounds of the input feature space X (if continuous parameters exist)
            options: The discrete options of the input feature space X (if discrete parameters exist).

        Returns:
            Tensor: The next X to evaluate (shape: `q x d`).
            Tensor: The acquisition function value at the next X (shape: `q`). # TODO: Is the shape correct?
        """
        if bounds is None and options is None:
            raise ValueError("Either Bounds or Options have to be defined!")

        if options is None:  # aka continuous parameter optimization
            next_x, acqf_values = optimize_acqf(
                acq_function=self,
                bounds=bounds,
                q=q,
                num_restarts=10,
                raw_samples=512
            )  # q x d
        elif bounds is None:  # aka discrete parameter optimization
            next_x, acqf_values = optimize_acqf_discrete(
                acq_function=self,
                q=q,
                choices=options,
                max_batch_size=self.max_batch_size
            )
        else:  # aka mixed parameter optimization
            fixed_feature_list = tensor_to_fixed_features_list(
                options,
                bounds.shape[1] - self.x_options.shape[1]   # index at which the discrete options start
            )  # assumes fixed structure of concatenated inputs: first continuous and then discrete parameters

            next_x, acqf_values = optimize_acqf_mixed(
                acq_function=self,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                fixed_features_list=fixed_feature_list
            )

        return next_x, acqf_values

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the acquisition function value for a given input X by
        
        1) computing the posterior distribution for each W option,
        2) sampling from the posterior distribution,
        3) aggregating the samples over all model outputs and W options,
        4) computing the utility function values,
        5) reducing the utility function values over the q and num_mc_samples dimension.

        Args:
            X: A `n x q x d` tensor of design points to evaluate.

        Returns:
            A `n` tensor of acquisition function values.
        """
        if self.model is None or self.utility is None:
            raise ValueError("The model and utility function must be set before calling the acquisition function!")

        posteriors = [self.model.posterior(X, w.repeat(X.shape[0], 1).unsqueeze(1)) for w in self.w_options]
        mc_samples = torch.stack([self.sampler(posterior) for posterior in posteriors], dim=-1)
        # num_mc_samples x n x q x m x r
        #   - `m` is the number of model outputs
        #   - `r` is the number of W options

        aggregated_samples = self.aggregation_function(mc_samples)  # num_mc_samples x n x q
        utility_values = self.utility(aggregated_samples)  # num_mc_samples x n x q

        # reduce over the q and num_mc_samples dimension
        return self.sample_reduction(self.q_reduction(utility_values, dim=-1), dim=0)  # n


class FantasyMCAcquisitionFunction(MCAcquisitionFunction):
    """
    Class for a Monte Carlo acquisition function that can be used as the inner acquisition function for a sample-based
    multi-step lookahead acquisition function. This class is used to compute the utility function values for samples
    from the posterior distribution of a set of fantasy models.

    The acquisition function is computed by
    
    1) for each fantasy model, computing the posterior distribution for each W option,
    2) for each fantasy model, sampling from the posterior distribution,
    3) reducing the samples over all fantasy models (i.e. over the 'num_outer_samples' dimension),
    4) aggregating the samples over all model outputs and W options,
    5) computing the utility function values,
    6) reducing the utility function values over the q and 'num_inner_samples' dimension.

    Args:
        fantasy_models: A list of fantasy models to use.
        utility: The utility function to use for the acquisition function.
        w_options: A tensor of shape `r x w` of possible objective function parameters, where `r` is the number of
                   objective functions that can be evaluated, and `w` is the number of parameters per objective function.
        aggregation_function: The aggregation metric to use for the acquisition strategy.
        sample_reduction: The reduction function to use for the samples.
        q_reduction: The reduction function to use for the q dimension.
        sampler: The MCSampler to use for sampling from the posterior distribution.
        num_mc_samples: The number of samples to draw from the posterior distribution.
        max_batch_size: The maximum batch size for optimization.
        **kwargs: Additional keyword arguments to pass to the acquisition function.
    """

    def __init__(
            self,
            fantasy_models: list[BaseSurrogate] = None,
            utility: BaseMCUtilityFunction = None,
            w_options: Tensor = None,
            aggregation_function: BaseAggregation = Mean(),
            sample_reduction: Callable = torch.mean,
            q_reduction: Callable = torch.amax,
            sampler: MCSampler = None,
            num_mc_samples: int = 3,
            max_batch_size: int = 1024,
            **kwargs
    ):

        super().__init__(
            model=None,
            utility=utility,
            w_options=w_options,
            aggregation_function=aggregation_function,
            sample_reduction=sample_reduction,
            q_reduction=q_reduction,
            sampler=sampler,
            num_mc_samples=num_mc_samples,
            max_batch_size=max_batch_size,
            **kwargs
        )

        self.models = fantasy_models

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the acquisition function value for a given input X by

        1) for each fantasy model, computing the posterior distribution for each W option,
        2) for each fantasy model, sampling from the posterior distribution,
        3) reducing the samples over all fantasy models (i.e. over the 'num_outer_samples' dimension),
        4) aggregating the samples over all model outputs and W options,
        5) computing the utility function values,
        6) reducing the utility function values over the q and 'num_inner_samples' dimension.
        """
        samples = []
        for model in self.models:
            posteriors = [model.posterior(X, w.repeat(X.shape[0], 1).unsqueeze(1)) for w in self.w_options]
            samples_ = torch.stack([self.sampler(posterior) for posterior in posteriors], dim=-1)  # `num_inner_samples` x `batch_size x q x m x r`
            samples.append(samples_)
        samples = torch.stack(samples, dim=0)  # `num_outer_samples x num_inner_samples x batch_size x q x m x r`

        aggregated_samples = self.aggregation_function(samples) # `num_outer_samples x num_inner_samples x batch_size x q`, recuces over m and r
        samples = self.sample_reduction(aggregated_samples, dim = 1) # `num_outer_samples x batch_size x q`, reduces over `num_inner_samples`
        utility_values = self.utility(samples) # `num_outer_samples x batch_size x q`, gets utility values
        return self.sample_reduction(self.q_reduction(utility_values, dim=-1), dim=0)  # `n_eval`

        #samples = self.sample_reduction(samples, dim=0)  # `num_inner_samples x n_eval x q x m x r`, averages over `num_outer_samples` = `num_mc_samples`
        #print("NEW SAMPLES: ", samples.shape)
        #aggregated_samples = self.aggregation_function(samples)  # `num_inner_samples x n_eval x q`, reduces over m and r in aggregation function
        #print("AGGREGATED_SAMPLES: ", aggregated_samples.shape, aggregated_samples)
        #utility_values = self.utility(aggregated_samples)  # `num_inner_samples x n_eval x q`, gets utility values
        #print("UTILITY_VALUES: ", utility_values.shape, utility_values)
        #print("FINAL: ", self.sample_reduction(self.q_reduction(utility_values, dim=-1), dim=0).shape, self.sample_reduction(self.q_reduction(utility_values, dim=-1), dim=0))
        #return self.sample_reduction(self.q_reduction(utility_values, dim=-1), dim=0)  # `n_eval`


class BaseMCAcquisitionStrategy(metaclass=ABCMeta):
    """
    Abstract base class for a Monte Carlo acquisition function that can be used in the context of "general" Bayesian
    Optimization.

    Args:
        x_bounds: The bounds of the input feature space X (if continuous parameters exist).
        x_options: The discrete options of the input feature space X (if discrete parameters exist).
        w_options: A tensor of shape `r x w` of possible objective function parameters, where `r` is the number of
                   objective functions that can be evaluated, and `w` is the number of parameters per objective function.
        aggregation_function: The aggregation metric to use for the acquisition strategy.
        sample_reduction: The reduction function to use for the samples.
        q_reduction: The reduction function to use for the q dimension.
        sampler_type: The MCSampler to use for sampling from the posterior distribution.
        num_mc_samples: The number of samples to draw from the posterior distribution.
        maximization: Whether to maximize the acquisition function.
        **kwargs: Additional keyword arguments to pass to the acquisition strategy.
    """
    def __init__(
            self,
            x_bounds: Optional[Tensor] = None,
            x_options: Optional[Tensor] = None,
            w_options: Tensor = None,
            aggregation_function: BaseAggregation = Mean(),
            sample_reduction: Callable = torch.mean,
            q_reduction: Callable = torch.amax,
            sampler_type: Type[MCSampler] = None,
            num_mc_samples: int = 3,
            maximization: bool = True,
            **kwargs
    ):

        self.aggregation_function = aggregation_function

        self.sample_reduction = sample_reduction
        self.q_reduction = q_reduction

        self.sampler_type = sampler_type
        self.num_mc_samples = num_mc_samples

        if x_bounds is None and x_options is None:
            raise ValueError("Either Bounds or Options have to be defined!")
        if w_options is None:
            raise ValueError("W Options have to be defined")
        self.x_bounds = x_bounds
        self.x_options = x_options
        self.w_options = w_options

        self.maximization = maximization

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def continuous_x(self) -> bool:
        return self.x_bounds is not None

    @property
    def discrete_x(self) -> bool:
        return self.x_bounds is None

    @abstractmethod
    def get_recommendation(
            self,
            model: BaseSurrogate,
            q: int = 1,
            **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Method to apply the acquisition strategy for recommending the next point(s) to evaluate.

        Args:
            model: The surrogate model to use.
            q: The number of points to return.
            **kwargs: Additional keyword arguments to pass to the acquisition strategy.

        Returns:
            Tensor: The next X to evaluate (shape: `q x d`).
            Tensor: The index of the next W to evaluate (shape: `q`).
        """
        raise NotImplementedError

    def get_final_recommendation(
            self,
            model: BaseSurrogate,
            utility: Type[BaseMCUtilityFunction] = SimpleRegret,
            utility_kwargs: Optional[dict] = None
    ) -> Tensor:
        """
        Identifies the final X to be recommended as the optimum by optimizing the given utility function (defaults to
        a fully greedy utility) over the input feature space X.

        Args:
            model: Trained surrogate model.
            utility: The utility function to use for the optimization step.
            utility_kwargs: Keyword arguments to pass to the utility function.

        Returns:
            Tensor: The final X to be recommended as the optimum (shape: `1 x d`).
        """
        if not model.trained:
            raise ValueError("The surrogate model must be trained before calling the acquisition strategy.")

        if self.sampler_type is not None:
            sampler = self.sampler_type(sample_shape=torch.Size([512]))
        else:
            sampler = None

        acqf = MCAcquisitionFunction(
            model=model,
            utility=SimpleRegret(maximize=self.maximization),
            aggregation_function=self.aggregation_function,
            sample_reduction=self.sample_reduction,
            q_reduction=self.q_reduction,
            w_options=self.w_options,
            sampler=sampler,
            num_mc_samples=3,
        )

        return acqf.optimize(q=1, bounds=self.x_bounds, options=self.x_options)[0]
