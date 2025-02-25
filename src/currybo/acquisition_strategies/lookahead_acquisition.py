from typing import Tuple, Type, Optional

import torch
from torch import Tensor

from botorch.sampling import MCSampler, SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import SampleReductionProtocol

from currybo.acquisition_strategies.base import BaseMCAcquisitionStrategy, FantasyMCAcquisitionFunction, MCAcquisitionFunction
from currybo.acquisition_strategies.utils import create_all_permutations
from currybo.acquisition_strategies.utility_function import BaseMCUtilityFunction, UncertaintyUtility, QuantitativeImprovement, SimpleRegret
from currybo.surrogate_models import BaseSurrogate
from currybo.aggregation_functions import BaseAggregation, Mean


class JointLookaheadAcquisition(BaseMCAcquisitionStrategy, torch.nn.Module):
    """
    Two-step-lookahead acquisition strategy that jointly picks the best X and W combination to be evaluated next.

    This strategy is based on the algorithm described by Angello et al. (Science 2022, https://doi.org/10.1126/science.adc8743).

    Args:
        x_bounds: A tensor (2 x d) of bounds for each dimension of the input space.
        x_options: A tensor (num_choices x m), where `m` is the representation of chemicals, of options for discrete
                   optimization.
        w_options: A tensor (r x w) of possible objective function parameters, where `r` is the number of objective
                   functions that can be evaluated, and `w` is the number of parameters per objective function.
        aggregation_function: The aggregation metric to use for the acquisition strategy.
        sample_reduction: The reduction function to use for aggregating the expected utility values over the Monte Carlo
                          samples.
        q_reduction: The reduction function to use for aggregating the expected utility values over the W options.
        sampler_type: The type of sampler to use for drawing Monte Carlo samples.
        num_mc_samples: The number of Monte Carlo samples to draw from the posterior distribution.
        num_inner_mc_samples: The number of Monte Carlo samples to draw for the inner optimization loop.
        maximization: If True, the objective should be maximized. Otherwise, it should be minimized.
        utility_type: The utility function to use for the optimization step.
        utility_kwargs: Keyword arguments to pass to the utility function.
        inference_batch_size: The batch size to use for the inference step.
    """
    def __init__(
            self,
            x_bounds: Optional[Tensor] = None,
            x_options: Optional[Tensor] = None,
            w_options: Tensor = None,
            aggregation_function: BaseAggregation = Mean(),
            sample_reduction: SampleReductionProtocol = torch.mean,
            q_reduction: SampleReductionProtocol = torch.amax,
            sampler_type: Type[MCSampler] = SobolQMCNormalSampler,
            num_mc_samples: int = 3,
            num_inner_mc_samples: int = 1,
            maximization: bool = True,
            utility_type: Type[BaseMCUtilityFunction] = QuantitativeImprovement,
            utility_kwargs: dict = None,
            inference_batch_size: int = 512,
    ):

        torch.nn.Module.__init__(self)

        BaseMCAcquisitionStrategy.__init__(
            self,
            x_bounds=x_bounds,
            x_options=x_options,
            w_options=w_options,
            aggregation_function=aggregation_function,
            sample_reduction=sample_reduction,
            q_reduction=q_reduction,
            sampler_type=sampler_type,
            num_mc_samples=num_mc_samples,
            num_inner_mc_samples=num_inner_mc_samples,
            maximization=maximization,
            _utility_type=utility_type,
            _utility_kwargs=utility_kwargs or {},
            _inference_batch_size=inference_batch_size
        )

        self.model: Optional[BaseSurrogate] = None
        self.utility: Optional[BaseMCUtilityFunction] = None

    def get_recommendation(
            self,
            model: BaseSurrogate,
            q: int = 1,
            **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Identify the next X and W to evaluate by maximizing the acquisition function of the aggregation metric over the
        input feature space X.

        Args:
            model: Trained surrogate model.
            q: The number of points to return.

        Returns:
            Tensor: The next X to evaluate (shape: `q x d`).
            Tensor: The index of the next W to evaluate (shape: `q`).
        """
        if not model.trained:
            raise ValueError("The surrogate model must be trained before calling the acquisition strategy.")
        self.model = model

        if self.sampler_type is not None:
            self.sampler = self.sampler_type(sample_shape=torch.Size([self.num_mc_samples]))
            self.inner_sampler = self.sampler_type(sample_shape=torch.Size([self.num_inner_mc_samples]))
        else:
            self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_mc_samples]))
            self.inner_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_inner_mc_samples]))

        # get current best value
        sampler = self.sampler_type(sample_shape=torch.Size([512]))

        greedy_acqf = MCAcquisitionFunction(
            model=model,
            utility=SimpleRegret(maximize=self.maximization, **self._utility_kwargs, **kwargs),
            aggregation_function=self.aggregation_function,
            sample_reduction=self.sample_reduction,
            q_reduction=self.q_reduction,
            w_options=self.w_options,
            sampler=sampler,
            num_mc_samples=self.num_mc_samples,
        )

        best_f, _ = greedy_acqf.optimize(q=1, bounds=self.x_bounds, options=self.x_options)

        self._utility_kwargs["best_f"] = best_f

        self.utility = self._utility_type(**self._utility_kwargs)

        if self.x_bounds is None:  # aka purely discrete parameter optimization

            all_xw, all_xw_idx = create_all_permutations(self.x_options, self.w_options)
            all_acqf_values = torch.cat(
                [self.forward(X_.unsqueeze(1)) for X_ in all_xw.split(self._inference_batch_size)],
                dim=0
            )
            next_x_idx, next_w_idx = all_xw_idx[torch.argmax(all_acqf_values)]
            next_x = self.x_options[next_x_idx].unsqueeze(0)

        else:
            raise NotImplementedError  # We would later need some mixed optimization here...

        return next_x, next_w_idx.unsqueeze(0)

    def forward(self, X: Tensor) -> Tensor:
        """
        Calculates a two-step lookahead acquisition function value for a given X+W input. Note that the argument is
        named `X`, which can be confusing, but needed to be done for consistency with the botorch API. In this
        docstring, the `X` (in quotes) refers to the argument passed to this function (i.e. the X+W combination).

        The value returned by this function effectively answers the following question: If we picked the passed
        `X` (i.e. choose this specific X+W combination) in the first step, what would an updated surrogate model
        predict as the expected utility of the best (i.e. most general) X over all W in the second step?

        Follows the logic of:
            1) Draw `num_mc_samples` samples from the posterior distribution of the model for the passed `X`.
            2) For each sample, and every X+W combination in the passed `X`, condition the model on the fantasized
               observation (`X`, Sample(`X`). Use this `fantasy model` to optimize a one-step lookahead acquisition
               function to find the best X (only the `true` X in this case, aggregated over all W) and the corresponding
               acquisition function value.
            3) Aggregate (i.e., use `sample_reduction` on) the expected utility values over all posterior samples to get
               the final acquisition function value for each of the passed `X`.

        Args:
            X: A `n x q x (d_x + d_w)` tensor of joint x-w design points to evaluate.

        Returns:
            A `n` tensor of acquisition function values.
        """
        posterior = self.model.posterior(X)
        mc_samples = self.sampler(posterior)  # num_mc_samples x n x q x m

        all_acqf_values = []

        i = 0
        for XW_, Y_ in zip(X, mc_samples.permute(1, 0, 2, 3)):  # X_: `q x (d_x + d_w)`, Y_: `num_mc_samples x q x m`
            fantasy_models = [self.model.condition_on_observations(XW_, Y__) for Y__ in Y_]

            fantasy_acqf = FantasyMCAcquisitionFunction(
                fantasy_models=fantasy_models,
                utility=self.utility,
                w_options=self.w_options,
                aggregation_function=self.aggregation_function,
                sample_reduction=self.sample_reduction,
                q_reduction=self.q_reduction,
                sampler=self.inner_sampler,
                max_batch_size=self._inference_batch_size
            )
            _, acqf_value = fantasy_acqf.optimize(q=1, bounds=self.x_bounds, options=self.x_options)
            all_acqf_values.append(acqf_value)

            i += 1

        return torch.stack(all_acqf_values, dim=0)  # `n`


class SequentialLookaheadAcquisition(BaseMCAcquisitionStrategy, torch.nn.Module):
    """
    Sequential two-step-lookahead acquisition strategy that first picks the best X to evaluate (based on a two-step
    lookahead), and then the best W (based on a two-step lookahead).

    This strategy is based on the algorithm described by Angello et al. (Science 2022, https://doi.org/10.1126/science.adc8743).

    Args:
        x_bounds: A tensor (2 x d) of bounds for each dimension of the input space.
        x_options: A tensor (num_choices x m), where `m` is the representation of chemicals, of options for discrete
                   optimization.
        w_options: A tensor (r x w) of possible objective function parameters, where `r` is the number of objective
                   functions that can be evaluated, and `w` is the number of parameters per objective function.
        aggregation_function: The aggregation metric to use for the acquisition strategy.
        sample_reduction: The reduction function to use for aggregating the expected utility values over the Monte Carlo
                          samples.
        q_reduction: The reduction function to use for aggregating the expected utility values over the W options.
        sampler_type: The type of sampler to use for drawing Monte Carlo samples.
        num_mc_samples: The number of Monte Carlo samples to draw from the posterior distribution.
        num_inner_mc_samples: The number of Monte Carlo samples to draw for the inner optimization loop.
        maximization: If True, the objective should be maximized. Otherwise, it should be minimized.
        x_utility: The utility function to use for the X optimization step.
        x_utility_kwargs: Keyword arguments to pass to the X utility function.
        w_utility: The utility function to use for the W optimization step.
        w_utility_kwargs: Keyword arguments to pass to the W utility function.
        inference_batch_size: The batch size to use for the inference step.
    """
    def __init__(
            self,
            x_bounds: Optional[Tensor] = None,
            x_options: Optional[Tensor] = None,
            w_options: Tensor = None,
            aggregation_function: BaseAggregation = Mean(),
            sample_reduction: SampleReductionProtocol = torch.mean,
            q_reduction: SampleReductionProtocol = torch.amax,
            sampler_type: Type[MCSampler] = SobolQMCNormalSampler,
            num_mc_samples: int = 3,
            num_inner_mc_samples: int = 1,
            maximization: bool = True,
            x_utility: Type[BaseMCUtilityFunction] = QuantitativeImprovement,
            x_utility_kwargs: dict = None,
            w_utility: Type[BaseMCUtilityFunction] = UncertaintyUtility,
            w_utility_kwargs: dict = None,
            inference_batch_size: int = 512,
    ):

        torch.nn.Module.__init__(self)

        BaseMCAcquisitionStrategy.__init__(
            self,
            x_bounds=x_bounds,
            x_options=x_options,
            w_options=w_options,
            aggregation_function=aggregation_function,
            sample_reduction=sample_reduction,
            q_reduction=q_reduction,
            sampler_type=sampler_type,
            num_mc_samples=num_mc_samples,
            num_inner_mc_samples=num_inner_mc_samples,
            maximization=maximization,
            _x_utility_type=x_utility,
            _x_utility_kwargs=x_utility_kwargs or {},
            _w_utility_type=w_utility,
            _w_utility_kwargs=w_utility_kwargs or {},
            _inference_batch_size=inference_batch_size
        )

        self.model: Optional[BaseSurrogate] = None
        self.x_utility: Optional[BaseMCUtilityFunction] = None
        self.w_utility: Optional[BaseMCUtilityFunction] = None

    def get_recommendation(
            self,
            model: BaseSurrogate,
            q: int = 1,
            **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Identify the next X and W to evaluate by first maximizing the acquisition function of the aggregation metric
        (over the input feature space X), and then maximizing the acquisition function over all options of W.

        Args:
            model: Trained surrogate model.
            q: The number of points to return.

        Returns:
            Tensor: The next X to evaluate (shape: `q x d`).
            Tensor: The index of the next W to evaluate (shape: `q`).
        """
        if not model.trained:
            raise ValueError("The surrogate model must be trained before calling the acquisition strategy.")
        self.model = model

        if self.sampler_type is not None:
            self.sampler = self.sampler_type(sample_shape=torch.Size([self.num_mc_samples]))
            self.inner_sampler = self.sampler_type(sample_shape=torch.Size([self.num_inner_mc_samples]))
        else:
            self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_mc_samples]))
            self.inner_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_inner_mc_samples]))

        # get current best value
        sampler = self.sampler_type(sample_shape=torch.Size([512]))

        greedy_acqf = MCAcquisitionFunction(
            model=model,
            utility=SimpleRegret(maximize=self.maximization, **self._x_utility_kwargs, **kwargs),
            aggregation_function=self.aggregation_function,
            sample_reduction=self.sample_reduction,
            q_reduction=self.q_reduction,
            w_options=self.w_options,
            sampler=sampler,
            num_mc_samples=self.num_mc_samples,
        )

        _, best_f = greedy_acqf.optimize(q=1, bounds=self.x_bounds, options=self.x_options)
        best_f = best_f.item()

        self._x_utility_kwargs["best_f"] = best_f

        self.x_utility = self._x_utility_type(**self._x_utility_kwargs)
        self.w_utility = self._w_utility_type(**self._w_utility_kwargs)

        # First, optimize over X
        if self.x_bounds is None:  # aka purely discrete parameter optimization
            all_acqf_values = torch.cat(
                [self.forward(X_.unsqueeze(1)) for X_ in self.x_options.split(self._inference_batch_size)],
                dim=0
            )
            next_x = self.x_options[torch.argmax(all_acqf_values)].unsqueeze(0)

        else:
            raise NotImplementedError  # No continuous optimization yet...

        # Second, optimize over W
        all_xw, all_xw_idx = create_all_permutations(next_x, self.w_options)
        all_acqf_values = torch.cat(
            [self._forward_w(X_.unsqueeze(1)) for X_ in all_xw.split(self._inference_batch_size)],
        )
        _, next_w_idx = all_xw_idx[torch.argmax(all_acqf_values)]

        return next_x, next_w_idx.unsqueeze(0)

    def forward(self, X: Tensor) -> Tensor:
        """
        Performs a two-step lookahead for computing the acquisition function values of a given set of x locations.

        The value returned by this function effectively answers the following question: If we picked the passed
        X first, and choose the W following the set decision policy, what would an updated surrogate model predict
        as the expected utility of the best (i.e. most general) X over all W in the second step?

        Follows the logic of:
            1) For each option of w, draw `num_mc_samples` samples from the posterior distribution of the model at the
               passed locations X.
            2) Calculate the values of the 'w utility' function based on the posterior samples, and aggregate them over
               all samples to obtain the 'w acquisition function' values, which can be used to pick the hypothetical
               next w for each location in X.
            3) For each posterior sample, and every location in X, condition the model on the fantasized observation at
               the passed X and the hypothetical W from 2). Use this `fantasy model` to optimize a one-step lookahead
               acquisition function to find the best X (aggregated over all W) and the corresponding acquisition
               function value.
            4) Aggregate (i.e., use `sample_reduction` on) the expected utility values over all posterior samples to get
               the final acquisition function value for each of the passed `X`.

        Tensor dimensions used in the code:
            - `n`: Number of locations in X to be evaluated.
            - `q`: Number of q-optimal locations to be returned (currently only supports q=1).
            - `d_x`: Dimensionality of the x features.
            - `d_w`: Dimensionality of the w features.
            - `m`: Number of outputs of the model (currently only supports m=1).
            - `r`: Number of discrete options for w.
            - `num_mc_samples`: Number of Monte Carlo samples to draw from the posterior distribution.


        Args:
            X: A `n x q x d_x` tensor of x design points to evaluate.

        Returns:
            A `n` tensor of acquisition function values.
        """
        if self.model is None or self.x_utility is None or self.w_utility is None:
            raise ValueError("The model and utility functions must be set before calling the acquisition function!")

        posteriors = [self.model.posterior(X, w.repeat(X.shape[0], 1).unsqueeze(1)) for w in self.w_options]
        mc_samples = torch.stack([self.sampler(posterior) for posterior in posteriors], dim=-1)  # num_mc_samples x n x q x m x r

        if mc_samples.shape[3] != 1:
            raise ValueError("The `num_outputs` dimension of the tensor must be 1 currently...")
        mc_samples = mc_samples.squeeze(dim=3)  # num_mc_samples x n x q x r

        w_utility_values = self.w_utility(mc_samples)  # num_mc_samples x n x q x r
        w_acqf_values = self.sample_reduction(self.q_reduction(w_utility_values, dim=-2), dim=0)  # n x r
        next_w_idx = torch.argmax(w_acqf_values, dim=-1, keepdim=True)  # n x 1,

        all_acqf_values = []

        for X_, Y_, W_idx_ in zip(X, mc_samples.permute(1, 0, 2, 3), next_w_idx):  # X_: `q x d_x`, Y_: `num_mc_samples x q x r`
            fantasy_models = [self.model.condition_on_observations(X_, Y__[:, W_idx_], self.w_options[W_idx_]) for Y__ in Y_]

            fantasy_acqf = FantasyMCAcquisitionFunction(
                fantasy_models=fantasy_models,
                utility=self.x_utility,
                w_options=self.w_options,
                aggregation_function=self.aggregation_function,
                sample_reduction=self.sample_reduction,
                q_reduction=self.q_reduction,
                sampler=self.inner_sampler,
                max_batch_size=self._inference_batch_size
            )
            _, acqf_value = fantasy_acqf.optimize(q=1, bounds=self.x_bounds, options=self.x_options)  # _, `1`
            all_acqf_values.append(acqf_value)

        return torch.stack(all_acqf_values, dim=0)  # `n`

    def _forward_w(self, X: Tensor) -> Tensor:
        """
        Calculates the acquisition function values for a given set of x locations, assuming the w options are fixed.

        Args:
            X: A `n x q x (d_x + d_w)` tensor of x-w design points to evaluate.

        Returns:
            A `n` tensor of acquisition function values.
        """
        posterior = self.model.posterior(X)
        mc_samples = self.sampler(posterior)  # num_mc_samples x n x q x m

        all_acqf_values = []

        for X_, Y_ in zip(X, mc_samples.permute(1, 0, 2, 3)):  # X_: `q x d_x`, Y_: `num_mc_samples x q x m`
            fantasy_models = [self.model.condition_on_observations(X_, Y__) for Y__ in Y_]

            fantasy_acqf = FantasyMCAcquisitionFunction(
                fantasy_models=fantasy_models,
                utility=self.w_utility,
                w_options=self.w_options,
                aggregation_function=self.aggregation_function,
                sample_reduction=self.sample_reduction,
                q_reduction=self.q_reduction,
                sampler=self.inner_sampler,
                max_batch_size=self._inference_batch_size
            )

            _, acqf_value = fantasy_acqf.optimize(q=1, bounds=self.x_bounds, options=self.x_options)

            all_acqf_values.append(acqf_value)

        return torch.stack(all_acqf_values, dim=0)  # `n`
