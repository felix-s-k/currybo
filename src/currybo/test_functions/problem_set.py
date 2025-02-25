from __future__ import annotations
from typing import Type, List, Tuple, Union, Optional, Dict, Callable
from abc import ABC, abstractmethod

import itertools
import numpy as np

import torch
from torch import Tensor

from .parametrizable_function import ParametrizedBaseTestProblem


# TODO: Implement a "general" ProblemSet class that can handle both analytical and dataset-based test problems.
#       Probably, all methods from below but the '_setup_problems' method should be implemented in the 'general' class.
#       Then 'AnalyticalProblemSet' (as well as a further class for dataset-based problems) can inherit from this class.
#       Later on, we should extend this framework to mixed continuous-categorical variable optimization problems.
# TODO: Implement the 'train_test_split' method for the 'GeneralProblemSet' class.
# TODO: implement DiscreteProblemSet and MixedProblemSet classes as GeneralProblemSet subclasses.

class GeneralProblemSet(ABC):
    """
    Abstract base class for a problem set that can handle various types of test problems.
    This can be extended to support both analytical and dataset-based problems,
    as well as mixed continuous-categorical optimization problems.
    
    Args:
        noise_std (Union[None, float, List[float]]): Standard deviation of the observation noise.
        negate (bool): If True, negate the function.
    """
    
    def __init__(
            self,
            noise_std: Union[None, float, List[float]] = None,
            negate: bool = False,
    ):
        self.noise_std = noise_std
        self.negate = negate

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of problems in the set.
        """
        pass
    
    # @abstractmethod
    # def __getitem__(self, idx: int) -> ParametrizedBaseTestProblem:
    #     """
    #     Returns a specific problem from the set by index.
    #     """
    #     pass

    @abstractmethod
    def _setup_problems(self, *args, **kwargs):
        """
        Abstract method to set up the problems. This needs to be implemented by subclasses.
        """
        pass

    @abstractmethod
    def evaluate_true(self, X: Tensor, idx: Tensor) -> Tensor:
        """
        Abstract method to evaluate the true value of the problem at the given points.
        
        Args:
            X: A tensor of points at which to evaluate the problem.
            idx: A tensor of indices specifying which problem to evaluate at each point.
        """
        pass

    def apply_noise(self, values: Tensor) -> Tensor:
        """
        Applies noise to the values if noise_std is specified.
        
        Args:
            values: A tensor of values to which noise should be added.
            
        Returns:
            A tensor of values with added noise.
        """
        if self.noise_std is not None:
            noise = torch.randn_like(values) * self.noise_std
            values = values + noise
        return values
    
    def negate_output(self, values: Tensor) -> Tensor:
        """
        Negates the values if negate is True.
        
        Args:
            values: A tensor of values to negate.
            
        Returns:
            A tensor of values, negated if negate is True.
        """
        return -values if self.negate else values

    def train_test_split(self, splitting_scheme: str = "random", **kwargs) -> Tuple[GeneralProblemSet, GeneralProblemSet]:
        """
        Splits the set of test problems into two disjoint sets for training and testing.
        
        Args:
            splitting_scheme: The method used to split the test problems. Default is 'random'.
            kwargs: Additional keyword arguments for the splitting method.
            
        Returns:
            Two instances of the GeneralProblemSet, one for training and one for testing.
        """
        raise NotImplementedError("Train-test splitting is not implemented.")


class AnalyticalProblemSet(GeneralProblemSet):
    """
    Class that describes a set of related test problems from the same test function family.

    Args:
        problem_family: The type of problem in the family.
        num_problems: The number of problems in the family.
        noise_std: Standard deviation of the observation noise. If a list is provided, specifies separate noise standard
                   deviations for each objective in a multiobjective problem. Argument as specified in botorch's
                   `BaseTestProblem` class.
        negate: If True, negate the function. Argument as specified in botorch's `BaseTestProblem` class.
        bounds: A list of (lower, upper) bounds for each dimension of the input space (length `d`).
        parameter_ranges: A dictionary of parameter names and their corresponding lower and upper bounds as tuples, i.e.
                          {'param1': (lower1, upper1), 'param2': (lower2, upper2), ...}, where 'param1', 'param2', ...
                          are the names of the variable parameters of the problem family.
        scalarize_surface: If True, the scalarization factor of the problem is used to generate a scalarized surface.
        scalarize_surface_kwargs: Upper and lower ranges for surface scalarization.
    """
    def __init__(
            self,
            problem_family: Type[ParametrizedBaseTestProblem],
            num_problems: int,
            noise_std: Union[None, float, List[float]] = None,
            negate: bool = False,
            dim: Optional[int] = None,
            bounds: Optional[List[Tuple[float, float]]] = None,
            parameter_ranges: Dict[str, Tuple[float, float]] = None,
            scalarize_surface: bool = False,
            scalarize_surface_kwargs: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    ):

        self.problem_family = problem_family
        self.dim = dim if dim else problem_family.dim
        self.num_problems = num_problems
        self.scalarize_surface = scalarize_surface
        self.scalarize_surface_kwargs = scalarize_surface_kwargs if scalarize_surface_kwargs else {}

        self._problems, self.w_options = self._setup_problems(noise_std, negate, bounds, **(parameter_ranges or {}))

    def _setup_problems(
            self,
            noise_std: Union[None, float, List[float]] = None,
            negate: bool = False,
            bounds: Optional[List[Tuple[float, float]]] = None,
            **params
    ) -> Tuple[List[ParametrizedBaseTestProblem], torch.Tensor]:
        """
        Sets up the individual members of the family by sampling from a uniform distribution for each parameter.

        Args:
            noise_std: Standard deviation of the observation noise. If a list is provided, specifies separate noise
                       standard deviations for each objective in a multiobjective problem. Argument as specified in
                       botorch's `BaseTestProblem` class.
            negate: If True, negate the function. Argument as specified in botorch's `BaseTestProblem` class.
            params: A dictionary of parameter names and their corresponding lower and upper bounds as tuples, i.e.
                    {'param1': (lower1, upper1), 'param2': (lower2, upper2), ...}, where 'param1', 'param2', ... are
                    the names of the variable parameters of the problem family.

        Returns:
            List[ParametrizedBaseTestProblem]: A list of individual instances of the problem in the family.
            torch.Tensor: A `num_problems x num_params` tensor of the sampled parameter values.
        """
        param_samples = {
            name: torch.FloatTensor(self.num_problems).uniform_(lower, upper)
            for name, (lower, upper) in params.items()
        }

        problems = [
            self.problem_family(
                noise_std=noise_std,
                negate=negate,
                dim=self.dim,
                bounds=bounds,
                parameter_dict={name: param_samples[name][i] for name in params.keys()},
                scalarize_surface=self.scalarize_surface,
                scalarize_surface_kwargs=self.scalarize_surface_kwargs
            )
            for i in range(self.num_problems)
        ]

        param_tensor = torch.stack(list(param_samples.values()), dim=-1)

        return problems, param_tensor

    def __len__(self) -> int:
        """
        Returns the number of problems in the family.
        """
        return self.num_problems

    def __getitem__(self, idx: int) -> ParametrizedBaseTestProblem:
        """
        Returns a specific problem from the family by index.
        """
        return self._problems[idx]

    @property
    def bounds(self) -> Tensor:
        """
        Returns the bounds of the problems in the family as a `2 x d` tensor.
        """
        return self._problems[0].bounds  # ATTN: Assumes all problems in the family have the same bounds

    def evaluate_true(self, X: Tensor, idx: Tensor) -> Tensor:
        """
        Evaluates the true value of the problem specified by the index `idx` at the given points `X`.

        Args:
            X: A `n x d` tensor of points at which to evaluate the problem.
            idx: A `n` tensor of indices specifying which problem to evaluate at each point.
        """
        return torch.stack(
            [
                self._problems[problem_idx].evaluate_true(X[sample_idx].unsqueeze(0))
                for sample_idx, problem_idx in enumerate(idx)
            ],
            dim=0
        )

    def train_test_split(self, splitting_scheme: str = "random", **kwargs) -> Tuple[AnalyticalProblemSet, AnalyticalProblemSet]:
        """
        Splits the set of test problems into two disjoint sets for training and testing.

        Currently implemented splitting schemes:
            - 'random': Randomly splits the set of test problems into two disjoint sets.
            - # TODO: Add more splitting schemes.

        Args:
            splitting_scheme: The method used to split the test problems. Default is 'random'.
            kwargs: Additional keyword arguments to be passed to the splitting method.

        Returns:
            AnalyticalProblemSet: The training set of test problems.
            AnalyticalProblemSet: The testing set of test problems.
        """
        # TODO: Implement train-test splitting of the problems (based on the specific problem parameters).

        raise NotImplementedError


class DiscreteProblemSet:
    """
    Class that describes a set of related discrete test problems from the same reaction dataset.

    Args:
        noise_std: Standard deviation of the observation noise. If a list is provided, specifies separate noise standard
                   deviations for each objective in a multiobjective problem. Argument as specified in botorch's
                   `BaseTestProblem` class.
        negate: If True, negate the function. Argument as specified in botorch's `BaseTestProblem` class.
        x_options: A dictionary of the different options in the optimization domain. 
                    Keys are the different dimensions, values a list of options for each dimension.
        w_options: A dictionary of the different options in the parameter domain. 
                    Keys are the different parameters, values a list of options for each parameter.
        proxy_model: Model to call for experiment.
        min_value: Assumed or known minimum value of the target variable.
        max_value: Assumed or known maximum value of the target variable.
    """
    def __init__(
        self,
        noise_std: Union[None, float, List[float]] = None,
        negate: bool = False,
        x_options: Dict[str, List[Tensor]] = None,
        w_options: Dict[str, List[Tensor]] = None,
        proxy_model: Callable[[Tensor], float] = None,
        min_value: float = False,
        max_value: float = None,
    ):
        
        self.negate = negate
        self.noise_std = noise_std
        self.min_value = min_value
        self.max_value = max_value
        self.proxy_model = proxy_model

        self._setup_problems(x_options, w_options)

    def _setup_problems(
            self, 
            x_options: Dict[str, List[Tensor]] = None,
            w_options: Dict[str, List[Tensor]] = None,
        ):

        """
        Creates the options tensors for the x and w domain.

        Args:
            x_options: A dictionary of the different options in the optimization domain. 
                    Keys are the different dimensions, values a list of options for each dimension.
            w_options: A dictionary of the different options in the parameter domain. 
                    Keys are the different parameters, values a list of options for each parameter.
        """

        w_combinations = list(itertools.product(*w_options.values()))
        w_options_combinations = [torch.cat(combo) for combo in w_combinations]
        w_options_tensor = torch.stack(w_options_combinations)

        x_combinations = list(itertools.product(*x_options.values()))
        x_options_combinations = [torch.cat(combo) for combo in x_combinations]
        x_options_tensor = torch.stack(x_options_combinations)

        self.w_options = w_options_tensor
        self.x_options = x_options_tensor

    def evaluate_true(self, X: Tensor, idx: Tensor) -> Tensor:
        """
        Evaluates the true value based on the ML model of the problem specified by the index `idx` at the given points `X`.

        Args:
            X: A `n x d` tensor of points at which to evaluate the problem.
            idx: A `n` tensor of indices specifying which problem to evaluate at each point.
        """

        #TODO: Add noise here

        w_input = self.w_options[idx]
        w_input = w_input.repeat(X.shape[0], 1)

        input = torch.cat((X, w_input), dim=-1)

        pred = torch.tensor(np.array([self.proxy_model(input)]))

        if self.negate:
            return -pred
        else:
            return pred
    
    def __len__(self) -> int:
        """
        Returns the number of problems in the family.
        """
        return self.w_options.shape[0]
    
class MixedProblemSet:

    def __init__(
            self, 
            noise_std: Union[None, float, List[float]] = None,
            negate: bool = False,
            continuous_bounds: Optional[List[Tuple[float, float]]] = None,
            x_options: Dict[str, List[Tensor]] = None,
            w_options: Dict[str, List[Tensor]] = None,
            proxy_model: Callable[[Tensor], float] = None,
            min_value: float = False,
            max_value: float = None,
        ):
        
        self.negate = negate
        self.noise_std = noise_std
        self.min_value = min_value
        self.max_value = max_value
        self.proxy_model = proxy_model

        self._setup_problems(continuous_bounds, x_options, w_options)

    def _setup_problems(
            self,
            continuous_bounds: Optional[List[Tuple[float, float]]] = None,
            x_options: Dict[str, List[Tensor]] = None,
            w_options: Dict[str, List[Tensor]] = None,
        ):

        continuous_bounds_tensor = torch.tensor(continuous_bounds).transpose(0, 1).double()
        self.continuous_bounds = continuous_bounds_tensor

        lengths = [value[0].size(0) for value in x_options.values()]
        additional_zeros = torch.tensor([0] * sum(lengths)).unsqueeze(0)
        additional_ones = torch.tensor([1] * sum(lengths)).unsqueeze(0) 
        lower_bounds = torch.cat((continuous_bounds_tensor[0], additional_zeros[0]))
        upper_bounds = torch.cat((continuous_bounds_tensor[1], additional_ones[0]))
        bounds = torch.stack((lower_bounds, upper_bounds))
        self.bounds = bounds

        w_combinations = list(itertools.product(*w_options.values()))
        w_options_combinations = [torch.cat(combo) for combo in w_combinations]
        w_options_tensor = torch.stack(w_options_combinations)

        x_combinations = list(itertools.product(*x_options.values()))
        x_options_combinations = [torch.cat(combo) for combo in x_combinations]
        x_options_tensor = torch.stack(x_options_combinations)

        self.w_options = w_options_tensor
        self.x_options = x_options_tensor

    def evaluate_true(self, X: Tensor, idx: Tensor) -> Tensor:
        """
        Evaluates the true value based on the ML model of the problem specified by the index `idx` at the given points `X`.

        Args:
            X: A `n x d` tensor of points at which to evaluate the problem.
            idx: A `n` tensor of indices specifying which problem to evaluate at each point.
        """

        #TODO: Add noise here

        w_input = self.w_options[idx]

        input = torch.cat((X, w_input), dim=-1)

        pred = torch.tensor([self.proxy_model(input)])

        if self.negate:
            return -pred
        else:
            return pred
    
    def __len__(self) -> int:
        """
        Returns the number of problems in the family.
        """
        return self.w_options.shape[0]
