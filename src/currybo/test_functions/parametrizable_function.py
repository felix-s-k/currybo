from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Optional, Dict, Union
import torch
from botorch.test_functions.base import BaseTestProblem


class ParametrizedBaseTestProblem(BaseTestProblem, metaclass=ABCMeta):
    """
    Abstract base class for parametrizable analytical test problems (i.e. families of analytical test problems).
    Inherits from botorch's `BaseTestProblem` class, and implements additional functionality to add custom
    parametrization and custom bounds that can be set upon initialization.

    Args:
        noise_std: Standard deviation of the observation noise. If a list is provided, specifies separate noise standard
                   deviations for each objective in a multiobjective problem.
        negate: If True, negate the function.
        bounds: A list of (lower, upper) bounds for each dimension of the input space (length `d`).
        parameter_dict: A dictionary of parameter names and their corresponding values.
        dim: Dimensionality of the problem.
        scalarize_surface: Bool whether the surface should be scalarized.
        scalarize_surface_kwargs: A dictionary to specify the lower and upper bounds for surface scalarization.
    """
    _parameter_defaults = {}

    dim: int = None
    _bounds: List[Tuple[float, float]] = None
    _optimal_value: float = None
    _optimizers: Optional[List[Tuple[float, ...]]] = None

    def __init__(
            self,
            noise_std: Union[None, float, List[float]] = None,
            negate: bool = False,
            dim: Optional[int] = None,
            bounds: Optional[List[Tuple[float, float]]] = None,
            parameter_dict: Optional[Dict[str, float]] = None,
            scalarize_surface: bool = False,
            scalarize_surface_kwargs: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None
    ):

        # Set up the input boundaries of the problem
        if dim:
            self.dim = dim
            self._bounds = [self._bounds[0] for _ in range(dim)]
        if bounds is not None:
            self._bounds = bounds
        if self._bounds is None:
            raise ValueError(f"No input bounds specified for {self.__class__.__name__}.")

        super().__init__(noise_std=noise_std, negate=negate)

        # Assert that at least one of the optimizers of the function lies within the specified bounds
        if self._optimizers is not None:

            def in_bounds(optimizer: Tuple[float, ...], bounds: List[Tuple[float, float]]) -> bool:
                for i, xopt in enumerate(optimizer):
                    lower, upper = bounds[i]
                    if xopt < lower or xopt > upper:
                        return False
                return True

            if not any(in_bounds(optimizer=optimizer, bounds=bounds) for optimizer in self._optimizers):
                raise ValueError("No global optimum found within custom bounds. Please specify bounds which include at "
                                 "least one point in `{self.__class__.__name__}._optimizers`.")

            self.register_buffer("optimizers", torch.tensor(self._optimizers, dtype=torch.get_default_dtype()))

        # Set the customizable parameters as attributes of the problem
        problem_parameters = self._parameter_defaults | (parameter_dict or {})
        for name, value in problem_parameters.items():
            setattr(self, name, value)

        setattr(self, "min_val", None)
        setattr(self, "max_val", None)

        if scalarize_surface:
            min_val, max_val = self.get_scalarization_factor()
            setattr(self, "min_val", min_val)
            setattr(self, "max_val", max_val)

            if len(scalarize_surface_kwargs) == 0:
                lower = 0
                upper = 1
            else:
                lower_low, lower_up = scalarize_surface_kwargs["lower"]
                lower = (lower_up - lower_low) * torch.rand(1).item() + lower_low
                upper_low, upper_up = scalarize_surface_kwargs["upper"]
                upper = (upper_up - upper_low) * torch.rand(1).item() + upper_low

            setattr(self, "lower", lower)
            setattr(self, "upper", upper)

    @abstractmethod
    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the true function values at the points `X`.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the function.

        Returns:
            A `batch_shape`-dim tensor of function evaluations.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_scalarization_factor(self) -> float:
        """
        Returns the scalarization factor for the problem.
        """
        raise NotImplementedError
