import random
import torch

from currybo.test_functions.problem_set import AnalyticalProblemSet
from currybo.test_functions.parametrizable_function import ParametrizedBaseTestProblem
from currybo.test_functions.synthetic_functions.branin import ParametrizedBranin


def test_analytical_problemset():

    num_problems = random.randint(1, 10)
    parameter_ranges = {"b": (3.0, 6.0), "c": (0.0, 2.0)}

    problem_set = AnalyticalProblemSet(
        problem_family=ParametrizedBranin,
        num_problems=num_problems,
        parameter_ranges=parameter_ranges,
    )

    assert isinstance(problem_set, AnalyticalProblemSet)

    problem_list, param_tensor = problem_set._setup_problems(**(parameter_ranges))

    assert isinstance(problem_list, list)

    assert len(problem_list) == num_problems

    for problem in problem_list:
        assert isinstance(problem, ParametrizedBaseTestProblem)
    
    assert isinstance(param_tensor, torch.Tensor)
    assert param_tensor.shape[0] == num_problems
    assert param_tensor.shape[1] == len(parameter_ranges)
    assert param_tensor.dim() == 2

    assert len(problem_set) == num_problems

    for i in range(num_problems):
        assert isinstance(problem_set[i], ParametrizedBaseTestProblem)

    assert isinstance(problem_set._problems, list)
    assert isinstance(problem_set._problems[0], ParametrizedBaseTestProblem)
    assert len(problem_set._problems) == num_problems

    assert problem_set.dim == problem_set[0].dim
    assert problem_set.dim == 2

    assert problem_set.bounds.shape[0] == 2
    assert problem_set.bounds.shape[1] == problem_set.dim
    assert problem_set.bounds.dim() == 2

    dummy_X = torch.rand((random.randint(1, 5), problem_set.dim))
    dummy_idx = torch.randint(0, num_problems, (dummy_X.shape[0],))

    dummy_output = problem_set.evaluate_true(dummy_X, dummy_idx)

    assert dummy_X.shape[0] == dummy_output.shape[0]
    assert dummy_X.dim() == 2
