import torch
import random
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import GaussianLikelihood

from currybo.campaign import GeneralBOCampaign
from currybo.surrogate_models import SimpleGP
from currybo.acquisition_strategies import (
    BaseMCAcquisitionStrategy, SequentialAcquisition,
    UncertaintyUtility, QuantitativeImprovement,
)
from currybo.aggregation_functions import Mean
from currybo.test_functions import AnalyticalProblemSet, ParametrizedBranin


def test_campaign():

    campaign = GeneralBOCampaign()

    assert isinstance(campaign, GeneralBOCampaign)

    campaign.problem = AnalyticalProblemSet(
        problem_family=ParametrizedBranin,
        num_problems=20,
        negate=True,
        parameter_ranges={"b": (3.0, 6.0), "c": (3.0, 6.0)},
    )

    campaign.surrogate_type = SimpleGP
    campaign.surrogate_kwargs = {"kernel": MaternKernel, "likelihood": GaussianLikelihood}

    campaign.acquisition_strategy = SequentialAcquisition(
        x_bounds=campaign.problem.bounds,
        w_options=campaign.problem.w_options,
        aggregation_function=Mean(),
        x_utility= QuantitativeImprovement,
        x_utility_kwargs = {"best_f": 0},
        w_utility= UncertaintyUtility,
        w_utility_kwargs = None
    )

    assert isinstance(campaign.surrogate_kwargs, dict)

    assert isinstance(campaign.acquisition_strategy, BaseMCAcquisitionStrategy)

    assert isinstance(campaign.problem, AnalyticalProblemSet)

    num_seeds = random.randint(1, 5)

    seed_data, seed_w_idx = campaign._generate_seed_data(num_seeds=num_seeds)

    assert isinstance(seed_data, torch.Tensor)
    assert isinstance(seed_w_idx, torch.Tensor)
    assert seed_data.dim() == 2
    assert seed_w_idx.dim() == 1

    assert seed_data.shape[0] == num_seeds
    assert seed_w_idx.shape[0] == num_seeds
    assert seed_data.shape[1] == campaign.problem.dim
