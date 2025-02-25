import torch
from gpytorch.kernels import MaternKernel
from gauche.kernels.fingerprint_kernels import TanimotoKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior
import time
from joblib import Parallel, delayed
import argparse
import warnings
import os
import numpy as np
import random

from currybo.campaign import GeneralBOCampaign
from currybo.surrogate_models import SimpleGP, AdditiveStructureGP
from currybo.acquisition_strategies import SequentialAcquisition, SequentialLookaheadAcquisition, JointLookaheadAcquisition
from currybo.aggregation_functions import Mean, Sigmoid, MSE, Min
from currybo.test_functions import ChemistryDatasetLoader
from currybo.test_functions import DiscreteProblemSet, CernakLoader, DenmarkLoader, DoyleLoader, DeoxyfluorinationLoader, BorylationLoader
from currybo.acquisition_strategies.utility_function import UncertaintyUtility, QuantileUtility, Random

torch.set_default_dtype(torch.float64)
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_benchmark(dataset: ChemistryDatasetLoader, num_samples: list, i: int, budget: int):

    NUM_BUDGET = budget

    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

    print(f"----- ITERATION {i+1} -----", flush=True)
    campaign_seq = GeneralBOCampaign()
    campaign_random = GeneralBOCampaign()
    campaign_w_random = GeneralBOCampaign()
    campaign_all_random = GeneralBOCampaign()
            
    try:

        dataset.random_separate_split(num_samples=num_samples)

        proxy_model = dataset.load_model(dataset.model_path)

        campaign_seq.problem = DiscreteProblemSet(
            x_options=dataset.x_options,
            w_options=dataset.w_options_train,
            proxy_model=proxy_model,
            min_value=dataset.min_value,
            max_value=dataset.max_value
        )

        campaign_seq.test_problem = DiscreteProblemSet(
            x_options=dataset.x_options,
            w_options=dataset.w_options_test,
            proxy_model=proxy_model,
            min_value=dataset.min_value,
            max_value=dataset.max_value
        )

        campaign_w_random.problem = DiscreteProblemSet(
            x_options=dataset.x_options,
            w_options=dataset.w_options_train,
            proxy_model=proxy_model,
            min_value=dataset.min_value,
            max_value=dataset.max_value
        )

        campaign_w_random.test_problem = DiscreteProblemSet(
            x_options=dataset.x_options,
            w_options=dataset.w_options_test,
            proxy_model=proxy_model,
            min_value=dataset.min_value,
            max_value=dataset.max_value
        )

        campaign_random.problem = DiscreteProblemSet(
            x_options=dataset.x_options,
            w_options=dataset.w_options_train,
            proxy_model=proxy_model,
            min_value=dataset.min_value,
            max_value=dataset.max_value
        )

        campaign_random.test_problem = DiscreteProblemSet(
            x_options=dataset.x_options,
            w_options=dataset.w_options_test,
            proxy_model=proxy_model,
            min_value=dataset.min_value,
            max_value=dataset.max_value
        )

        campaign_all_random.problem = DiscreteProblemSet(
            x_options=dataset.x_options,
            w_options=dataset.w_options_train,
            proxy_model=proxy_model,
            min_value=dataset.min_value,
            max_value=dataset.max_value
        )

        campaign_all_random.test_problem = DiscreteProblemSet(
            x_options=dataset.x_options,
            w_options=dataset.w_options_test,
            proxy_model=proxy_model,
            min_value=dataset.min_value,
            max_value=dataset.max_value
        )

        campaign_seq.surrogate_type = SimpleGP
        campaign_seq.surrogate_kwargs = {"kernel": TanimotoKernel, "likelihood": GaussianLikelihood}
        campaign_w_random.surrogate_type = SimpleGP
        campaign_w_random.surrogate_kwargs = {"kernel": TanimotoKernel, "likelihood": GaussianLikelihood}
        campaign_random.surrogate_type = SimpleGP
        campaign_random.surrogate_kwargs = {"kernel": TanimotoKernel, "likelihood": GaussianLikelihood}

        campaign_seq.acquisition_strategy = SequentialAcquisition(
            x_bounds=None,
            x_options=campaign_seq.problem.x_options,
            w_options=campaign_seq.problem.w_options,
            aggregation_function=Mean(),
            x_utility=QuantileUtility,
            x_utility_kwargs={},
            w_utility=UncertaintyUtility,
            w_utility_kwargs={},
            maximization=True
        )

        campaign_w_random.acquisition_strategy = SequentialAcquisition(
            x_bounds=None,
            x_options=campaign_w_random.problem.x_options,
            w_options=campaign_w_random.problem.w_options,
            aggregation_function=Mean(),
            x_utility=QuantileUtility,
            x_utility_kwargs={},
            w_utility=Random,
            w_utility_kwargs={},
            maximization=True
        )

        campaign_random.acquisition_strategy = SequentialAcquisition(
            x_bounds=None,
            x_options=campaign_random.problem.x_options,
            w_options=campaign_random.problem.w_options,
            aggregation_function=Mean(),
            x_utility=Random,
            x_utility_kwargs={},
            w_utility=Random,
            w_utility_kwargs={},
            maximization=True
        )

        campaign_all_random.acquisition_strategy = SequentialAcquisition(
            x_bounds=None,
            x_options=campaign_random.problem.x_options,
            w_options=campaign_random.problem.w_options,
            aggregation_function=Mean(),
            x_utility=Random,
            x_utility_kwargs={},
            w_utility=Random,
            w_utility_kwargs={},
            maximization=True
        )

        start_time = time.time()

        if os.path.exists(f"{dataset.dataset_name}_sequential_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt"):
            campaign_seq.load_from_file(f"{dataset.dataset_name}_sequential_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt")
        campaign_seq.run_optimization(budget=NUM_BUDGET, num_seeds=1, random_seed=i, save_file=f"{dataset.dataset_name}_sequential_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt")
        campaign_seq.save(f"{dataset.dataset_name}_sequential_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt")

        end_time = time.time()
        print(f" OPTIMIZATION SEQUENTIAL DONE, TIME: {end_time - start_time}", flush=True)

        start_time = time.time()

        if os.path.exists(f"{dataset.dataset_name}_w_random_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt"):
            campaign_w_random.load_from_file(f"{dataset.dataset_name}_w_random_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt")
        campaign_w_random.run_optimization(budget=NUM_BUDGET, num_seeds=1, random_seed=i, save_file=f"{dataset.dataset_name}_w_random_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt")
        campaign_w_random.save(f"{dataset.dataset_name}_w_random_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt")

        end_time = time.time()
        print(f" OPTIMIZATION W RANDOM DONE, TIME: {end_time - start_time}", flush=True)

        start_time = time.time()

        if os.path.exists(f"{dataset.dataset_name}_random_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt"):
            campaign_random.load_from_file(f"{dataset.dataset_name}_random_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt")
        campaign_random.run_optimization(budget=NUM_BUDGET, num_seeds=1, random_seed=i, save_file=f"{dataset.dataset_name}_random_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt")
        campaign_random.save(f"{dataset.dataset_name}_random_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt")

        end_time = time.time()
        print(f" OPTIMIZATION RANDOM DONE, TIME: {end_time - start_time}", flush=True)

        # Get values for completely random acquisition
        campaign_all_random.observations_x = campaign_all_random.problem.x_options[torch.randint(0, campaign_all_random.problem.x_options.shape[0], (NUM_BUDGET + 1,))]
        campaign_all_random.observations_w_idx = torch.randint(0, campaign_all_random.problem.w_options.shape[0], (NUM_BUDGET + 1,))
        campaign_all_random.observations_w = campaign_all_random.problem.w_options[campaign_all_random.observations_w_idx]
        campaign_all_random.optimum_x = campaign_all_random.problem.x_options[torch.randint(0, campaign_all_random.problem.x_options.shape[0], (NUM_BUDGET + 1,))]
        obs_y = []
        for n in range(campaign_all_random.observations_x.shape[0]):
            y = campaign_all_random.problem.evaluate_true(campaign_all_random.observations_x[n].unsqueeze(0), campaign_all_random.observations_w_idx[n].unsqueeze(0))
            obs_y.append(y)
        campaign_all_random.observations_y = torch.cat(obs_y)
        campaign_all_random.global_optimum = campaign_seq.global_optimum
        campaign_all_random.generality_train_set = campaign_all_random.evaluate_generalizability(optimum=campaign_all_random.optimum_x, test_problem=campaign_all_random.problem)
        campaign_all_random.generality_test_set = campaign_all_random.evaluate_generalizability(optimum=campaign_all_random.optimum_x, test_problem=campaign_all_random.test_problem)
        campaign_all_random.save(f"{dataset.dataset_name}_all_random_{i}{'_enhance' if dataset.enhanced else ''}_mean.pt")
        print(f" OPTIMIZATION ALL RANDOM DONE, TIME: {end_time - start_time}", flush=True)

    except Exception as e:
        print(f"Error: {e}", flush=True)

    return (campaign_seq, campaign_w_random, campaign_random, campaign_all_random)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Key to define analytical problem set.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset set name', choices=["Cernak", "Denmark", "Doyle", "Deoxyfluorination", "Borylation"])
    parser.add_argument('--enhanced', action='store_true', default=True, help='Whether to use enhanced dataset')
    parser.add_argument('--not-enhanced', action='store_false', dest='enhanced', help='Do not use enhanced dataset')


    args = parser.parse_args()

    if args.dataset == "Cernak":
        variable_names = ["Catalyst", "Base"]
        target = "Conversion"
        dataset = CernakLoader(variable_names=variable_names, target=target, enhance=args.enhanced, prefix="../../")
        num_samples = [12]
    elif args.dataset == "Denmark":
        variable_names = ["Catalyst"]
        target = "Delta_Delta_G"
        dataset = DenmarkLoader(variable_names=variable_names, target=target, enhance=args.enhanced, prefix="../../")
        num_samples = [3, 3]
    elif args.dataset == "Doyle":
        variable_names = ["Ligand", "Additive", "Base"]
        target = "Yield"
        dataset = DoyleLoader(variable_names=variable_names, target=target, enhance=args.enhanced, prefix="../../")
        num_samples = [10]
    elif args.dataset == "Deoxyfluorination":
        variable_names = ["base", "fluoride"]
        target = "yield"
        dataset = DeoxyfluorinationLoader(variable_names=variable_names, target=target, enhance=args.enhanced, prefix="../../")
        num_samples = [25]
    elif args.dataset == "Borylation":
        variable_names = ["ligand", "solvent"]
        target = "yield"
        dataset = BorylationLoader(variable_names=variable_names, target=target, enhance=args.enhanced, prefix="../../")
        num_samples = [20]
    else:
        ValueError("Unallowed discrete dataset called!")

    NUM_ITER = 30
    NUM_BUDGET = 100

    parallel = Parallel(n_jobs = NUM_ITER, return_as="generator")

    collected_output = parallel(delayed(run_benchmark)(dataset=dataset, num_samples=num_samples, i=i, budget=NUM_BUDGET) for i in range(NUM_ITER))