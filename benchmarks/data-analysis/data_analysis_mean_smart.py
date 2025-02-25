from currybo.test_functions import CernakLoader, DenmarkLoader, DoyleLoader, BorylationLoader, DeoxyfluorinationLoader
import torch
from torch import tensor
import numpy as np
from analysis_utils import find_most_general_option, get_generalizability_score
from currybo.aggregation_functions import Mean, Sigmoid, MSE, Min
import time
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from functools import reduce
import pub_ready_plots
from pypalettes import load_cmap
import random

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(1)
np.random.seed(1)

def perform_analysis(dataset, proxy_model, aggregation_function, test_set):

    train_score, train_worst_score, train_option = find_most_general_option(x_options=dataset.x_options, w_options=dataset.w_options_train, proxy_model=proxy_model, aggregation_function=aggregation_function)
    test_score, test_worst_score, test_option = find_most_general_option(x_options=dataset.x_options, w_options=test_set, proxy_model=proxy_model, aggregation_function=aggregation_function)

    scaled_score = get_generalizability_score(x_options_tensor=train_option, w_options=test_set, proxy_model=proxy_model, aggregation_function=aggregation_function, min=test_worst_score, max=test_score)
    return scaled_score.item(), train_option, test_option


def dataset_analysis(loader, variable_names, target, enhance, aggregation_function, maximum_split):

    dataset = loader(variable_names=variable_names, target=target, enhance=enhance, prefix="../../")
    proxy_model = dataset.load_model(dataset.model_path)
    dataset.random_separate_split(num_samples=maximum_split)
    train_set = dataset.w_options_train
    test_set = dataset.w_options_test
    dataset.w_options = train_set

    generate_ranges_list = lambda lst: [list(t) for t in product(*[range(1, x+1) for x in lst])]
    splits = generate_ranges_list(maximum_split)

    score_matrix = torch.zeros((1, reduce(lambda x, y: x*y, maximum_split)))
    score_counts = torch.zeros_like(score_matrix)

    for split in splits:
        dataset.random_smart_split(num_samples=split)

        scaled_score_sr, train_option_sr, test_option_sr = perform_analysis(dataset=dataset, proxy_model=proxy_model, aggregation_function=aggregation_function, test_set=test_set)

        dataset.w_options = train_set

        index = reduce(lambda x, y: x * y, split) - 1
        score_matrix[0, index] = (score_matrix[0, index] * score_counts[0, index] + scaled_score_sr) / (score_counts[0, index] + 1)
    
        score_counts[0, index] += 1

    return score_matrix

def plot_results(score_matrix, label, ylabel, ratio=False):

    if ratio:
        score_matrix / score_matrix[:, 0].unsqueeze(1)

    cmap = load_cmap("Austria")

    with pub_ready_plots.get_context(
        width_frac=1,  # between 0 and 1
        height_frac=0.35,  # between 0 and 1
        layout="neurips",  # or "iclr", "neurips", "poster-portrait", "poster-landscape"
        single_col=False,  # only works for the "icml" layout
        nrows=1,  # depending on your subplots, default = 1
        ncols=1,  # depending on your subplots, default = 1
        override_rc_params={"lines.linewidth": 2.0},  # Overriding rcParams
        sharey=True,  # Additional keyword args for `plt.subplots`
    ) as (fig, axs):

        mean = score_matrix.mean(dim=0)
        ste = score_matrix.std(dim=0) / torch.sqrt(torch.tensor(score_matrix.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs.errorbar(torch.arange(1, score_matrix.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(0),
                    ecolor = 'black',
                    elinewidth = 1.5,
                    capsize = 2,
                    capthick = 1.5
        )

        axs.set_ylabel(ylabel)
        axs.set_xlabel("Number of substrates")
        axs.grid(True)
        fig.tight_layout()

        fig.savefig(f"dataset_analysis_{label}_mean_smart.pdf")

        plt.clf()


def main():

    start_time = time.time()

    variable_names_cernak = ["Catalyst", "Base"]
    target_cernak = "Conversion"
    variable_names_denmark = ["Catalyst"]
    target_denmark = "Delta_Delta_G"
    variable_names_doyle = ["Ligand", "Additive", "Base"]
    target_doyle = "Yield"
    variable_names_borylation = ["ligand", "solvent"]
    target_borylation = "yield"
    variable_names_deoxyfluorination = ["base", "fluoride"]
    target_deoxyfluorination = "yield"

    NUM_ITERATIONS = 30
    AGGREGATION = Mean()

    datasets = [
        (CernakLoader, variable_names_cernak, target_cernak, True, AGGREGATION, [12], r"Scaled generality ($\uparrow$)"),
        (DenmarkLoader, variable_names_denmark, target_denmark, True, AGGREGATION, [3, 3], r"Scaled generality ($\uparrow$)"),
        (DoyleLoader, variable_names_doyle, target_doyle, True, AGGREGATION, [10], r"Scaled generality ($\uparrow$)"),
        (BorylationLoader, variable_names_borylation, target_borylation, True, AGGREGATION, [20], r"Scaled generality ($\uparrow$)"),
        (DeoxyfluorinationLoader, variable_names_deoxyfluorination, target_deoxyfluorination, True, AGGREGATION, [25], r"Scaled generality ($\uparrow$)"),
        (CernakLoader, variable_names_cernak, target_cernak, False, AGGREGATION, [12], r"Scaled generality ($\uparrow$)"),
        (DenmarkLoader, variable_names_denmark, target_denmark, False, AGGREGATION, [3, 3], r"Scaled generality ($\uparrow$)"),
        (DoyleLoader, variable_names_doyle, target_doyle, False, AGGREGATION, [10], r"Scaled generality ($\uparrow$)"),
        (BorylationLoader, variable_names_borylation, target_borylation, False, AGGREGATION, [20], r"Scaled generality ($\uparrow$)"),
        (DeoxyfluorinationLoader, variable_names_deoxyfluorination, target_deoxyfluorination, False, AGGREGATION, [25], r"Scaled generality ($\uparrow$)"),
    ]

    for dataset_num, (loader, variable_names, target, enhance, aggregation_function, maximum_split, ylabel) in enumerate(datasets):

        print("NEW DATASET")

        matrices = []

        for iteration in range(NUM_ITERATIONS):

            print(f"ITERATION {iteration + 1}")

            torch.manual_seed(iteration * 5)
            np.random.seed(iteration * 5)
            random.seed(iteration * 5)

            matrices.append(dataset_analysis(loader=loader, variable_names=variable_names, target=target, enhance=enhance, aggregation_function=aggregation_function, maximum_split=maximum_split))

        score_matrix = torch.vstack(matrices)

        label = f"{loader(variable_names=variable_names, target=target, prefix="../../").dataset_name}{'_augmented' if enhance else ''}_scaled"
        torch.save(score_matrix, f"dataset_analysis_{label}_mean_smart.pt")

    end_time = time.time()
    print(f"TIME: {end_time - start_time}")

if __name__ == "__main__":

    main()