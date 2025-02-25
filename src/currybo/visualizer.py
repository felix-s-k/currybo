from typing import Union
from pathlib import Path
import torch

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import pub_ready_plots
from pypalettes import load_cmap

from currybo.campaign import GeneralBOCampaign


def visualize_surface(campaign: GeneralBOCampaign, file: Union[str, Path]) -> None:
    """
    visualizes the surface of the optimization campaign and creates a gif.

    Args:
        file: The file to save the visualization to.
    """

    ndim = campaign.observations_x.shape[1]

    if ndim not in (1,2):
        raise ValueError("Can only visualize surfaces in maximum two dimensions.")
        
    if ndim == 1:
 
        x_points = torch.linspace(campaign.problem.bounds[0,0], campaign.problem.bounds[1,0], 100).unsqueeze(1)
        y_points = []
        for i in range(len(campaign.problem)):
            y_points.append(campaign.problem.evaluate_true(x_points, torch.tensor([i]).repeat(x_points.shape[0])).squeeze(-1))

        y_points = [tensor.unsqueeze(0) if tensor.ndimension() == 1 else tensor for tensor in y_points]
        y_true = torch.cat(y_points, dim=0).t()
        y_agg = campaign.acquisition_strategy.aggregation_function(y_true).unsqueeze(1)

        y_max, y_argmax = torch.max(y_agg.squeeze(-1), dim=0)

        os.makedirs('temporary_visualization', exist_ok=True)

        for j in range(campaign.observations_x.shape[0]):

            surrogate = campaign.surrogate_type(
                train_X=campaign.observations_x[:j+1],
                train_W=campaign.observations_w[:j+1],
                train_Y=campaign.observations_y[:j+1],
                **campaign.surrogate_kwargs
            )
            surrogate.fit()

            means, variances = [], []

            for w in campaign.problem.w_options:
                posterior = surrogate.posterior(x_points, w.repeat(x_points.shape[0], 1))
                means.append(posterior.mean)  # shape: (n, 1)
                variances.append(posterior.variance)  # shape: (n, 1)

            plt.figure(figsize=(10, 6))
            for k in range(j+1):
                plt.plot(campaign.observations_x[k], campaign.observations_y[k], 'ro', color='red')
            for i in range(y_true.shape[1]):
                plt.plot(x_points, y_true[:,i], 'b--', color='lightblue')
            plt.plot(x_points, y_agg, 'b-', color='darkblue', label='Aggregation')
            plt.plot(x_points[y_argmax], y_max, '*', color='yellow', markersize=14, label='Optimum')
            plt.vlines(campaign.optimum_x[j], y_true.min() - 0.2 * (y_true.max() - y_true.min()).item(), y_true.max() + 0.2 * (y_true.max() - y_true.min()).item(), color='green', label='Predicted Optimum')
            plt.ylim([(y_true.min() - 0.2 * (y_true.max() - y_true.min())).item(), (y_true.max() + 0.2 * (y_true.max() - y_true.min())).item()])
            plt.legend()
            plt.grid()
            plt.savefig(f'temporary_visualization/temporary_visualization_{j}.png')
            plt.clf()

        imageio.mimsave(file, [imageio.imread(f'temporary_visualization/temporary_visualization_{j}.png') for j in range(campaign.observations_x.shape[0])], fps=1)

        shutil.rmtree('temporary_visualization')
    
    if ndim == 2:

        x1 = torch.linspace(campaign.problem.bounds[0, 0], campaign.problem.bounds[1, 0], 100)
        x2 = torch.linspace(campaign.problem.bounds[0, 1], campaign.problem.bounds[1, 1], 100)

        x1, x2 = torch.meshgrid(x1, x2, indexing='ij')
        x_points = torch.stack((x1, x2), dim=2).reshape(-1, 2)

        y_points = []
        for i in range(len(campaign.problem)):
            y_points.append(campaign.problem.evaluate_true(x_points, torch.tensor([i]).repeat(x_points.shape[0])).squeeze(-1))

        y_points = [tensor.unsqueeze(0) if tensor.ndimension() == 1 else tensor for tensor in y_points]
        y_true = torch.cat(y_points, dim=0).t()
        y_agg = campaign.acquisition_strategy.aggregation_function(y_true).unsqueeze(1)

        y_max, y_argmax = torch.max(y_agg.squeeze(-1), dim=0)

        max_location_x1, max_location_x2 = x_points[y_argmax, 0], x_points[y_argmax, 1]

        os.makedirs('temporary_visualization', exist_ok=True)

        for j in range(campaign.observations_x.shape[0]):

            plt.scatter(x_points[:, 0], x_points[:, 1], c=y_agg.squeeze(), cmap='Blues')
            for k in range(j+1):
                plt.plot(campaign.observations_x[k, 0], campaign.observations_x[k, 1], 'ro', color='red')
            plt.plot(max_location_x1, max_location_x2, '*', color='yellow', markersize=14)
            plt.plot(campaign.optimum_x[j, 0], campaign.optimum_x[j, 1], '*', color='green', markersize=14)
            plt.savefig(f'temporary_visualization/temporary_visualization_{j}.png')
            plt.clf()

        imageio.mimsave(file, [imageio.imread(f'temporary_visualization/temporary_visualization_{j}.png') for j in range(campaign.observations_x.shape[0])], fps=1)

        shutil.rmtree('temporary_visualization')

def visualize_run(campaign_dict: dict, file: str, vis_surface: bool, MAX_BUDGET: int, zoomed_run: bool, plot_test_set: bool = False, gap: bool = False) -> None:
    '''
    Visualizes the recommended optimum at each iteration of the optimization campaign for multiple runs and allows comparison.

    Args:
        campaign_dict: A dictionary with label as keys and list of campaign objects to be plotted with the mean and standard error. Can also have a single campaign object as value.
        file: Filename to story the visualisation.
        vis_surface: Whether to visualize the surface of the optimization campaign(s).
        MAX_BUDGET: Maximum budget.
        zoomed_run: Whether to create a second plot with the zoomed maximum.
        plot_test_set: Whether to plot the optimum in the test set.
    '''
    cmap = load_cmap("Austria")

    with pub_ready_plots.get_context(
        width_frac=1,  # between 0 and 1
        height_frac=0.45,  # between 0 and 1
        layout="neurips",  # or "iclr", "neurips", "poster-portrait", "poster-landscape"
        single_col=False,  # only works for the "icml" layout
        nrows=1,  # depending on your subplots, default = 1
        ncols=1,  # depending on your subplots, default = 1
        override_rc_params={"lines.linewidth": 1.2, "font.family": 'sans-serif'},  # Overriding rcParams
        sharey=True,  # Additional keyword args for `plt.subplots`
    ) as (fig, axs):

        continuous_campaign = True
        axs.set_xlim([0, MAX_BUDGET])
        for i, (label, campaigns) in enumerate(campaign_dict.items()):
            if isinstance(campaigns, GeneralBOCampaign) or (isinstance(campaigns, list) and len(campaigns) == 1):
                if isinstance(campaigns, list):
                    campaigns = campaigns[0]

                if campaigns._problem_type != 'continuous':
                    continuous_campaign = False

                axs.plot(range(campaigns.generality_train_set.shape[0]), campaigns.generality_train_set, label=label, color=cmap(i))

                if plot_test_set and (campaigns.test_problem is not None):
                    axs.plot(range(campaigns.generality_test_set.shape[0]), campaigns.generality_test_set, label=label, color=cmap(i), linestyle='dashed')
                
            elif isinstance(campaigns, list) and len(campaigns) > 1:

                all_campaigns_test = True
                for j, campaign in enumerate(campaigns):
                    
                    if campaign._problem_type != 'continuous':
                        continuous_campaign = False

                    if campaign.test_problem is None:
                        all_campaigns_test = False

                generality_train_set = [campaign.generality_train_set for campaign in campaigns]
                generality_stacked = torch.stack(generality_train_set)
                mean_tensor = torch.mean(generality_stacked, dim=0)
                std_error_tensor = torch.std(generality_stacked, dim=0) / np.sqrt(generality_stacked.size(0))
                y_mean = mean_tensor.view(-1).numpy()
                y_ste = std_error_tensor.view(-1).numpy()

                if not gap:

                    if i == 0:
                        global_optimum_set = [campaign.global_optimum for campaign in campaigns]
                        global_optimum_stacked = torch.stack(global_optimum_set)
                        global_optimum_mean = torch.mean(global_optimum_stacked, dim = 0)
                        global_optimum_ste = torch.std(global_optimum_stacked, dim = 0) / np.sqrt(global_optimum_stacked.size(0)).squeeze()

                        global_optimum_mean = global_optimum_mean.squeeze()
                        global_optimum_ste = global_optimum_ste.squeeze()
                        axs.axhline(y=global_optimum_mean, color='black', linestyle = 'dashed', label = "Global optimum")
                        axs.fill_between(range(y_mean.shape[0]), (global_optimum_mean - global_optimum_ste), (global_optimum_mean + global_optimum_ste), alpha=0.3, color='black')


                    axs.plot(range(y_mean.shape[0]), y_mean, label=campaigns[0].label if hasattr(campaigns[0], label) else label, color=cmap(i))
                    axs.fill_between(range(y_mean.shape[0]), (y_mean - y_ste), (y_mean + y_ste), alpha=0.3, color=cmap(i))

                    axs.set_ylabel(campaigns[0].ylabel if campaigns[0].ylabel is not None else 'Generality metric', size=10)

                    if plot_test_set and all_campaigns_test and not gap:
                        
                        generality_test_set = [campaign.generality_test_set for campaign in campaigns]
                        generality_stacked = torch.stack(generality_test_set)
                        mean_tensor = torch.mean(generality_stacked, dim=0)
                        std_error_tensor = torch.std(generality_stacked, dim=0) / np.sqrt(generality_stacked.size(0))
                        y_mean = mean_tensor.view(-1).numpy()
                        y_ste = std_error_tensor.view(-1).numpy()

                        axs.plot(range(y_mean.shape[0]), y_mean, label=label, color=plt.get_cmap('tab10')(i), linestyle='dashed')
                        #plt.fill_between(range(y_mean.shape[0]), (y_mean - y_ste), (y_mean + y_ste), alpha=0.3, color=plt.get_cmap('tab10')(i), linestyle='dashed')

                else:

                    global_optimum_set = [campaign.global_optimum for campaign in campaigns]
                    global_optimum_stacked = torch.cat(global_optimum_set)
                    generality_gap = generality_stacked.squeeze()
                    diff = global_optimum_stacked - generality_gap[:, 0].unsqueeze(1)
                    if (diff == 0).any():
                        zeros = (diff == 0).all(dim=1)
                    # Skip the rows with the starting conditions that are already optimal
                    generality_gap = generality_gap[~zeros]
                    global_optimum_stacked = global_optimum_stacked[~zeros]

                    gap_value = (generality_gap - generality_gap[:, 0].unsqueeze(1)) / (global_optimum_stacked - generality_gap[:, 0].unsqueeze(1))
                    gap_value = gap_value.unsqueeze(-1)
                    
                    mean_gap = torch.mean(gap_value, dim=0)
                    std_error_gap = torch.std(gap_value, dim=0) / np.sqrt(gap_value.size(0))
                    y_mean = mean_gap.view(-1).numpy()
                    y_ste = std_error_gap.view(-1).numpy()

                    axs.plot(range(y_mean.shape[0]), y_mean, label=campaigns[0].label if hasattr(campaigns[0], label) else label, color=cmap(i))
                    axs.fill_between(range(y_mean.shape[0]), (y_mean - y_ste), (y_mean + y_ste), alpha=0.3, color=cmap(i))

                    axs.set_ylabel(r'GAP ($\uparrow$)', size=10)

                    axs.set_ylim([0, 1])
                    

            else:
                raise ValueError("The value of the dictionary should be a GeneralBOCampaign object or a list of GeneralBOCampaign objects.")

        axs.set_xlabel('Experiment number', size=10)
        axs.set_xticks(ticks=range(0, MAX_BUDGET + 1, 2))
        axs.set_xticks(range(0, MAX_BUDGET + 1, 1), minor=True)
        axs.xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs.grid(which='both', linestyle='--', linewidth=0.1)
        axs.legend(fontsize=9, ncol=2)
        fig.savefig(file)

        if zoomed_run:
            original_y_min, original_y_max = plt.gca().get_ylim()
            upper_limit = original_y_max
            lower_limit = original_y_min + 0.8 * (original_y_max - original_y_min)
            axs.ylim(lower_limit, upper_limit)
            fig.savefig(file.split('.')[0] + '_zoomed.pdf')

        plt.clf()

    # Only visualize surface for continuous campaign
    if vis_surface and continuous_campaign:
        for label, campaigns in campaign_dict.items():
            if isinstance(campaigns, GeneralBOCampaign) or (isinstance(campaigns, list) and len(campaigns) == 1):
                if isinstance(campaigns, list):
                    campaigns = campaigns[0]
                visualize_surface(campaign=campaigns, file=file.split('.')[0] + "_" + label + '_surface.gif')
            elif isinstance(campaigns, list) and len(campaigns) > 1:
                for j, campaign in enumerate(campaigns):
                    visualize_surface(campaign=campaign, file= file.split('.')[0] + label + '_surface_' + str(j) + '.gif')
            else:
                raise ValueError("The value of the dictionary should be a GeneralBOCampaign object or a list of GeneralBOCampaign objects.")
            
def plot_generalizability(runs_dict: dict, file: str) -> None:

    """
    Plot mean and standard error for multiple runs for continuous optimization domains.

    Args:
        runs_dict (Dict[Dict[str|list[float]]]): Dictionary containing all the runs. Should have the format:
            runs_dict = {
                'DataSet1': {'Optimization_method A': [generalizability_values], 'Optimization_method B': [generalizability_values]},
                'DataSet2': {'Optimization_method A': [generalizability_values], 'Optimization_method B': [generalizability_values]}
            }
        file (str): File name to save the plot.
    """

    def calculate_stats(data):
        mean = np.mean(data)
        stderr = np.std(data, ddof=1) / np.sqrt(len(data))
        return mean, stderr
    
    outer_keys = list(runs_dict.keys())
    inner_keys = list({key for subdict in runs_dict.values() for key in subdict.keys()})

    means = {key: [] for key in inner_keys}
    errors = {key: [] for key in inner_keys}

    for outer_key in outer_keys:
        for inner_key in inner_keys:
            values = runs_dict[outer_key].get(inner_key, [0])
            mean, error = calculate_stats(values)
            means[inner_key].append(mean)
            errors[inner_key].append(error)

    x = np.arange(len(outer_keys))
    width = 0.05
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, inner_key in enumerate(inner_keys):
        ax.bar(x + i*width, means[inner_key], width, label=inner_key, yerr=errors[inner_key], capsize=5)
    
    ax.set_ylabel('Generality metric')
    ax.set_xticks(x + width * (len(inner_keys) - 1) / 2)
    ax.set_xticklabels(outer_keys)
    ax.legend()

    plt.savefig(file)

    plt.clf()
