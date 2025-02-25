from currybo.campaign import GeneralBOCampaign
import torch
import pub_ready_plots
from pypalettes import load_cmap
import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings 
warnings.filterwarnings('ignore') 

def nanstd(x):
    return torch.sqrt(torch.nanmean(torch.pow(x - torch.nanmean(x, dim=0), 2), dim = 0))

def create_plot(first_element, second_element, MAX_BUDGET, file_name, gap=True):

    cmap = load_cmap("Klein")

    with pub_ready_plots.get_context(
        width_frac=1,  # between 0 and 1
        height_frac=0.2,  # between 0 and 1
        layout=pub_ready_plots.Layout.ICML,  # or "iclr", "neurips", "poster-portrait", "poster-landscape"
        single_col=False,  # only works for the "icml" layout
        nrows=1,  # depending on your subplots, default = 1
        ncols=2,  # depending on your subplots, default = 1
        override_rc_params={"lines.linewidth": 1.2, "text.usetex": True},  # Overriding rcParams
        sharey=True,  # Additional keyword args for `plt.subplots`
    ) as (fig, axs):
        
        COLOR_IDX = [4, 2, 7, 8]

        axs[0].set_xlim([0, MAX_BUDGET])
        overall_max = 0
        for i, (label, campaigns) in enumerate(first_element.items()):
            generality_train_set = [campaign.generality_train_set for campaign in campaigns]
            generality_train_set = [tensor[:(MAX_BUDGET + 1), :] for tensor in generality_train_set]
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
                    axs[0].axhline(y=global_optimum_mean, color='black', linestyle = 'dashed', label = "Global optimum")
                    axs[0].fill_between(range(y_mean.shape[0]), (global_optimum_mean - global_optimum_ste), (global_optimum_mean + global_optimum_ste), alpha=0.3, color='black')
                    overall_max = max(overall_max, global_optimum_mean + global_optimum_ste)


                axs[0].plot(range(y_mean.shape[0]), y_mean, label=r'\textsc{' + campaigns[0].label + '}', color=cmap(i))
                axs[0].fill_between(range(y_mean.shape[0]), (y_mean - y_ste), (y_mean + y_ste), alpha=0.3, color=cmap(i))
                axs[0].set_ylabel(campaigns[0].ylabel if campaigns[0].ylabel is not None else 'Generality metric', size=10)

            else:

                global_optimum_set = [campaign.global_optimum for campaign in campaigns]
                global_optimum_stacked = torch.cat(global_optimum_set)
                generality_gap = generality_stacked.squeeze()
                diff = global_optimum_stacked - generality_gap[:, 0].unsqueeze(1)
                if torch.isclose(diff, torch.zeros_like(diff), atol=0.01).any():
                    zeros = torch.isclose(diff, torch.zeros_like(diff), atol=0.01).all(dim=1)
                    generality_gap = generality_gap[~zeros]
                    global_optimum_stacked = global_optimum_stacked[~zeros]

                gap_value = (generality_gap - generality_gap[:, 0].unsqueeze(1)) / (global_optimum_stacked - generality_gap[:, 0].unsqueeze(1))
                gap_value = gap_value.unsqueeze(-1)
                mean_gap = torch.mean(gap_value, dim=0)
                std_error_gap = torch.std(gap_value, dim=0) / np.sqrt(gap_value.size(0))
                y_mean = mean_gap.view(-1).numpy()
                y_ste = std_error_gap.view(-1).numpy()

                axs[0].plot(range(y_mean.shape[0]), y_mean, label=r'\textsc{' + campaigns[0].label + '}', color=cmap(COLOR_IDX[i]))
                axs[0].fill_between(range(y_mean.shape[0]), (y_mean - y_ste), (y_mean + y_ste), alpha=0.3, color=cmap(COLOR_IDX[i]))

                axs[0].set_ylabel(r'GAP ($\uparrow$)', size=10)

                axs[0].set_ylim([0, 1])

        axs[0].set_xlabel('Experiment number', size=10)
        axs[0].set_xticks(ticks=range(0, MAX_BUDGET + 1, 20))
        axs[0].set_xticks(range(0, MAX_BUDGET + 1, 10), minor=True)
        axs[0].xaxis.set_minor_locator(plt.MultipleLocator(10))
        axs[0].grid(which='both', linestyle='--', linewidth=0.1)
        axs[0].set_title("Mean Aggregation", size=10)

        axs[1].set_xlim([0, MAX_BUDGET])
        overall_max = 0
        for i, (label, campaigns) in enumerate(second_element.items()):
            generality_train_set = [campaign.generality_train_set for campaign in campaigns]
            generality_train_set = [tensor[:(MAX_BUDGET + 1), :] for tensor in generality_train_set]
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
                    axs[1].axhline(y=global_optimum_mean, color='black', linestyle = 'dashed', label = "Global optimum")
                    axs[1].fill_between(range(y_mean.shape[0]), (global_optimum_mean - global_optimum_ste), (global_optimum_mean + global_optimum_ste), alpha=0.3, color='black')
                    overall_max = max(overall_max, global_optimum_mean + global_optimum_ste)


                axs[1].plot(range(y_mean.shape[0]), y_mean, label=r'\textsc{' + campaigns[0].label + '}', color=cmap(i))
                axs[1].fill_between(range(y_mean.shape[0]), (y_mean - y_ste), (y_mean + y_ste), alpha=0.3, color=cmap(i))
                axs[1].set_ylabel(campaigns[0].ylabel if campaigns[0].ylabel is not None else 'Generality metric', size=10)

            else:

                global_optimum_set = [campaign.global_optimum for campaign in campaigns]
                global_optimum_stacked = torch.cat(global_optimum_set)
                generality_gap = generality_stacked.squeeze()
                diff = global_optimum_stacked - generality_gap[:, 0].unsqueeze(1)
                if torch.isclose(diff, torch.zeros_like(diff), atol=0.01).any():
                    zeros = torch.isclose(diff, torch.zeros_like(diff), atol=0.01).all(dim=1)
                    generality_gap = generality_gap[~zeros]
                    global_optimum_stacked = global_optimum_stacked[~zeros]

                gap_value = (generality_gap - generality_gap[:, 0].unsqueeze(1)) / (global_optimum_stacked - generality_gap[:, 0].unsqueeze(1))
                gap_value = gap_value.unsqueeze(-1)
                    
                mean_gap = torch.mean(gap_value, dim=0)
                std_error_gap = torch.std(gap_value, dim=0) / np.sqrt(gap_value.size(0))
                y_mean = mean_gap.view(-1).numpy()
                y_ste = std_error_gap.view(-1).numpy()

                axs[1].plot(range(y_mean.shape[0]), y_mean, color=cmap(COLOR_IDX[i]))
                axs[1].fill_between(range(y_mean.shape[0]), (y_mean - y_ste), (y_mean + y_ste), alpha=0.3, color=cmap(COLOR_IDX[i]))

                axs[1].set_ylim([0, 1])

        axs[1].set_xlabel('Experiment number', size=10)
        axs[1].set_xticks(ticks=range(0, MAX_BUDGET + 1, 20))
        axs[1].set_xticks(range(0, MAX_BUDGET + 1, 10), minor=True)
        axs[1].xaxis.set_minor_locator(plt.MultipleLocator(10))
        axs[1].grid(which='both', linestyle='--', linewidth=0.1)
        axs[1].set_title("Threshold Aggregation", size=10)

        fig.legend(loc="outside lower center", ncol=2)

        fig.savefig(file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Key to define analytical problem set.")
    parser.add_argument('--enhanced', action='store_true', default=True, help='Whether to use enhanced dataset')
    parser.add_argument('--not-enhanced', action='store_false', dest='enhanced', help='Do not use enhanced dataset')
    args = parser.parse_args()

    campaign_dict_overall = {}
    campaign_dict_overall["sequential_look"] = []
    campaign_dict_overall["sequential_complete"] = []
    campaign_dict_overall["sequential_look_complete"] = []
    campaign_dict_overall["sequential_look_complete_EI"] = []

    campaign_dict_overall_frac = {}
    campaign_dict_overall_frac["sequential_look"] = []
    campaign_dict_overall_frac["sequential_complete"] = []
    campaign_dict_overall_frac["sequential_look_complete"] = []
    campaign_dict_overall_frac["sequential_look_complete_EI"] = []

    campaign_dicts = []

    for dataset, ylabel in zip(["Denmark", "Deoxyfluorination", "Borylation", "Cernak"], [r"$\phi(f(\hat{\mathbf{x}}; \mathbf{w}), \mathcal{W})$ ($\uparrow$)", r"$\phi(f(\hat{\mathbf{x}}; \mathbf{w}), \mathcal{W})$ ($\uparrow$)", r"$\phi(f(\hat{\mathbf{x}}; \mathbf{w}), \mathcal{W})$ ($\uparrow$)", r"$\phi(f(\hat{\mathbf{x}}; \mathbf{w}), \mathcal{W})$ ($\uparrow$)"]):

        campaign_dict = {}
        campaign_dict["sequential_look"] = []
        campaign_dict["sequential_complete"] = []
        campaign_dict["sequential_look_complete"] = []
        campaign_dict["sequential_look_complete_EI"] = []

        campaign_labels = ["Seq 2LA-UCB-PV", "Seq 1LA-UCB-Complete", "Seq 2LA-UCB-Complete", "Seq 2LA-EI-Complete"]

        for i in range(30):

            for j, (key, val) in enumerate(campaign_dict.items()):

                path = f"benchmark_mean/{dataset}_{key}_{i}{'_enhance' if args.enhanced else ''}_mean.pt"

                campaign = GeneralBOCampaign()
                campaign_load = torch.load(path)
                campaign.observations_x = campaign_load['observations_x']
                campaign.observations_y = campaign_load['observations_y']
                campaign.observations_w = campaign_load['observations_w']
                campaign.observations_w_idx = campaign_load['observations_w_idx']
                campaign.optimum_x = campaign_load['optimum_x']
                campaign.global_optimum = campaign_load['global_optimum']
                campaign.generality_train_set = campaign_load['generality train set']
                campaign.generality_test_set = campaign_load['generality test set']
                campaign._problem_type = 'discrete'
                campaign.label = campaign_labels[j]
                campaign.ylabel = ylabel

                val.append(campaign)

        campaign_dicts.append(campaign_dict)

        for j, (key, val) in enumerate(campaign_dict_overall.items()):

            val.extend(campaign_dict[key])

    for dataset, ylabel in zip(["Denmark", "Deoxyfluorination", "Borylation", "Cernak"], [r"$\phi(f(\hat{\mathbf{x}}; \mathbf{w}), \mathcal{W})$ ($\uparrow$)", r"$\phi(f(\hat{\mathbf{x}}; \mathbf{w}), \mathcal{W})$ ($\uparrow$)", r"$\phi(f(\hat{\mathbf{x}}; \mathbf{w}), \mathcal{W})$ ($\uparrow$)", r"$\phi(f(\hat{\mathbf{x}}; \mathbf{w}), \mathcal{W})$ ($\uparrow$)"]):

        campaign_dict = {}
        campaign_dict["sequential_look"] = []
        campaign_dict["sequential_complete"] = []
        campaign_dict["sequential_look_complete"] = []
        campaign_dict["sequential_look_complete_EI"] = []

        campaign_labels = ["Seq 2LA-UCB-PV", "Seq 1LA-UCB-Complete", "Seq 2LA-UCB-Complete", "Seq 2LA-EI-Complete"]

        for i in range(30):

            for j, (key, val) in enumerate(campaign_dict.items()):

                path = f"benchmark_frac/{dataset}_{key}_{i}{'_enhance' if args.enhanced else ''}_frac.pt"

                campaign = GeneralBOCampaign()
                campaign_load = torch.load(path)
                campaign.observations_x = campaign_load['observations_x']
                campaign.observations_y = campaign_load['observations_y']
                campaign.observations_w = campaign_load['observations_w']
                campaign.observations_w_idx = campaign_load['observations_w_idx']
                campaign.optimum_x = campaign_load['optimum_x']
                campaign.global_optimum = campaign_load['global_optimum']
                campaign.generality_train_set = campaign_load['generality train set']
                campaign.generality_test_set = campaign_load['generality test set']
                campaign._problem_type = 'discrete'
                campaign.label = campaign_labels[j]
                campaign.ylabel = ylabel


                val.append(campaign)

        campaign_dicts.append(campaign_dict)

        for j, (key, val) in enumerate(campaign_dict_overall_frac.items()):

            val.extend(campaign_dict[key])

    for j, dataset in enumerate(["Denmark", "Deoxyfluorination", "Borylation", "Cernak"]):
        create_plot(first_element = campaign_dicts[j], second_element = campaign_dicts[j + 4], file_name =f"complete_{dataset}{'_enhance' if args.enhanced else ''}.pdf", gap = True, MAX_BUDGET = 100)

    create_plot(first_element = campaign_dict_overall, second_element = campaign_dict_overall_frac, file_name = f"complete{'_enhance' if args.enhanced else ''}.pdf", gap = True, MAX_BUDGET = 100)
