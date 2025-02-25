import torch
import pub_ready_plots
from pypalettes import load_cmap
from currybo.campaign import GeneralBOCampaign
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from adjustText import adjust_text

import warnings 
warnings.filterwarnings('ignore') 

def nanstd(x):
    return torch.sqrt(torch.nanmean(torch.pow(x - torch.nanmean(x, dim=0), 2), dim = 0))

def create_plot_all(first_element, second_element, third_element, fourth_element, fifth_element, sixth_element, seventh_element, eigth_element, file_name, generality_y = [0, 1], frac = False):

    with pub_ready_plots.get_context(
        width_frac=1,  # between 0 and 1
        height_frac=0.5,  # between 0 and 1
        layout=pub_ready_plots.Layout.ICML,  # or "iclr", "neurips", "poster-portrait", "poster-landscape"
        single_col=(not frac),  # only works for the "icml" layout
        nrows=4,  # depending on your subplots, default = 1
        ncols=2,  # depending on your subplots, default = 1
        override_rc_params={"lines.linewidth": 1.2, "text.usetex": True},  # Overriding rcParams
        sharey=True,  # Additional keyword args for `plt.subplots`
    ) as (fig, axs):
        cmap = load_cmap("Klein")

        mean = first_element.nanmean(dim=0)
        ste = nanstd(first_element) / torch.sqrt(torch.tensor(first_element.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[0, 0].errorbar(torch.arange(1, first_element.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(0),
                    ecolor = 'black',
                    elinewidth = 1.5,
                    capsize = 2,
                    capthick = 1.5,
                    marker=".",
                    label ="Pd-catalyzed coupling"
        )
        axs[0, 0].set_xticks(ticks=range(0, first_element.shape[1] + 1, 2))
        axs[0, 0].set_xticks(range(0, first_element.shape[1] + 1, 1), minor=True)
        axs[0, 0].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[0, 0].grid(True)
        axs[0, 0].set_ylim(generality_y[0])
        axs[0, 0].set_title("Original Search Space", size=9)
        spearmanr = stats.spearmanr(torch.arange(1, first_element.shape[1] + 1), mean)
        texts = [axs[0, 0].text(0.95, 0.15, rf"Spearman's $\rho$: {spearmanr.statistic:.2f}", transform=axs[0, 0].transAxes, fontsize=7, va='bottom', ha='center'), axs[0, 0].text(0.95, 0.05, rf"Max. score: {torch.max(mean):.2f}", transform=axs[0, 0].transAxes, fontsize=7, va='bottom', ha='center')]
        adjust_text(texts, ax=axs[0, 0])

        mean = second_element.mean(dim=0)
        ste = nanstd(second_element) / torch.sqrt(torch.tensor(second_element.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[0, 1].errorbar(torch.arange(1, second_element.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(0),
                    ecolor = 'black',
                    elinewidth = 1.5,
                    capsize = 2,
                    capthick = 1.5,
                    marker="."
        )
        axs[0, 1].set_xticks(ticks=range(0, second_element.shape[1] + 1, 2))
        axs[0, 1].set_xticks(range(0, second_element.shape[1] + 1, 1), minor=True)
        axs[0, 1].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[0, 1].grid(True)
        axs[0, 1].set_ylim(generality_y[0])
        axs[0, 1].set_title("Augmented Search Space", size=9)
        spearmanr = stats.spearmanr(torch.arange(1, second_element.shape[1] + 1), mean)
        texts = [axs[0, 1].text(0.95, 0.15, rf"Spearman's $\rho$: {spearmanr.statistic:.2f}", transform=axs[0, 1].transAxes, fontsize=7, va='bottom', ha='center'), axs[0, 1].text(0.95, 0.05, rf"Max. score: {torch.max(mean):.2f}", transform=axs[0, 1].transAxes, fontsize=7, va='bottom', ha='center')]
        adjust_text(texts, ax=axs[0, 1])

        mean = third_element.nanmean(dim=0)
        ste = nanstd(third_element) / torch.sqrt(torch.tensor(third_element.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[1, 0].errorbar(torch.arange(1, third_element.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(1),
                    ecolor = 'black',
                    elinewidth = 1.5,
                    capsize = 2,
                    capthick = 1.5,
                    marker=".",
                    label = "N,S-Acetal formation"
        )
        axs[1, 0].set_xticks(ticks=range(0, third_element.shape[1] + 1, 2))
        axs[1, 0].set_xticks(range(0, third_element.shape[1] + 1, 1), minor=True)
        axs[1, 0].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[1, 0].grid(True)
        axs[1, 0].set_ylim(generality_y[1])
        x = torch.arange(1, third_element.shape[1] + 1)
        x = x[~torch.isnan(mean)]
        mean = mean[~torch.isnan(mean)]
        spearmanr = stats.spearmanr(x, mean)
        texts = [axs[1, 0].text(0.95, 0.15, rf"Spearman's $\rho$: {spearmanr.statistic:.2f}", transform=axs[1, 0].transAxes, fontsize=7, va='bottom', ha='center'), axs[1, 0].text(0.95, 0.05, rf"Max. score: {torch.max(mean):.2f}", transform=axs[1, 0].transAxes, fontsize=7, va='bottom', ha='center')]
        adjust_text(texts, ax=axs[1, 0])

        mean = fourth_element.mean(dim=0)
        ste = nanstd(fourth_element) / torch.sqrt(torch.tensor(fourth_element.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[1, 1].errorbar(torch.arange(1, fourth_element.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(1),
                    ecolor = 'black',
                    elinewidth = 1.5,
                    capsize = 2,
                    capthick = 1.5,
                    marker="."
        )
        axs[1, 1].set_xticks(ticks=range(0, fourth_element.shape[1] + 1, 2))
        axs[1, 1].set_xticks(range(0, fourth_element.shape[1] + 1, 1), minor=True)
        axs[1, 1].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[1, 1].grid(True)
        axs[1, 1].set_ylim(generality_y[1])
        x = torch.arange(1, fourth_element.shape[1] + 1)
        x = x[~torch.isnan(mean)]
        mean = mean[~torch.isnan(mean)]
        spearmanr = stats.spearmanr(x, mean)
        texts = [axs[1, 1].text(0.95, 0.15, rf"Spearman's $\rho$: {spearmanr.statistic:.2f}", transform=axs[1, 1].transAxes, fontsize=7, va='bottom', ha='center'), axs[1, 1].text(0.95, 0.05, rf"Max. score: {torch.max(mean):.2f}", transform=axs[1, 1].transAxes, fontsize=7, va='bottom', ha='center')]
        adjust_text(texts, ax=axs[1, 1])

        mean = fifth_element.nanmean(dim=0)
        ste = nanstd(fifth_element) / torch.sqrt(torch.tensor(fifth_element.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[2, 0].errorbar(torch.arange(1, fifth_element.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(2),
                    ecolor = 'black',
                    elinewidth = 1.5,
                    capsize = 2,
                    capthick = 1.5,
                    marker=".",
                    label = "Borylation reaction"
        )
        axs[2, 0].set_xticks(ticks=range(0, fifth_element.shape[1] + 1, 4))
        axs[2, 0].set_xticks(range(0, fifth_element.shape[1] + 1, 2), minor=True)
        axs[2, 0].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[2, 0].grid(True)
        axs[2, 0].set_ylim(generality_y[2])
        spearmanr = stats.spearmanr(torch.arange(1, fifth_element.shape[1] + 1), mean)
        if frac:
            texts = [axs[2, 0].text(0.95, 0.85, rf"Spearman's $\rho$: {spearmanr.statistic:.2f}", transform=axs[2, 0].transAxes, fontsize=7, va='top', ha='center'), axs[2, 0].text(0.95, 0.65, rf"Max. score: {torch.max(mean):.2f}", transform=axs[2, 0].transAxes, fontsize=7, va='top', ha='center')]
        else:
            texts = [axs[2, 0].text(0.95, 0.15, rf"Spearman's $\rho$: {spearmanr.statistic:.2f}", transform=axs[2, 0].transAxes, fontsize=7, va='bottom', ha='center'), axs[2, 0].text(0.95, 0.05, rf"Max. score: {torch.max(mean):.2f}", transform=axs[2, 0].transAxes, fontsize=7, va='bottom', ha='center')]
        adjust_text(texts, ax=axs[2, 0])

        mean = sixth_element.mean(dim=0)
        ste = nanstd(sixth_element) / torch.sqrt(torch.tensor(sixth_element.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[2, 1].errorbar(torch.arange(1, sixth_element.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(2),
                    ecolor = 'black',
                    elinewidth = 1.5,
                    capsize = 2,
                    capthick = 1.5,
                    marker="."
        )
        axs[2, 1].set_xticks(ticks=range(0, sixth_element.shape[1] + 1, 4))
        axs[2, 1].set_xticks(range(0, sixth_element.shape[1] + 1, 2), minor=True)
        axs[2, 1].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[2, 1].grid(True)
        axs[2, 1].set_ylim(generality_y[2])
        spearmanr = stats.spearmanr(torch.arange(1, sixth_element.shape[1] + 1), mean)
        if frac:
            texts = [axs[2, 1].text(0.95, 0.85, rf"Spearman's $\rho$: {spearmanr.statistic:.2f}", transform=axs[2, 1].transAxes, fontsize=7, va='top', ha='center'), axs[2, 1].text(0.95, 0.65, rf"Max. score: {torch.max(mean):.2f}", transform=axs[2, 1].transAxes, fontsize=7, va='top', ha='center')]
        else:
            texts = [axs[2, 1].text(0.95, 0.15, rf"Spearman's $\rho$: {spearmanr.statistic:.2f}", transform=axs[2, 1].transAxes, fontsize=7, va='bottom', ha='center'), axs[2, 1].text(0.95, 0.05, rf"Max. score: {torch.max(mean):.2f}", transform=axs[2, 1].transAxes, fontsize=7, va='bottom', ha='center')]
        adjust_text(texts, ax=axs[2, 1])

        mean = seventh_element.nanmean(dim=0)
        ste = nanstd(seventh_element) / torch.sqrt(torch.tensor(seventh_element.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[3, 0].errorbar(torch.arange(1, seventh_element.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(3),
                    ecolor = 'black',
                    elinewidth = 1.5,
                    capsize = 2,
                    capthick = 1.5,
                    marker=".",
                    label = "Deoxyfluorination reaction"
        )
        axs[3, 0].set_xlabel("Number of substrates", size = 10)
        axs[3, 0].set_xticks(ticks=range(0, seventh_element.shape[1] + 1, 4))
        axs[3, 0].set_xticks(range(0, seventh_element.shape[1] + 1, 2), minor=True)
        axs[3, 0].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[3, 0].grid(True)
        axs[3, 0].set_ylim(generality_y[3])
        spearmanr = stats.spearmanr(torch.arange(1, seventh_element.shape[1] + 1), mean)
        texts = [axs[3, 0].text(0.95, 0.15, rf"Spearman's $\rho$: {spearmanr.statistic:.2f}", transform=axs[3, 0].transAxes, fontsize=7, va='bottom', ha='center'), axs[3, 0].text(0.95, 0.05, rf"Max. score: {torch.max(mean):.2f}", transform=axs[3, 0].transAxes, fontsize=7, va='bottom', ha='center')]
        adjust_text(texts, ax=axs[3, 0])

        mean = eigth_element.mean(dim=0)
        ste = nanstd(eigth_element) / torch.sqrt(torch.tensor(eigth_element.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[3, 1].errorbar(torch.arange(1, eigth_element.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(3),
                    ecolor = 'black',
                    elinewidth = 1.5,
                    capsize = 2,
                    capthick = 1.5,
                    marker=".",
        )
        axs[3, 1].set_xlabel("Number of substrates", size = 10)
        axs[3, 1].set_xticks(ticks=range(0, eigth_element.shape[1] + 1, 4))
        axs[3, 1].set_xticks(range(0, eigth_element.shape[1] + 1, 2), minor=True)
        axs[3, 1].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[3, 1].grid(True)
        axs[3, 1].set_ylim(generality_y[3])
        spearmanr = stats.spearmanr(torch.arange(1, eigth_element.shape[1] + 1), mean)
        texts = [axs[3, 1].text(0.95, 0.15, rf"Spearman's $\rho$: {spearmanr.statistic:.2f}", transform=axs[3, 1].transAxes, fontsize=7, va='bottom', ha='center'), axs[3, 1].text(0.95, 0.05, rf"Max. score: {torch.max(mean):.2f}", transform=axs[3, 1].transAxes, fontsize=7, va='bottom', ha='center')]
        adjust_text(texts, ax=axs[3, 1])

        labels = []
        handles = []

        for ax in [axs[0, 0], axs[1, 0], axs[2, 0], axs[3, 0]]:
            handle, label = ax.get_legend_handles_labels()
            handles.extend(handle)
            labels.extend(label)
        
        fig.legend(loc="outside lower center", ncol=2)

        fig.supylabel(r'Generality score ($\uparrow$)', size=12)

        fig.savefig(file_name)

        plt.clf()

def create_plot_method(first_element_random, first_element_fps, first_element_smart, second_element_random, second_element_fps, second_element_smart, file_name, generality_y = [0, 1]):

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
        cmap = load_cmap("Klein")

        mean = first_element_random.nanmean(dim=0)
        ste = nanstd(first_element_random) / torch.sqrt(torch.tensor(first_element_random.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[0].errorbar(torch.arange(1, first_element_random.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(0),
                    ecolor = cmap(0),
                    elinewidth = 1,
                    capsize = 1,
                    capthick = 1,
                    marker=".",
                    label="Random Sampling",
        )

        mean = first_element_fps.nanmean(dim=0)
        ste = nanstd(first_element_fps) / torch.sqrt(torch.tensor(first_element_fps.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[0].errorbar(torch.arange(1, first_element_fps.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(1),
                    ecolor = cmap(1),
                    elinewidth = 1,
                    capsize = 1,
                    capthick = 1,
                    marker=".",
                    label="Farthest Point Sampling",
        )

        mean = first_element_smart.nanmean(dim=0)
        ste = nanstd(first_element_smart) / torch.sqrt(torch.tensor(first_element_smart.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[0].errorbar(torch.arange(1, first_element_smart.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(2),
                    ecolor = cmap(2),
                    elinewidth = 1,
                    capsize = 1,
                    capthick = 1,
                    marker=".",
                    label="Average Sampling",
        )

        axs[0].set_ylabel(r'Generality score ($\uparrow$)', size=7)
        axs[0].set_xlabel("Number of substrates", size = 10)
        axs[0].set_xticks(ticks=range(0, first_element_random.shape[1] + 1, 2))
        axs[0].set_xticks(range(0, first_element_random.shape[1] + 1, 1), minor=True)
        axs[0].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[0].grid(True)
        axs[0].set_ylim(generality_y)
        axs[0].set_title("Original Search Space", size=10)

        mean = second_element_random.nanmean(dim=0)
        ste = nanstd(second_element_random) / torch.sqrt(torch.tensor(second_element_random.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[1].errorbar(torch.arange(1, second_element_random.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(0),
                    ecolor = cmap(0),
                    elinewidth = 1,
                    capsize = 1,
                    capthick = 1,
                    marker=".",
        )

        mean = second_element_fps.nanmean(dim=0)
        ste = nanstd(second_element_fps) / torch.sqrt(torch.tensor(second_element_fps.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[1].errorbar(torch.arange(1, second_element_fps.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(1),
                    ecolor = cmap(1),
                    elinewidth = 1,
                    capsize = 1,
                    capthick = 1,
                    marker=".",
        )

        mean = second_element_smart.nanmean(dim=0)
        ste = nanstd(second_element_smart) / torch.sqrt(torch.tensor(second_element_smart.shape[0], dtype=torch.float32), )
        mean = mean.masked_fill(mean == 0, torch.nan)
        ste = ste.masked_fill(ste == 0, torch.nan)
        axs[1].errorbar(torch.arange(1, second_element_smart.shape[1] + 1),
                    mean, 
                    yerr=ste,
                    fmt='o',
                    color = cmap(2),
                    ecolor = cmap(2),
                    elinewidth = 1,
                    capsize = 1,
                    capthick = 1,
                    marker=".",
        )
        axs[1].set_xlabel("Number of substrates", size = 10)
        axs[1].set_xticks(ticks=range(0, second_element_random.shape[1] + 1, 2))
        axs[1].set_xticks(range(0, second_element_random.shape[1] + 1, 1), minor=True)
        axs[1].xaxis.set_minor_locator(plt.MultipleLocator(1))
        axs[1].grid(True)
        axs[1].set_ylim(generality_y)
        axs[1].set_title("Augmented Search Space", size=10)

        labels = []
        handles = []

        for ax in [axs[0]]:
            handle, label = ax.get_legend_handles_labels()
            handles.extend(handle)
            labels.extend(label)
        
        fig.legend(loc="outside lower center", ncol=3)

        fig.savefig(file_name)

        plt.clf()

def transform_generality_list(generality_list):
    output = []
    for bounds in generality_list:
        lowers = [bound[0] for bound in bounds]
        uppers = [bound[1] for bound in bounds]

        min_lower = min(lowers)
        max_upper = max(uppers)

        output.append(tuple([[min_lower, max_upper] for _ in bounds]))

    return output

datasets = ["Cernak", "Denmark", "Borylation", "Deoxyfluorination"]
methods = ["mean", "frac"]
generality_y_list = [([0.5, 1], [0.92, 0.1], [0.8, 1], [0.7, 1]), ([0.4, 1], [0.9, 1], [0, 1], [0.18, 1])]
generality_y_list_methods = [([0.5, 1], [0.4, 1]), ([0.88, 1], [0.88, 1]), ([0.75, 1], [0, 1]), ([0.65, 1], [0.18, 1])]
generality_y_list = transform_generality_list(generality_y_list)
TRANSFORM_GENERALITY = False

for j, method in enumerate(methods):

    score_matrix_collected = []
    score_matrix_augmented_collected = []
        
    for i, dataset in enumerate(datasets):
    
        fname = 'dataset_analysis_' + dataset + '_scaled_' + method + '.pt'
        score_matrix = torch.load(fname)

        fname = 'dataset_analysis_' + dataset + '_augmented_scaled_' + method + '.pt'
        score_matrix_augmented = torch.load(fname)

        fname = 'dataset_analysis_' + dataset + '_scaled_' + method + '_fps.pt'
        score_matrix_fps = torch.load(fname)

        fname = 'dataset_analysis_' + dataset + '_augmented_scaled_' + method + '_fps.pt'
        score_matrix_augmented_fps = torch.load(fname)

        fname = 'dataset_analysis_' + dataset + '_scaled_' + method + '_smart.pt'
        score_matrix_smart = torch.load(fname)

        fname = 'dataset_analysis_' + dataset + '_augmented_scaled_' + method + '_smart.pt'
        score_matrix_augmented_smart = torch.load(fname)

        score_matrix = torch.where(score_matrix == 0, torch.tensor(float('nan')), score_matrix.float())
        score_matrix_augmented = torch.where(score_matrix_augmented == 0, torch.tensor(float('nan')), score_matrix_augmented.float())
        score_matrix_fps = torch.where(score_matrix_fps == 0, torch.tensor(float('nan')), score_matrix_fps.float())
        score_matrix_augmented_fps = torch.where(score_matrix_augmented_fps == 0, torch.tensor(float('nan')), score_matrix_augmented_fps.float())
        score_matrix_smart = torch.where(score_matrix_smart == 0, torch.tensor(float('nan')), score_matrix_smart.float())
        score_matrix_augmented_smart = torch.where(score_matrix_augmented_smart == 0, torch.tensor(float('nan')), score_matrix_augmented_smart.float())

        score_matrix_collected.append(score_matrix)
        score_matrix_augmented_collected.append(score_matrix_augmented)

        create_plot_method(
            first_element_random=score_matrix,
            first_element_fps=score_matrix_fps,
            first_element_smart=score_matrix_smart,
            second_element_random=score_matrix_augmented,
            second_element_fps=score_matrix_augmented_fps,
            second_element_smart=score_matrix_augmented_smart,
            file_name='dataset_analysis_' + dataset + '_sampling_' + method + '.pdf',
            generality_y=generality_y_list_methods[i][j]
        )

    create_plot_all(
        first_element=score_matrix_collected[0],
        second_element=score_matrix_augmented_collected[0],
        third_element=score_matrix_collected[1],
        fourth_element=score_matrix_augmented_collected[1],
        fifth_element=score_matrix_collected[2],
        sixth_element=score_matrix_augmented_collected[2],
        seventh_element=score_matrix_collected[3],
        eigth_element=score_matrix_augmented_collected[3],     
        file_name='dataset_analysis_' + method + '.pdf',
        generality_y=generality_y_list[j] if not TRANSFORM_GENERALITY else transform_generality_list(generality_y_list[j]),
        frac = ("frac" in method)
    )