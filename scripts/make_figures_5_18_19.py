from typing import List, Dict
import os
from functools import reduce
from math import floor

import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from experiment_utils import configs, utils, files, command_line
from experiment_utils.plotting import plot_grid, plot_cell, defaults

from plot_configs import constants  # type: ignore
from exp_configs import EXPERIMENTS  # type: ignore
from scaffold.uci_names import HYPERPLANE_ABLATIONS

# plot configuration #

# CONSTANTS #

marker_spacing = 0.2
marker_size = 10
line_width = 5


def line_key_gen(exp_dict):
    """Load line key."""
    method = exp_dict["method"]
    model = exp_dict["model"]
    n_samples = model["sign_patterns"]["n_samples"]
    key = method["name"]

    return f"{key}_{n_samples}"


processing_fns = [
    utils.drop_start(5, lambda key: True),
    utils.cum_sum(lambda key: key[1] == "time"),
]

lambda_to_try = (
    np.logspace(-6, -2, 20).tolist() + np.logspace(-2, 0, 10).tolist()[1:]
)
figure_labels = {
    "x_labels": {},
    "y_labels": {},
    "col_titles": {},
    "row_titles": {},
}

settings = defaults.DEFAULT_SETTINGS
settings["y_labels"] = "every_col"
settings["x_labels"] = "bottom_row"
settings["legend_cols"] = 6
settings["fig_width"] = 6
settings["fig_height"] = 6
settings["titles_fs"] = 22
settings["axis_labels_fs"] = 20
settings["legend_fs"] = 13
settings["tick_fs"] = 12
settings["wspace"] = 0.18
settings["line_alpha"] = 0.8


line_kwargs = {
    "augmented_lagrangian_10": {
        "c": constants.line_colors[5],
        "label": "10 Patterns",
        "linestyle": "-",
        "linewidth": line_width,
        "marker": "X",
        "markersize": marker_size,
        "markevery": marker_spacing,
    },
    "augmented_lagrangian_100": {
        "c": constants.line_colors[1],
        "label": "100 Patterns",
        "linestyle": "-",
        "linewidth": line_width,
        "marker": "D",
        "markersize": marker_size,
        "markevery": marker_spacing,
    },
    "augmented_lagrangian_1000": {
        "c": constants.line_colors[9],
        "label": "1000 Patterns",
        "linestyle": "-",
        "linewidth": line_width,
        "marker": "s",
        "markersize": marker_size,
        "markevery": marker_spacing,
    },
    "fista_10": {
        "c": constants.line_colors[5],
        "label": "10 Patterns",
        "linestyle": "-",
        "linewidth": line_width,
        "marker": "X",
        "markersize": marker_size,
        "markevery": marker_spacing,
    },
    "fista_100": {
        "c": constants.line_colors[1],
        "label": "100 Patterns",
        "linestyle": "-",
        "linewidth": line_width,
        "marker": "D",
        "markersize": marker_size,
        "markevery": marker_spacing,
    },
    "fista_1000": {
        "c": constants.line_colors[9],
        "label": "1000 Patterns",
        "linestyle": "-",
        "linewidth": line_width,
        "marker": "s",
        "markersize": marker_size,
        "markevery": marker_spacing,
    },
}

row_key = ("data", "name")


def repeat_key(config):
    repeat_key = config["model"]["sign_patterns"]["seed"]
    lam = config["model"]["regularizer"]["lambda"]

    return (lam, repeat_key)


metrics = [
    "train_objective",
    "train_grad_norm",
    "train_accuracy",
    "test_accuracy",
    "group_sparsity",
    "time",
    "active_neurons",
]


processing_fns = [
    utils.drop_start(0, lambda key: True),
    utils.cum_sum(lambda key: key[1] == "time"),
]

exp_ids = ["figure_5"]

config_list: List[Dict] = reduce(
    lambda acc, eid: acc + EXPERIMENTS[eid], exp_ids, []
)
results_dir = [os.path.join("results", eid) for eid in exp_ids]

metric_grid = files.load_and_clean_experiments(
    config_list,
    results_dir,
    metrics,
    row_key,
    line_key_gen,
    repeat_key,
    utils.quantile_metrics,
    keep=[],
    remove=[],
    filter_fn=None,
    transform_fn=utils.replace_x_axis,
    processing_fns=processing_fns,
    x_vals=lambda_to_try,
)

x_limits = {
    "primary-tumor": [1e-6, 2e-3],
    "adult": [],
    "chess-krvkp": [],
    "hepatitis": [],
    "heart-va": [],
    "synthetic-control": [],
    "statlog-heart": [],
    "monks-3": [],
    "yeast": [],
    "nursery": [],
}

paper_dataset = "primary-tumor"

appendix_datasets = [
    "adult",
    "chess-krvkp",
    "hepatitis",
    "heart-va",
    "synthetic-control",
    "statlog-heart",
    "monks-3",
    "yeast",
    "nursery",
]

# === Plot for the Main Paper === #

line_width = 8
marker_size = 14

for line in line_kwargs.values():
    line["linewidth"] = line_width
    line["markersize"] = marker_size

fig = plt.figure(figsize=(6, 6))
spec = fig.add_gridspec(ncols=2, nrows=2)
ax0 = fig.add_subplot(spec[0, 0])

gated_lines = {
    key: val
    for (key, val) in metric_grid[paper_dataset]["test_accuracy"].items()
    if "fista" in key
}
plot_cell.make_convergence_plot(ax0, gated_lines, line_kwargs, settings)
ax0.set_xscale("log")
ax0.set_xlim(x_limits["primary-tumor"])
ax0.set_title("Test Accuracy", fontsize=settings["titles_fs"])
ax0.tick_params(labelsize=settings["tick_fs"])
ax0.set_ylabel("Gated ReLU", fontsize=settings["axis_labels_fs"])

ax1 = fig.add_subplot(spec[0, 1])

gated_lines = {
    key: val
    for (key, val) in metric_grid[paper_dataset]["active_neurons"].items()
    if "fista" in key
}
plot_cell.make_convergence_plot(ax1, gated_lines, line_kwargs, settings)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(x_limits["primary-tumor"])
ax1.set_title("Active Neurons", fontsize=settings["titles_fs"])
ax1.tick_params(labelsize=settings["tick_fs"])

ax3 = fig.add_subplot(spec[1, 0])

relu_lines = {
    key: val
    for (key, val) in metric_grid[paper_dataset]["test_accuracy"].items()
    if "augmented_lagrangian" in key
}
plot_cell.make_convergence_plot(ax3, relu_lines, line_kwargs, settings)
ax3.set_xscale("log")
ax3.set_xlim(x_limits["primary-tumor"])
ax3.tick_params(labelsize=settings["tick_fs"])
ax3.set_xlabel("Regularization", fontsize=settings["axis_labels_fs"])
ax3.set_ylabel("ReLU", fontsize=settings["axis_labels_fs"])

ax4 = fig.add_subplot(spec[1, 1])

relu_lines = {
    key: val
    for (key, val) in metric_grid[paper_dataset]["active_neurons"].items()
    if "augmented_lagrangian" in key
}
plot_cell.make_convergence_plot(ax4, relu_lines, line_kwargs, settings)
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xlim(x_limits["primary-tumor"])
ax4.tick_params(labelsize=settings["tick_fs"])
ax4.set_xlabel("Regularization", fontsize=settings["axis_labels_fs"])

handles, labels = ax4.get_legend_handles_labels()
legend = fig.legend(
    handles=handles,
    labels=labels,
    loc="lower center",
    borderaxespad=0.1,
    fancybox=False,
    shadow=False,
    ncol=3,
    fontsize=settings["legend_fs"],
    frameon=False,
)

plt.tight_layout()
fig.subplots_adjust(
    wspace=settings["wspace"], hspace=settings["vspace"], bottom=0.16
)
plt.savefig("figures/figure_5.pdf")


# === Plot for the Appendix === #

# make plots for the appendix

settings["titles_fs"] = 32
settings["axis_labels_fs"] = 26
settings["legend_fs"] = 30
settings["tick_fs"] = 20
settings["wspace"] = 0.18
settings["vspace"] = 0.3
settings["line_alpha"] = 0.8
bottom = 0.15

fig = plt.figure(figsize=(18, 12))
spec = fig.add_gridspec(ncols=3, nrows=3)

axes = []

x_lims = {
    "adult": [None, 5e-3],
    "chess-krvkp": [None, 5e-2],
    "hepatitis": [None, 1e-1],
    "heart-va": [None, 1e-2],
    "synthetic-control": [None, 5e-2],
    "statlog-heart": [None, 1e-1],
    "monks-3": [None, 1e-1],
    "yeast": [None, 1e-3],
    "nursery": [None, 1e-3],
}

for i, dataset in enumerate(appendix_datasets):
    gated_lines = {
        key: val
        for (key, val) in metric_grid[dataset]["test_accuracy"].items()
        if "fista" in key
    }

    row = int(floor(i / 3))
    col = i % 3
    axi = fig.add_subplot(spec[row, col])
    axes.append(axi)

    plot_cell.make_convergence_plot(axi, gated_lines, line_kwargs, settings)
    axi.set_xscale("log")

    axi.tick_params(labelsize=settings["tick_fs"])
    axi.set_title(dataset, fontsize=settings["titles_fs"])
    axi.set_xlim(x_lims[dataset])
    if row == 2:
        axi.set_xlabel(
            "Regularization ($\\lambda$)", fontsize=settings["axis_labels_fs"]
        )

    if col == 0:
        axi.set_ylabel("Test Accuracy", fontsize=settings["axis_labels_fs"])

handles, labels = axes[-1].get_legend_handles_labels()

legend = fig.legend(
    handles=handles,
    labels=labels,
    loc="lower center",
    borderaxespad=0.1,
    fancybox=False,
    shadow=False,
    ncol=len(handles),
    fontsize=settings["legend_fs"],
    frameon=False,
)

plt.tight_layout()

fig.subplots_adjust(
    wspace=settings["wspace"],
    hspace=settings["vspace"],
    bottom=bottom,
)
plt.savefig("figures/figure_7.pdf")


fig = plt.figure(figsize=(18, 12))
spec = fig.add_gridspec(ncols=3, nrows=3)

axes = []

x_lims = {
    "adult": [None, 1e-2],
    "chess-krvkp": [None, 5e-2],
    "hepatitis": [None, 1e-1],
    "heart-va": [None, 1e-2],
    "synthetic-control": [None, 5e-2],
    "statlog-heart": [None, 1e-1],
    "monks-3": [None, 1e-1],
    "yeast": [None, 1e-3],
    "nursery": [None, 1e-3],
}

for i, dataset in enumerate(appendix_datasets):
    gated_lines = {
        key: val
        for (key, val) in metric_grid[dataset]["test_accuracy"].items()
        if "lagrangian" in key
    }

    row = int(floor(i / 3))
    col = i % 3
    axi = fig.add_subplot(spec[row, col])
    axes.append(axi)

    plot_cell.make_convergence_plot(axi, gated_lines, line_kwargs, settings)
    axi.set_xscale("log")

    axi.tick_params(labelsize=settings["tick_fs"])
    axi.set_title(dataset, fontsize=settings["titles_fs"])
    axi.set_xlim(x_lims[dataset])
    if row == 2:
        axi.set_xlabel(
            "Regularization ($\\lambda$)", fontsize=settings["axis_labels_fs"]
        )

    if col == 0:
        axi.set_ylabel("Test Accuracy", fontsize=settings["axis_labels_fs"])

handles, labels = axes[-1].get_legend_handles_labels()

legend = fig.legend(
    handles=handles,
    labels=labels,
    loc="lower center",
    borderaxespad=0.1,
    fancybox=False,
    shadow=False,
    ncol=len(handles),
    fontsize=settings["legend_fs"],
    frameon=False,
)

plt.tight_layout()

fig.subplots_adjust(
    wspace=settings["wspace"],
    hspace=settings["vspace"],
    bottom=bottom,
)
plt.savefig("figures/figure_8.pdf")
