"""Plot configuration for SLS datasets.
"""
from copy import deepcopy

import numpy as np

from experiment_utils import utils
from experiment_utils.plotting import defaults
from experiment_utils.plotting.plot_cell import make_error_bar_plot

from plot_configs import constants
from scaffold.uci_names import UCI_BIN_SUBSET

# plot configuration #

# CONSTANTS #

row_key = ("data", "name")

repeat_key = ("repeat",)

metrics = [
    "train_objective",
    "test_nc_accuracy",
]


def line_key_gen(exp_dict):
    """Load line key."""
    method = exp_dict["method"]
    key = method["name"]

    if "torch" in key:
        step_size = method["step_size"]
        key = key + "_" + str(step_size)

    return key


processing_fns = []

figure_labels = {
    "x_labels": {
        "train_objective": "Time (S)",
        "test_nc_accuracy": "Time (S)",
    },
    "y_labels": {},
    "col_titles": {
        "train_objective": "Training Objective",
        "test_nc_accuracy": "Test Accuracy",
    },
    "row_titles": {},
}

limits = {
    "train_objective": ([0, 1000], None),
    "test_nc_accuracy": ([0, 1000], [0.8, 1.01]),
}

settings = defaults.DEFAULT_SETTINGS
settings["y_labels"] = "every_col"
settings["x_labels"] = "bottom_row"
settings["legend_cols"] = 4
settings["show_legend"] = True


datasets_to_plot = [
    "mnist",
]

# complicated filtering for plots.

plots = []

for dataset_name in datasets_to_plot:

    keep = [
        (("data", "name"), [dataset_name]),
        # (("method", "step_size"), [1, 0.1, 0.01]),
        # (("method", "name"), ["torch_sgd", "fista"]),
        # (("model", "regularizer", "lambda"), lambda_to_try),
    ]
    remove = []

    line_key = line_key_gen

    def filter_fn(exp_config):
        model = exp_config["model"]

        method = exp_config["method"]

        keep = True

        if method["name"] == "torch_adam":
            keep = method["step_size"] == 0.1
        elif method["name"] == "torch_sgd":
            keep = method["step_size"] == 10

        if "hidden_layers" in model:
            return keep and model["hidden_layers"][0]["name"] == "gated_relu"
        else:
            return keep

    line_kwargs = {
        "fista": {
            "c": constants.line_colors[0],
            "label": "Convex",
            "linewidth": 3,
            "marker": "v",
            "markevery": 0.1,
            "markersize": 8,
        },
        "torch_adam_0.1": {
            "c": constants.line_colors[1],
            "label": "Adam",
            "linewidth": 3,
            "marker": "D",
            "markevery": 0.1,
            "markersize": 8,
        },
        "torch_sgd_10": {
            "c": constants.line_colors[2],
            "label": "SGD",
            "linewidth": 3,
            "marker": "X",
            "markevery": 0.1,
            "markersize": 8,
        },
    }

    # figure_labels["title"] = dataset_name
    figure_labels = deepcopy(figure_labels)

    plot_config = deepcopy(
        {
            "name": dataset_name,
            "row_key": row_key,
            "metrics": metrics,
            "line_key": line_key,
            "repeat_key": repeat_key,
            "x_key": "time",
            "metrics_fn": utils.quantile_metrics,
            "keep": keep,
            "remove": remove,
            "filter_fn": filter_fn,
            "processing_fns": processing_fns,
            "figure_labels": figure_labels,
            "line_kwargs": line_kwargs,
            "log_scale": constants.log_scale,
            "limits": limits,
            "settings": settings,
        }
    )
    plots.append(plot_config)

PLOTS = {"mnist_timing": plots}
