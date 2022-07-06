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
    "train_grad_norm",
    "train_nc_accuracy",
    "test_nc_accuracy",
    "time",
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
    "x_labels": constants.x_labels,
    "y_labels": {},
    "col_titles": constants.y_labels,
    "row_titles": {},
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
        (("method", "step_size"), [1, 0.1, 0.01]),
        (("method", "name"), ["torch_adam", "fista"]),
        # (("model", "regularizer", "lambda"), lambda_to_try),
    ]
    remove = []

    line_key = line_key_gen

    def filter_fn(exp_config):
        model = exp_config["model"]
        if "hidden_layers" in model:
            return model["hidden_layers"][0]["name"] == "relu"
        else:
            return True

    line_kwargs = {
        "fista": {
            "c": constants.line_colors[0],
            "label": "Convex",
            "linewidth": 3,
        },
        "torch_adam_1.0": {
            "c": constants.line_colors[1],
            "label": "Adam, ss: 1",
            "linewidth": 3,
        },
        "torch_adam_0.1": {
            "c": constants.line_colors[2],
            "label": "Adam, ss: 0.1",
            "linewidth": 3,
        },
        "torch_adam_0.01": {
            "c": constants.line_colors[3],
            "label": "Adam, ss: 0.01",
            "linewidth": 3,
        },
        "torch_adam_0.001": {
            "c": constants.line_colors[4],
            "label": "Adam, ss: 0.001",
            "linewidth": 3,
        },
        "torch_adam_0.0001": {
            "c": constants.line_colors[5],
            "label": "Adam, ss: 0.0001",
            "linewidth": 3,
        },
        "torch_adam_1e-05": {
            "c": constants.line_colors[6],
            "label": "Adam, ss: e-5",
            "linewidth": 3,
        },
        "torch_sgd_1.0": {
            "c": constants.line_colors[7],
            "label": "SGD, ss: 1",
            "linewidth": 3,
        },
        "torch_sgd_0.1": {
            "c": constants.line_colors[8],
            "label": "SGD, ss: 0.1",
            "linewidth": 3,
        },
        "torch_sgd_0.01": {
            "c": constants.line_colors[9],
            "label": "SGD, ss: 0.01",
            "linewidth": 3,
        },
        "torch_sgd_0.001": {
            "c": constants.line_colors[10],
            "label": "SGD, ss: 0.001",
            "linewidth": 3,
        },
        "torch_sgd_0.0001": {
            "c": constants.line_colors[11],
            "label": "SGD, ss: 0.0001",
            "linewidth": 3,
        },
        "torch_sgd_1e-05": {
            "c": constants.line_colors[12],
            "label": "SGD, ss: e-5",
            "linewidth": 3,
        },
    }

    figure_labels["title"] = dataset_name
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
            "limits": constants.limits,
            "settings": settings,
        }
    )
    plots.append(plot_config)

PLOTS = {"mnist_timing": plots}
