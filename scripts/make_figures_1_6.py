import os
from copy import deepcopy

import numpy as np

from experiment_utils import configs, utils, files
from experiment_utils.plotting import defaults
from experiment_utils.plotting.plot_cell import make_error_bar_plot
from experiment_utils.plotting import plot_grid, plot_cell, defaults

from plot_configs import constants
from scaffold.uci_names import UCI_BIN_SUBSET

from exp_configs import EXPERIMENTS

# plot configuration #

# CONSTANTS #

row_key = (
    lambda config: config["model"]["regularizer"]["lambda"]
    if "regularizer" in config["model"]
    else config["model"]["weight_decay"]
)

repeat_key = ("data", "name")

metrics = [
    "train_base_objective",
    "train_grad_norm",
    "train_nc_accuracy",
    "test_nc_accuracy",
]


def line_key_gen(exp_dict):
    """Load line key."""
    method = exp_dict["method"]
    model = exp_dict["model"]
    key = method["name"]

    if model["name"] == "torch_mlp":
        if model["hidden_layers"][0]["name"] == "relu":
            seed = model["hidden_layers"][0]["seed"]
        else:
            seed = model["hidden_layers"][0]["sign_patterns"]["seed"]
    else:
        seed = model["sign_patterns"]["seed"]

    return f"{key}_{seed}"


processing_fns = [
    utils.cum_sum(lambda key: key[1] == "time"),
    utils.extend(1500, lambda key: True),
]

lambda_to_try = [1e-4]
figure_labels = {
    "x_labels": {
        "train_base_objective": "# Data Passes",
        "train_nc_accuracy": "# Data Passes",
    },
    "y_labels": {},
    "col_titles": {
        "train_base_objective": "Training Objective",
        "train_nc_accuracy": "Training Accuracy",
    },
    # "row_titles": {lam: "lambda = " + str(round(lam, 4)) for lam in lambda_to_try},
}

settings = defaults.DEFAULT_SETTINGS
settings["y_labels"] = "every_col"
settings["x_labels"] = "bottom_row"
settings["legend_cols"] = 14
settings["fig_height"] = 6
settings["fig_width"] = 5
settings["show_legend"] = False
settings["bottom_margin"] = 0.12
settings["alpha"] = 0.75
settings["wspace"] = 0.18
settings["axis_labels_fs"] = 26
settings["tick_fs"] = 18


methods_to_plot = [
    "torch_sgd",
    "fista",
    "augmented_lagrangian",
]

# complicated filtering for plots.

n_samples_to_plot = 100
d = 50
keep = [
    (("method", "name"), ["torch_sgd", "augmented_lagrangian"]),
    (("method", "step_size"), [10]),
    (("data", "d"), [d]),
    (("model", "regularizer", "lambda"), lambda_to_try),
]

remove = [(("seed"), [655])]
line_key = line_key_gen


def filter_fn(exp_dict):
    model = exp_dict["model"]
    model_name = model["name"]

    if model_name in ["convex_mlp", "al_mlp"]:
        return (
            model["sign_patterns"]["n_samples"] == n_samples_to_plot
            and "active_set" in exp_dict
            and model["sign_patterns"]["seed"] == 650
        )
    elif (
        model_name == "torch_mlp"
        and model["hidden_layers"][0]["name"] == "relu"
    ):
        return model["hidden_layers"][0]["p"] == 100


marker_size = 16
line_width = 5.5
line_kwargs = {
    "torch_sgd_650": {
        "c": constants.line_colors[10],
        "label": "1",
        "linestyle": "--",
        "linewidth": line_width,
    },
    "torch_sgd_651": {
        "c": constants.line_colors[1],
        "label": "2",
        "linestyle": "--",
        "linewidth": line_width,
    },
    "torch_sgd_652": {
        "c": constants.line_colors[2],
        "label": "3",
        "linestyle": "--",
        "linewidth": line_width,
    },
    "torch_sgd_653": {
        "c": constants.line_colors[3],
        "label": "4",
        "linestyle": "--",
        "linewidth": line_width,
    },
    "torch_sgd_654": {
        "c": constants.line_colors[4],
        "label": "5",
        "linestyle": "--",
        "linewidth": line_width,
    },
    "torch_sgd_655": {
        "c": constants.line_colors[5],
        "label": "6",
        "linestyle": "--",
        "linewidth": line_width,
    },
    "torch_sgd_656": {
        "c": constants.line_colors[6],
        "label": "7",
        "linestyle": "--",
        "linewidth": line_width,
    },
    "torch_sgd_657": {
        "c": constants.line_colors[7],
        "label": "8",
        "linestyle": "--",
        "linewidth": line_width,
    },
    "torch_sgd_658": {
        "c": constants.line_colors[8],
        "label": "9",
        "linestyle": "--",
        "linewidth": line_width,
    },
    "torch_sgd_659": {
        "c": constants.line_colors[9],
        "linestyle": "--",
        "label": "10",
        "linewidth": line_width,
    },
    "augmented_lagrangian_650": {
        "c": constants.line_colors[0],
        "label": "1",
        "linestyle": "-",
        "marker": "X",
        "markersize": marker_size,
        "markevery": 0.1,
        "linewidth": line_width,
    },
    "augmented_lagrangian_651": {
        "c": constants.line_colors[1],
        "label": "2",
        "linestyle": "-",
        "marker": "X",
        "markersize": marker_size,
        "markevery": 0.1,
        "linewidth": line_width,
    },
    "augmented_lagrangian_652": {
        "c": constants.line_colors[2],
        "label": "3",
        "linestyle": "-",
        "marker": "X",
        "markersize": marker_size,
        "markevery": 0.1,
        "linewidth": line_width,
    },
    "augmented_lagrangian_653": {
        "c": constants.line_colors[3],
        "label": "4",
        "linestyle": "-",
        "marker": "X",
        "markersize": marker_size,
        "markevery": 0.1,
        "linewidth": line_width,
    },
    "augmented_lagrangian_654": {
        "c": constants.line_colors[4],
        "label": "5",
        "linestyle": "-",
        "marker": "X",
        "markersize": marker_size,
        "markevery": 0.1,
        "linewidth": line_width,
    },
    "augmented_lagrangian_655": {
        "c": constants.line_colors[5],
        "label": "6",
        "linestyle": "-",
        "marker": "X",
        "markersize": marker_size,
        "markevery": 0.1,
        "linewidth": line_width,
    },
    "augmented_lagrangian_656": {
        "c": constants.line_colors[6],
        "label": "7",
        "linestyle": "-",
        "marker": "X",
        "markersize": marker_size,
        "markevery": 0.1,
        "linewidth": line_width,
    },
    "augmented_lagrangian_657": {
        "c": constants.line_colors[7],
        "label": "8",
        "linestyle": "-",
        "marker": "X",
        "markersize": marker_size,
        "markevery": 0.1,
        "linewidth": line_width,
    },
    "augmented_lagrangian_658": {
        "c": constants.line_colors[8],
        "label": "9",
        "linestyle": "-",
        "marker": "X",
        "markersize": marker_size,
        "markevery": 0.1,
        "linewidth": line_width,
    },
    "augmented_lagrangian_659": {
        "c": constants.line_colors[9],
        "linestyle": "-",
        "marker": "X",
        "markersize": marker_size,
        "markevery": 0.1,
        "label": "10",
        "linewidth": line_width,
    },
}

figure_labels["title"] = ""
figure_labels = deepcopy(figure_labels)
constants.log_scale["train_nc_accuracy"] = "linear-log"
constants.log_scale["test_nc_accuracy"] = "linear-log"
constants.log_scale["train_nc_objective"] = "linear-log"
constants.log_scale["train_base_objective"] = "linear-log"
constants.log_scale["train_grad_norm"] = "log-log"

limits = {
    "train_base_objective": ([5, 1500], [0.0, 0.051]),
    "train_nc_accuracy": ([5, 1500], [0.9, 1.01]),
    "train_grad_norm": ([5, 1500], [1e-7, 0.01]),
    "test_nc_accuracy": ([5, 1500], (0.88, 0.95)),
}

config_list = EXPERIMENTS["figure_1"]
results_dir = "results/figure_1"

metrics = [
    "train_base_objective",
    # "train_grad_norm",
    # "train_nc_accuracy",
    "test_nc_accuracy",
]

plot_config = {
    "name": "figure_1",
    "row_key": row_key,
    "metrics": metrics,
    "line_key": line_key,
    "repeat_key": repeat_key,
    # "x_key": "time",
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

# Figure 1

metric_grid = files.load_and_clean_experiments(
    config_list,
    results_dir,
    metrics=plot_config["metrics"],
    row_key=plot_config["row_key"],
    line_key=plot_config["line_key"],
    repeat_key=plot_config["repeat_key"],
    metric_fn=plot_config["metrics_fn"],
    keep=plot_config.get("keep", []),
    remove=plot_config.get("remove", []),
    filter_fn=plot_config.get("filter_fn", None),
    transform_fn=plot_config.get("transform_fn", None),
    processing_fns=plot_config.get("processing_fns", None),
    x_key=plot_config.get("x_key", None),
    x_vals=plot_config.get("x_vals", None),
)

plot_grid.plot_grid(
    plot_fn=plot_config.get("plot_fn", plot_cell.make_convergence_plot),
    results=metric_grid,
    figure_labels=plot_config["figure_labels"],
    line_kwargs=plot_config["line_kwargs"],
    limits=plot_config["limits"],
    log_scale=plot_config["log_scale"],
    base_dir=os.path.join("figures/figure_1.pdf"),
    settings=plot_config["settings"],
)


# Figure 6

plot_config = deepcopy(plot_config)
plot_config["name"] = "figure_6"

plot_config["metrics"] = [
    "train_base_objective",
    "train_grad_norm",
    "train_nc_accuracy",
    "test_nc_accuracy",
]

plot_config["settings"]["fig_width"] = 6

metric_grid = files.load_and_clean_experiments(
    config_list,
    results_dir,
    metrics=plot_config["metrics"],
    row_key=plot_config["row_key"],
    line_key=plot_config["line_key"],
    repeat_key=plot_config["repeat_key"],
    metric_fn=plot_config["metrics_fn"],
    keep=plot_config.get("keep", []),
    remove=plot_config.get("remove", []),
    filter_fn=plot_config.get("filter_fn", None),
    transform_fn=plot_config.get("transform_fn", None),
    processing_fns=plot_config.get("processing_fns", None),
    x_key=plot_config.get("x_key", None),
    x_vals=plot_config.get("x_vals", None),
)

plot_grid.plot_grid(
    plot_fn=plot_config.get("plot_fn", plot_cell.make_convergence_plot),
    results=metric_grid,
    figure_labels=plot_config["figure_labels"],
    line_kwargs=plot_config["line_kwargs"],
    limits=plot_config["limits"],
    log_scale=plot_config["log_scale"],
    base_dir=os.path.join("figures/figure_6.pdf"),
    settings=plot_config["settings"],
)
