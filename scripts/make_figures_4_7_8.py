from functools import partial

import numpy as np

from experiment_utils.plotting.defaults import DEFAULT_SETTINGS
from experiment_utils import configs

from exp_configs import EXPERIMENTS  # type: ignore
from plotting_utils.performance_profile import compute_obj_success_ratios  # type: ignore
from plot_configs import constants  # type: ignore

from matplotlib import pyplot as plt  # type: ignore

plt.rcParams.update({"text.usetex": True})

marker_spacing = 0.1
marker_size = 16
line_width = 5
max_x = 1e3
min_x = 1e-2

settings = DEFAULT_SETTINGS
settings["titles_fs"] = 26
settings["axis_labels_fs"] = 24
settings["legend_fs"] = 20
settings["ticks_fs"] = 16
settings["wspace"] = 0.18

line_kwargs = {
    "augmented_lagrangian": {
        "c": constants.line_colors[1],
        "label": "AL (Ours)",
        "linewidth": line_width,
        "marker": "v",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
    "fista": {
        "c": constants.line_colors[0],
        "label": "R-FISTA (Ours)",
        "linewidth": line_width,
        "marker": "v",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
    "torch_sgd": {
        "c": constants.line_colors[2],
        "label": "SGD",
        "linewidth": line_width,
        "marker": "s",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
    "torch_adam": {
        "c": constants.line_colors[3],
        "label": "Adam",
        "linewidth": line_width,
        "marker": "X",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
    "cvxpy": {
        "c": constants.line_colors[4],
        "label": "MOSEK",
        "linewidth": line_width,
        "marker": "D",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
}


def method_key(exp_dict):
    """Load line key."""
    method = exp_dict["method"]
    key = method["name"]

    return key


def problem_key(exp_dict):
    dataset_name = exp_dict["data"]["name"]
    lam = exp_dict["model"]["regularizer"]["lambda"]

    return f"{dataset_name}_{lam}"


def filter_result(exp_metrics, exp_config):

    """Remove experiments corresponding to null models."""
    if (
        exp_metrics["active_neurons"][-1]
        == 0
        # or exp_config["data"]["name"] not in two_class
    ):

        return True
    else:
        return False


def compute_xy_values_tol(exp_metrics, exp_config, min_obj, tol):

    # treat all CVXPY solves as successful for now.
    time_list = np.array(exp_metrics["time"])

    time = np.sum(exp_metrics["time"])

    if (
        "train_nc_objective" in exp_metrics
        and exp_metrics["train_nc_objective"][-1] != -1
    ):
        best_obj = np.min(exp_metrics["train_nc_objective"])
        best_ind = np.argmin(exp_metrics["train_nc_objective"])
    else:
        best_obj = np.min(exp_metrics["train_objective"])
        best_ind = np.argmin(exp_metrics["train_objective"])

    rel_diff = (best_obj - min_obj) / min_obj

    success = rel_diff <= tol

    if success:
        time = np.sum(time_list[0 : best_ind + 1])

    return time, success


gated_exp_list = configs.expand_config_list(
    EXPERIMENTS["figure_4_grelu"] + EXPERIMENTS["figure_4_grelu_cvxpy"]
)

tol_list = [1.0, 0.5, 0.1]

for tol in tol_list:
    compute_xy_values = partial(compute_xy_values_tol, tol=tol)

    gated_success_ratios, gated_n_problems = compute_obj_success_ratios(
        ["figure_4_grelu", "figure_4_grelu_cvxpy"],
        gated_exp_list,
        compute_xy_values,
        problem_key,
        method_key,
        filter_result,
        convex_key="fista",
    )

    relu_exp_list = configs.expand_config_list(
        EXPERIMENTS["figure_4_relu"]
        + EXPERIMENTS["figure_4_relu_cvxpy"]
        + EXPERIMENTS["figure_4_nc_relu"]
    )

    relu_success_ratios, relu_n_problems = compute_obj_success_ratios(
        ["figure_4_relu", "figure_4_relu_cvxpy", "figure_4_nc_relu"],
        relu_exp_list,
        compute_xy_values,
        problem_key,
        method_key,
        filter_result,
        convex_key="augmented_lagrangian",
    )

    fig = plt.figure(figsize=(12, 6))
    spec = fig.add_gridspec(ncols=2, nrows=1)
    ax0 = fig.add_subplot(spec[0, 0])

    ax0.set_ylim(0, 1)
    ax0.set_xscale("log")
    ax0.set_xlim(min_x, max_x)

    n_vals = 15

    def extend_data(x, y):
        last_val = y[-1]
        x = np.concatenate([x, np.linspace(x[-1], 10**3, n_vals)])
        y = np.concatenate([y, np.repeat(last_val, n_vals)])
        return x, y

    for key, (x, y) in gated_success_ratios.items():
        x, y = extend_data(np.squeeze(x), np.squeeze(y))
        ax0.plot(x, y, alpha=settings["line_alpha"], **line_kwargs[key])

    ax0.axhline(y=0.50, linestyle="--", linewidth="4", c="k")
    ax0.set_title("Gated ReLU Activations", fontsize=settings["subtitle_fs"])
    ax0.set_ylabel(
        "Prop. of Problems Solved", fontsize=settings["axis_labels_fs"]
    )
    ax0.set_xlabel("Time (Seconds)", fontsize=settings["axis_labels_fs"])
    ax0.tick_params(labelsize=settings["tick_fs"])

    ax1 = fig.add_subplot(spec[0, 1])

    ax1.set_ylim(0, 1)
    ax1.set_xscale("log")
    # ax1.set_xlim(min_x, max_x)

    for key, (x, y) in relu_success_ratios.items():
        x, y = extend_data(np.squeeze(x), np.squeeze(y))
        ax1.plot(x, y, alpha=settings["line_alpha"], **line_kwargs[key])

    ax1.axhline(y=0.50, linestyle="--", linewidth="4", c="k")
    ax1.set_title("ReLU Activations", fontsize=settings["subtitle_fs"])
    ax1.set_xlabel("Time (Seconds)", fontsize=settings["axis_labels_fs"])
    ax1.tick_params(labelsize=settings["tick_fs"])

    handles0, labels0 = ax0.get_legend_handles_labels()
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles, labels = handles0 + handles1, labels0 + labels1
    # handles, labels = handles0, labels0

    handles_to_plot, labels_to_plot = [], []
    for i, label in enumerate(labels):
        if label not in labels_to_plot:
            handles_to_plot.append(handles[i])
            labels_to_plot.append(label)

    legend = fig.legend(
        handles=handles_to_plot,
        labels=labels_to_plot,
        loc="lower center",
        borderaxespad=0.1,
        fancybox=False,
        shadow=False,
        ncol=len(handles_to_plot),
        fontsize=settings["legend_fs"],
        frameon=False,
    )

    plt.tight_layout()
    fig.subplots_adjust(
        wspace=settings["wspace"],
        hspace=settings["vspace"],
        bottom=0.23,
    )
    if tol == 1.0:
        figure_name = "figure_4"
    elif tol == 0.5:
        figure_name = "figure_7"
    else:
        figure_name = "figure_8"

    plt.savefig(f"figures/{figure_name}.pdf")

    plt.close()
