import os

import numpy as np

from experiment_utils import utils
from experiment_utils.plotting.defaults import DEFAULT_SETTINGS
from experiment_utils.plotting import plot_grid, plot_cell
from experiment_utils import files, configs

from scaffold.uci_names import PERFORMANCE_PROFILE

from exp_configs import EXPERIMENTS  # type: ignore
from plotting_utils.performance_profile import compute_success_ratios  # type: ignore
from plot_configs import constants

from matplotlib import pyplot as plt  # type: ignore

plt.rcParams.update({"text.usetex": True})

marker_spacing = 0.1
marker_size = 10
line_width = 3

settings = DEFAULT_SETTINGS
settings["titles_fs"] = 22
settings["axis_labels_fs"] = 18
settings["legend_fs"] = 11
# settings["line_alpha"] = 0.8

line_kwargs = {
    "1_True": {
        "c": constants.line_colors[1],
        "label": "$\\delta = 1$",
        "linewidth": line_width,
        "linestyle": "-",
        "marker": "v",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
    "10_True": {
        "c": constants.line_colors[2],
        "label": "$\\delta = 10$",
        "linewidth": line_width,
        "linestyle": "-",
        "marker": "s",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
    "100_True": {
        "c": constants.line_colors[3],
        "label": "$\\delta = 10^2$",
        "linewidth": line_width,
        "linestyle": "-",
        "marker": "X",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
    "1000_True": {
        "c": constants.line_colors[4],
        "label": "$\\delta = 10^3$",
        "linewidth": line_width,
        "linestyle": "-",
        "marker": "D",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
    "10000_True": {
        "c": constants.line_colors[5],
        "label": "$\\delta = 10^4$",
        "linewidth": line_width,
        "linestyle": "-",
        "marker": "^",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
    "1_False": {
        "c": constants.line_colors[1],
        "label": "",
        "linewidth": line_width,
        "linestyle": "--",
    },
    "10_False": {
        "c": constants.line_colors[2],
        "label": "",
        "linewidth": line_width,
        "linestyle": "--",
    },
    "100_False": {
        "c": constants.line_colors[3],
        "label": "",
        "linewidth": line_width,
        "linestyle": "--",
    },
    "1000_False": {
        "c": constants.line_colors[4],
        "label": "",
        "linewidth": line_width,
        "linestyle": "--",
    },
    "10000_False": {
        "c": constants.line_colors[5],
        "label": "",
        "linewidth": line_width,
        "linestyle": "--",
    },
}


def problem_key(exp_dict):
    dataset_name = exp_dict["data"]["name"]
    lam = exp_dict["model"]["regularizer"]["lambda"]

    return f"{dataset_name}_{lam}"


def method_key(exp_dict):
    """Load line key."""
    method = exp_dict["method"]
    model = exp_dict["model"]
    delta = model["delta"]
    use_delta_init = method.get("use_delta_init", True)

    key = f"{delta}_{use_delta_init}"

    return key


def filter_result(exp_metrics):

    """Remove experiments corresponding to null models."""
    if exp_metrics["active_neurons"][-1] == 0:

        return True
    else:
        return False


def compute_xy_values(exp_metrics, exp_config):
    total_f_evals = np.sum(np.array(exp_metrics["num_backtracks"]) + 1) - 2
    success = (
        1
        if exp_metrics["train_grad_norm"][-1] <= 1e-6
        and exp_metrics["train_constraint_gaps"][-1] <= 1e-6
        else 0
    )

    # method diverged
    if not success and total_f_evals < 2000:
        total_f_evals = 2000
    return total_f_evals, success


exp_list = configs.expand_config_list(EXPERIMENTS["figure_15"])

success_ratios, n_problems = compute_success_ratios(
    ["figure_15"],
    exp_list,
    compute_xy_values,
    problem_key,
    method_key,
    filter_result,
    remove_degenerate_problems=False,
)

fig = plt.figure(figsize=(6, 6))
spec = fig.add_gridspec(ncols=1, nrows=1)
ax = fig.add_subplot(spec[0, 0])

max_x = 10000

ax.set_ylim(0, 1)
ax.set_xlim(0, max_x)

for key, (x, y) in success_ratios.items():
    x, y = np.squeeze(x), np.squeeze(y)
    ax.plot(x, y, alpha=settings["line_alpha"], **line_kwargs[key])

ax.axhline(y=0.80, linestyle="--", linewidth="2", c="k")

handles, labels = ax.get_legend_handles_labels()
final_labels = []
final_handles = []
for i, lb in enumerate(labels):
    if "Fixed" not in lb:
        final_labels.append(lb)
        final_handles.append(handles[i])


legend = fig.legend(
    handles=final_handles,
    labels=final_labels,
    loc="lower center",
    borderaxespad=0.1,
    fancybox=False,
    shadow=False,
    ncol=5,
    fontsize=settings["legend_fs"],
    frameon=False,
)
ax.set_title("Choosing $\\delta$ by Windowing", fontsize=settings["titles_fs"])
ax.set_ylabel("Prop. of Problems Solved", fontsize=settings["axis_labels_fs"])
ax.set_xlabel("Number of Data Passes", fontsize=settings["axis_labels_fs"])
ax.tick_params(labelsize=settings["tick_fs"])

plt.tight_layout()
fig.subplots_adjust(
    wspace=settings["wspace"],
    hspace=settings["vspace"],
    bottom=0.15,
)
plt.savefig("figures/figure_15.pdf")

plt.close()

# plot convergence for two randomly selected datasets
rng = np.random.default_rng(610)
datasets_to_plot = rng.choice(PERFORMANCE_PROFILE, size=2, replace=False)

marker_size = 0
line_width = 3
marker_spacing = 0.5

line_kwargs = {
    "1_True": {
        "c": constants.line_colors[3],
        "label": "Windowing ($1$)",
        "linewidth": line_width,
        "linestyle": "-",
        "marker": "v",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
    "10000_True": {
        "c": constants.line_colors[4],
        "label": "Windowing ($10^4$)",
        "linewidth": line_width,
        "linestyle": "-",
        "marker": "^",
        "markevery": marker_spacing,
        "markersize": marker_size,
    },
    "1_False": {
        "c": constants.line_colors[1],
        "label": "Fixed ($1$)",
        "linewidth": line_width,
        "linestyle": "-",
    },
    "10000_False": {
        "c": constants.line_colors[5],
        "label": "Fixed ($10^4$)",
        "linewidth": line_width,
        "linestyle": "-",
    },
}

line_key = method_key

lambda_list = np.logspace(-5, -1, 6)
lambda_to_plot = {
    "monks-2": lambda_list[[0, -3]],
    "ilpd-indian-liver": lambda_list[[0, -3]],
}

plot_limits = {
    "monks-2": {
        (lambda_list[0], "train_constraint_gaps"): ([10, None], None),
        (lambda_list[-3], "train_constraint_gaps"): ([10, None], None),
        (lambda_list[0], "train_grad_norm"): ([10, None], None),
        (lambda_list[-3], "train_grad_norm"): ([10, None], None),
    },
    "ilpd-indian-liver": {
        (lambda_list[0], "train_constraint_gaps"): ([10, None], None),
        (lambda_list[-3], "train_constraint_gaps"): ([10, None], None),
        (lambda_list[0], "train_grad_norm"): ([10, None], None),
        (lambda_list[-3], "train_grad_norm"): ([10, None], None),
    },
}


for i, dataset_name in enumerate(datasets_to_plot):

    limits = plot_limits[dataset_name]
    exp_ids = ["figure_15"]
    results_dir = [os.path.join("results", eid) for eid in exp_ids]
    metrics = [
        "train_grad_norm",
        "train_constraint_gaps",
    ]

    processing_fns = [
        utils.total_f_evals(lambda key: key[1] == "num_backtracks")
    ]

    metric_grid = files.load_and_clean_experiments(
        exp_list,
        results_dir,
        metrics,
        ("model", "regularizer", "lambda"),
        line_key,
        ("seed"),
        utils.quantile_metrics,
        keep=[
            (("data", "name"), [dataset_name]),
            (("model", "delta"), [1, 10**4]),
            (("model", "regularizer", "lambda"), lambda_to_plot[dataset_name]),
        ],
        remove=[],
        filter_fn=None,
        processing_fns=processing_fns,
        x_key="num_backtracks",
    )

    labels = {
        "col_titles": {
            "train_constraint_gaps": "Norm of Constraint Gap",
            "train_grad_norm": "(Sub)-Gradient Norm for $\\mathcal{L}_{\\delta}$",
        },
        "row_titles": {
            lam: "$\\lambda$ = {:.1E}".format(lam) for lam in lambda_list
        },
        "x_labels": {
            "train_constraint_gaps": "Number of Data Passes",
            "train_grad_norm": "Number of Data Passes",
        },
    }
    log_scale = {
        "train_constraint_gaps": "log-linear",
        "train_grad_norm": "log-linear",
    }

    settings["fig_width"] = 9
    settings["fig_height"] = 5
    settings["line_alpha"] = 0.8
    settings["show_legend"] = True
    settings["star_final"] = True
    settings["legend_cols"] = 6
    settings["bottom_margin"] = 0.34
    settings["legend_fs"] = 27
    settings["tick_fs"] = 26
    settings["axis_labels_fs"] = 32
    settings["subtitle_fs"] = 38
    settings["x_labels"] = "bottom_row"

    plot_grid.plot_grid(
        plot_fn=plot_cell.make_convergence_plot,
        results=metric_grid,
        figure_labels=labels,
        line_kwargs=line_kwargs,
        limits=limits,
        log_scale=log_scale,
        base_dir=os.path.join("figures", f"figure_{16+i}.pdf"),
        settings=settings,
    )
