import os
from collections import defaultdict
from typing import Dict, List
from functools import reduce

from experiment_utils import utils, files

from exp_configs import EXPERIMENTS  # type: ignore

# relu experiments

row_key = ("data", "name")
metrics = (
    "train_base_objective",
    "test_nc_accuracy",
    "train_accuracy",
    "test_accuracy",
    "nc_train_accuracy",
    # "group_sparsity",
    # "active_neurons",
)


def line_key(exp_dict):
    """Load line key."""
    method = exp_dict["method"]
    key = method["name"]
    if exp_dict["model"]["name"] == "torch_mlp_l1":
        layer_type = exp_dict["model"]["hidden_layers"][0]["name"]
        key = f"{key}_{layer_type}"

    return key


def repeat_key(exp_dict):
    model = exp_dict["model"]
    if "sign_patterns" in model:
        return model["sign_patterns"]["seed"]
    elif model["hidden_layers"][0]["name"] == "gated_relu":
        return model["hidden_layers"][0]["sign_patterns"]["seed"]
    else:
        return exp_dict["src_hash"]


exp_ids = [
    "table_3_grelu_final",
    "table_3_relu_final",
    "table_3_nc_relu_final",
]
config_list: List[Dict] = reduce(
    lambda acc, eid: acc + EXPERIMENTS[eid], exp_ids, []
)

results_dir = [os.path.join("results", eid) for eid in exp_ids]

metric_grid = files.load_and_clean_experiments(
    config_list,
    results_dir,
    metrics,
    row_key,
    line_key,
    repeat_key,
    utils.quantile_metrics,
    keep=[],
    remove=[],
    filter_fn=None,
    processing_fns=[],
    x_key=None,
    x_vals=None,
)

flipped_grid = defaultdict(lambda: defaultdict(dict))

for dataset in metric_grid.keys():
    for metric in metric_grid[dataset].keys():
        for line in metric_grid[dataset][metric].keys():
            flipped_grid[dataset][line][metric] = metric_grid[dataset][metric][
                line
            ]

method_to_str = {
    "torch_sgd_gated_relu": "NC-GReLU (SGD)",
    "torch_adam_gated_relu": "NC-GReLU (Adam)",
    "torch_sgd_relu": "NC-ReLU (SGD)",
    "torch_adam_relu": "NC-ReLU (Adam)",
    "fista": "C-GReLU",
    "augmented_lagrangian": "C-ReLU",
}

# generate table
n_digits = 1
for include_var in ["", "_quartiles"]:
    result_str = ""
    for dataset in flipped_grid.keys():
        result_str = result_str + dataset + ", "
        method_str = ""
        for method in flipped_grid[dataset].keys():
            method_str = method_str + ", " + method_to_str[method]
            accuracy_key = "test_nc_accuracy"
            if "torch" in method:
                accuracy_key = "test_accuracy"

            center = (
                flipped_grid[dataset][method][accuracy_key]["center"][-1] * 100
            )
            upper = (
                flipped_grid[dataset][method][accuracy_key]["upper"][-1] * 100
            )
            lower = (
                flipped_grid[dataset][method][accuracy_key]["lower"][-1] * 100
            )

            center, upper, lower = (
                round(center, n_digits),
                round(upper, n_digits),
                round(lower, n_digits),
            )
            if include_var == "_quartiles":
                result_str += f"{center} ({lower}/{upper}), "
            else:
                result_str += f"{center}, "
        result_str = result_str + " \n"

    csv_string = method_str + "\n" + result_str

    with open(f"tables/table_3{include_var}.csv", "w") as f:
        f.write(csv_string)
