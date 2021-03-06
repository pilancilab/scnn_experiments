"""
Extract final accuracy results for subset of the UCI datasets. 
"""
import os
from collections import defaultdict
from typing import Dict, Any, List
from functools import reduce
import pickle as pkl

import numpy as np

from experiment_utils import utils, files, configs

from scaffold.uci_names import SMALL_UCI_BIN_SUBSET

from exp_configs import EXPERIMENTS  # type: ignore


def row_key(exp_dict):
    dataset = exp_dict["data"]["name"]
    return dataset


metrics = (
    "test_accuracy",
    "test_nc_accuracy",
)


def line_key(exp_dict):
    """Load line key."""
    method = exp_dict["method"]
    key = method["name"]
    if exp_dict["model"]["name"] == "torch_mlp_l1":
        layer_type = exp_dict["model"]["hidden_layers"][0]["name"]
        key = f"{key}_{layer_type}"
    step_size = method.get("step_size", None)
    lam = exp_dict["model"]["regularizer"]["lambda"]

    sign_patterns = exp_dict["model"].get("sign_patterns", None)
    prop = None
    if sign_patterns is not None:
        prop = sign_patterns.get("active_proportion", None)

    return (key, step_size, lam, prop)


def repeat_key(exp_dict):
    return (exp_dict["seed"], exp_dict["data"]["fold_index"])


exp_ids = ["table_3_grelu_gs", "table_3_relu_gs", "table_3_nc_relu_gs"]
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

best_params = {}
for dataset in flipped_grid.keys():
    if dataset not in best_params:
        best_params[dataset] = {}

    for (method, step_size, lam, prop) in flipped_grid[dataset].keys():
        nc_accuracy = flipped_grid[dataset][(method, step_size, lam, prop)][
            "test_nc_accuracy"
        ]["center"][-1]

        if nc_accuracy >= 0:
            objective = nc_accuracy
        else:
            objective = flipped_grid[dataset][(method, step_size, lam, prop)][
                "test_accuracy"
            ]["center"][-1]

        if (
            method not in best_params[dataset]
            or best_params[dataset][method]["val"] < objective
        ):
            best_params[dataset][method] = best_params[dataset].get(method, {})
            best_params[dataset][method]["val"] = objective
            best_params[dataset][method]["key"] = (step_size, lam, prop)

# save parameters
with open("scripts/exp_configs/table_3_best_params.pkl", "wb") as f:
    pkl.dump(best_params, f)
