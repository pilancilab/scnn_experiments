import os
from collections import defaultdict
from typing import Dict, List
from functools import reduce

from experiment_utils import utils, files

from exp_configs import EXPERIMENTS  # type: ignore

# relu experiments

row_key = ("data", "name")
metrics = (
    "test_nc_accuracy",
    "test_accuracy",
    "time",
    "group_norms",
)


def line_key(exp_dict):
    """Load line key."""
    method = exp_dict["method"]
    key = method["name"]
    if exp_dict["method"].get("post_process", None) is not None:
        decomp_type = exp_dict["method"]["post_process"]["name"]
        key = f"{key}_{decomp_type}"

    return key


repeat_key = ("model", "sign_patterns", "seed")

exp_ids = ["table_1"]
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
    keep=[(("model", "regularizer", "lambda"), [0.001])],
    remove=[],
    filter_fn=None,
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

# generate table
n_digits = 1
metric_keys = ["test_nc_accuracy", "group_norms", "time"]
method_keys = [
    "fista",
    "fista_min_l2_decomp",
    "fista_approximate_decomp",
    "augmented_lagrangian",
]

method_to_col = {
    "fista": "R-FISTA",
    "fista_min_l2_decomp": "CD-SOCP",
    "fista_approximate_decomp": "CD-A",
    "augmented_lagrangian": "AL",
}

metric_to_str = {
    "test_nc_accuracy": "Accuracy",
    "group_norms": "Norm",
    "time": "Time",
}

result_str = ""
for method in method_keys:
    result_str = result_str + ", " + method_to_col[method] + ", , "

result_str = result_str + " \n "

for method in method_keys:
    for metric in metric_keys:
        result_str = result_str + ", " + metric_to_str[metric]

result_str = result_str + " \n"

for dataset in flipped_grid.keys():
    result_str = result_str + dataset + ", "
    for method in method_keys:
        for key in metric_keys:
            center = flipped_grid[dataset][method][key]["center"][-1]
            upper = flipped_grid[dataset][method][key]["upper"][-1]
            lower = flipped_grid[dataset][method][key]["lower"][-1]

            if "accuracy" in key:
                center, upper, lower = (
                    round(center * 100, n_digits),
                    round(upper * 100, n_digits),
                    round(lower * 100, n_digits),
                )
            elif "time" in key:
                center, upper, lower = (
                    round(center, 2),
                    round(upper, 2),
                    round(lower, 2),
                )

            else:
                center, upper, lower = (
                    "{:.2e}".format(center),
                    "{:.2e}".format(upper),
                    "{:.2e}".format(lower),
                )

            result_str += f"{center}, "
    result_str = result_str + " \n"

with open("tables/table_1.csv", "w") as f:
    f.write(result_str)
