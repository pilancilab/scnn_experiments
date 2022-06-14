"""
Utilities for generating performance profile plots.
"""
import os
from typing import Callable, Dict, Any, List
from operator import itemgetter
from collections import defaultdict
from copy import deepcopy
import math

import numpy as np

from experiment_utils import files


def compute_success_ratios(
    experiment_names: List[str],
    exp_list: Dict[str, Any],
    compute_xy_values: Callable,
    problem_key: Callable,
    method_key: Callable,
    filter_result: Callable,
    remove_degenerate_problems: bool = True,
    convex_key: str = "fista",
) -> Dict[str, Any]:

    results: Dict[str, Any] = defaultdict(lambda: defaultdict(list))
    best_results = defaultdict(list)

    # organize experiments by method
    for exp in exp_list:
        exp_metrics = None
        for name in experiment_names:
            try:
                exp_metrics = files.load_experiment(
                    exp,
                    results_dir=os.path.join("results", name),
                    load_metrics=True,
                )["metrics"]
            except:
                print("failed to load")
                continue

        if exp_metrics is None or filter_result(exp_metrics):
            continue

        x_value, success = compute_xy_values(exp_metrics, exp)

        results[method_key(exp)][problem_key(exp)].append([x_value, success])

    for mkey in results.keys():
        for pkey in results[mkey].keys():
            if remove_degenerate_problems and pkey not in results[convex_key]:
                continue

            # find the best run by minimizing over x-axis
            successes = list(filter(itemgetter(1), results[mkey][pkey]))
            if len(successes) > 0:
                best_results[mkey].append(min(successes, key=itemgetter(0)))
            else:
                best_results[mkey].append(max(results[mkey][pkey], key=itemgetter(0)))

    n_problems = max([len(best_results[key]) for key in best_results.keys()])

    for key in best_results.keys():
        ordered = list(sorted(best_results[key], key=itemgetter(0)))
        values = np.array(ordered).T
        cumulative_successes = np.cumsum(values[1]) / n_problems
        best_results[key] = (values[0], cumulative_successes)

    return best_results, n_problems


def compute_obj_success_ratios(
    experiment_names: List[str],
    exp_list: Dict[str, Any],
    compute_xy_values: Callable,
    problem_key: Callable,
    method_key: Callable,
    filter_result: Callable,
    remove_degenerate_problems: bool = True,
    convex_key: str = "fista",
) -> Dict[str, Any]:

    results: Dict[str, Any] = defaultdict(lambda: defaultdict(list))
    best_results = defaultdict(list)
    best_obj = {}

    # organize experiments by method
    metrics = []
    for exp in exp_list:
        exp_metrics = None
        for name in experiment_names:
            try:
                exp_metrics = files.load_experiment(
                    exp,
                    results_dir=os.path.join("results", name),
                    load_metrics=True,
                )["metrics"]
            except:
                continue

        if exp_metrics is None or filter_result(exp_metrics, exp):
            continue

        metrics.append((exp, exp_metrics))
        key = problem_key(exp)

        if (
            "train_nc_objective" in exp_metrics
            and exp_metrics["train_nc_objective"][-1] != -1
        ):
            min_obj = np.min(exp_metrics["train_nc_objective"])
        else:
            min_obj = np.min(exp_metrics["train_objective"])

        if not math.isnan(min_obj) and (key not in best_obj or min_obj < best_obj[key]):
            best_obj[key] = min_obj

    for exp, exp_metrics in metrics:
        pkey = problem_key(exp)
        x_value, success = compute_xy_values(exp_metrics, exp, best_obj[pkey])
        results[method_key(exp)][pkey].append([x_value, success])

    for mkey in results.keys():
        for pkey in results[mkey].keys():
            if remove_degenerate_problems and pkey not in results[convex_key]:
                continue

            # find the best run by minimizing over x-axis
            successes = list(filter(itemgetter(1), results[mkey][pkey]))
            if len(successes) > 0:
                best_results[mkey].append(min(successes, key=itemgetter(0)))
            else:
                best_results[mkey].append(max(results[mkey][pkey], key=itemgetter(0)))

    n_problems = max([len(best_results[key]) for key in best_results.keys()])

    for key in best_results.keys():
        ordered = list(sorted(best_results[key], key=itemgetter(0)))
        values = np.array(ordered).T
        cumulative_successes = np.cumsum(values[1]) / n_problems
        best_results[key] = (values[0], cumulative_successes)

    return best_results, n_problems


def compute_acc_success_ratios(
    experiment_names: List[str],
    exp_list: Dict[str, Any],
    compute_xy_values: Callable,
    problem_key: Callable,
    method_key: Callable,
    filter_result: Callable,
    remove_degenerate_problems: bool = True,
    convex_key: str = "fista",
) -> Dict[str, Any]:

    results: Dict[str, Any] = defaultdict(lambda: defaultdict(list))
    best_results = defaultdict(list)
    best_acc = {}

    # organize experiments by method
    metrics = []
    for exp in exp_list:
        exp_metrics = None
        for name in experiment_names:
            try:
                exp_metrics = files.load_experiment(
                    exp,
                    results_dir=os.path.join("results", name),
                    load_metrics=True,
                )["metrics"]
            except:
                continue

        if exp_metrics is None or filter_result(exp_metrics, exp):
            continue

        metrics.append((exp, exp_metrics))
        key = problem_key(exp)

        max_acc = np.max(exp_metrics["test_accuracy"])

        if not math.isnan(max_acc) and (key not in best_acc or max_acc > best_acc[key]):
            best_acc[key] = max_acc

    for exp, exp_metrics in metrics:
        pkey = problem_key(exp)
        x_value, success = compute_xy_values(exp_metrics, exp, best_acc[pkey])
        results[method_key(exp)][pkey].append([x_value, success])

    for mkey in results.keys():
        for pkey in results[mkey].keys():
            if remove_degenerate_problems and pkey not in results[convex_key]:
                continue

            # find the best run by minimizing over x-axis
            successes = list(filter(itemgetter(1), results[mkey][pkey]))
            if len(successes) > 0:
                best_results[mkey].append(min(successes, key=itemgetter(0)))
            else:
                best_results[mkey].append(max(results[mkey][pkey], key=itemgetter(0)))

    n_problems = max([len(best_results[key]) for key in best_results.keys()])

    for key in best_results.keys():
        ordered = list(sorted(best_results[key], key=itemgetter(0)))
        values = np.array(ordered).T
        cumulative_successes = np.cumsum(values[1]) / n_problems
        best_results[key] = (values[0], cumulative_successes)

    return best_results, n_problems
