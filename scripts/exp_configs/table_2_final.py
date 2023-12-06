"""
Initial attempt at performance profiles.
"""
from copy import deepcopy
from typing import Dict, List
import numpy as np
import pickle as pkl
from experiment_utils import configs

from scaffold.uci_names import BINARY_SMALL_UCI_DATASETS


max_iters = 2000
gate_n_samples = 5000
relu_n_samples = 2500
outer_iters = 5000
max_total_iters = 10000
arrangement_seed = (650 + np.array([0, 1, 2, 3, 4])).tolist()


FISTA_GL1 = {
    "name": "fista",
    "ls_cond": {"name": "quadratic_bound"},
    "backtrack_fn": {"name": "backtrack", "beta": 0.8},
    "step_size_update": {
        "name": "lassplore",
        "alpha": 1.25,
        "threshold": 5.0,
    },
    "init_step_size": 1.0,
    "term_criterion": {"name": "grad_norm", "tol": 1e-7},
    "ls_type": "prox_path",
    "prox": {"name": "group_l1"},
    "max_iters": max_iters,
    "restart_rule": "gradient_mapping",
}

AL = {
    "name": "augmented_lagrangian",
    "term_criterion": {
        "name": "constrained_heuristic",
        "grad_tol": 1e-7,
        "constraint_tol": 1e-7,
    },
    "use_delta_init": True,
    "subproblem_solver": FISTA_GL1,
    "max_iters": outer_iters,
    "max_total_iters": max_total_iters,
}

ConvexGated_GL1 = {
    "name": "convex_mlp",
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": gate_n_samples,
        "seed": arrangement_seed,
        "active_proportion": None,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": None,
    },
    "initializer": [
        {"name": "zero"},
    ],
}

ConvexRelu_GL1 = {
    "name": "al_mlp",
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": relu_n_samples,
        "seed": arrangement_seed,
        "active_proportion": None,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": None,
    },
    "initializer": {"name": ["zero"]},
    "delta": 100,
}

DeepConvexGated_GL1 = {
    "name": "deep_convex_mlp",
    "kernel": "einsum",
    "xgb_config": {
        "seed": arrangement_seed,
        "depth": None,
        "n_estimators": None,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": None,
    },
    "initializer": [
        {"name": "zero"},
    ],
}

uci_data = {
    "name": BINARY_SMALL_UCI_DATASETS,
    "split_seed": 1995,
    "use_valid": False,
    "add_bias": True,
}

metrics = (
    ["objective", "grad_norm"],
    [],
    ["num_backtracks"],
)

final_metrics = (
    [
        "base_objective",
        "accuracy",
        "squared_error",
        "constraint_gaps",
        "nc_accuracy",
    ],
    ["base_objective", "accuracy", "squared_error", "nc_accuracy"],
    ["active_neurons", "group_sparsity"],
)

try:
    with open("scripts/exp_configs/table_2_best_params.pkl", "rb") as f:
        best_params = pkl.load(f)

    convex_gated = {
        "method": FISTA_GL1,
        "model": ConvexGated_GL1,
        "data": uci_data,
        "metrics": metrics,
        "final_metrics": final_metrics,
        "seed": 778,
        "repeat": 1,
        "backend": "torch",
        "device": "cuda",
        "dtype": "float32",
    }

    convex_relu = {
        "method": AL,
        "model": ConvexRelu_GL1,
        "data": uci_data,
        "metrics": metrics,
        "final_metrics": final_metrics,
        "seed": 778,
        "repeat": 1,
        "backend": "torch",
        "device": "cuda",
        "dtype": "float32",
    }

    deep_convex_gated = {
        "method": FISTA_GL1,
        "model": DeepConvexGated_GL1,
        "data": uci_data,
        "metrics": metrics,
        "final_metrics": final_metrics,
        "seed": 778,
        "repeat": 1,
        "backend": "torch",
        "device": "cuda",
        "dtype": "float32",
    }

    expanded_gated = configs.expand_config(convex_gated)
    expanded_relu = configs.expand_config(convex_relu)
    expanded_deep = configs.expand_config(deep_convex_gated)

    fista_configs = []

    for config in expanded_gated:
        best_fista = deepcopy(config)
        best_fista["model"]["regularizer"]["lambda"] = best_params[
            config["data"]["name"]
        ]["fista"]["key"][1]
        best_fista["model"]["sign_patterns"][
            "active_proportion"
        ] = best_params[config["data"]["name"]]["fista"]["key"][2]
        fista_configs.append(best_fista)

    al_configs = []

    for config in expanded_relu:
        best_al = deepcopy(config)
        best_al["model"]["regularizer"]["lambda"] = best_params[
            config["data"]["name"]
        ]["augmented_lagrangian"]["key"][1]
        best_al["model"]["sign_patterns"]["active_proportion"] = best_params[
            config["data"]["name"]
        ]["augmented_lagrangian"]["key"][2]
        al_configs.append(best_al)

    deep_configs = []

    for config in expanded_deep:
        best_deep = deepcopy(config)
        best_deep["model"]["regularizer"]["lambda"] = best_params[
            config["data"]["name"]
        ]["fista_deep"]["key"][1]

        best_deep["model"]["xgb_config"]["depth"] = best_params[
            config["data"]["name"]
        ]["fista_deep"]["key"][2][0]

        best_deep["model"]["xgb_config"]["n_estimators"] = best_params[
            config["data"]["name"]
        ]["fista_deep"]["key"][2][1]

        deep_configs.append(best_deep)

    EXPERIMENTS: Dict[str, List] = {
        "table_2_final": fista_configs + al_configs,
        "table_2_final_deep": deep_configs,
    }

except:
    EXPERIMENTS = {}
