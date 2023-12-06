"""
Initial attempt at performance profiles.
"""
from copy import deepcopy
from typing import Dict, List
import numpy as np
from experiment_utils import configs

from scaffold.uci_names import BINARY_SMALL_UCI_DATASETS

max_iters = 2000
gate_n_samples = 5000
relu_n_samples = 2500
outer_iters = 5000
max_total_iters = 10000
arrangement_seed = 650
lambda_to_try = np.logspace(-8, -3, 10).tolist()
n_estimators_to_try = [100, 200, 300]
depths_to_try = [2, 4]


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
        "active_proportion": [0.5, None],
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": lambda_to_try,
    },
    "initializer": [
        {"name": "zero"},
    ],
}

DeepConvexGated_GL1 = {
    "name": "deep_convex_mlp",
    "kernel": "einsum",
    "xgb_config": {
        "seed": 650,
        "depth": depths_to_try,
        "n_estimators": n_estimators_to_try,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": lambda_to_try,
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
        "active_proportion": [0.5, None],
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": lambda_to_try,
    },
    "initializer": {"name": ["zero"]},
    "delta": 100,
}

uci_data = {
    "name": BINARY_SMALL_UCI_DATASETS,
    "split_seed": 1995,
    "n_folds": 5,
    "fold_index": list(range(5)),
    "add_bias": True,
}

uci_data_tuning = {
    "name": BINARY_SMALL_UCI_DATASETS,
    "split_seed": 1995,
    "n_folds": 5,
    "fold_index": list(range(5)),
    "add_bias": [True, False],
    "unitize_data_cols": [True, False],
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

ReLU_EXPS = {
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


GATED_EXPS = {
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

DEEP_GATED_EXPS = {
    "method": FISTA_GL1,
    "model": DeepConvexGated_GL1,
    "data": uci_data_tuning,
    "metrics": metrics,
    "final_metrics": final_metrics,
    "seed": 778,
    "repeat": 1,
    "backend": "torch",
    "device": "cuda",
    "dtype": "float32",
}


EXPERIMENTS: Dict[str, List] = {
    "table_2_gs": [ReLU_EXPS, GATED_EXPS],
    "table_2_deep_gs": [DEEP_GATED_EXPS],
}
