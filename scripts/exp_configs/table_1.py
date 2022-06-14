from typing import Dict, List
from copy import deepcopy
from experiment_utils import configs
from scaffold.uci_names import cone_decomp

import numpy as np

outer_iters = 5000
max_total_iters = 10000
max_iters = 2000
lambda_to_try = 0.001
pattern_seed = (650 + np.arange(5)).tolist()

approximation_lambda = [1e-12, 1e-10, 1e-8, 1e-6]
approximation_tol = [1e-12, 1e-10, 1e-8, 1e-6]

FISTA_GL1 = {
    "name": "fista",
    "ls_cond": {"name": "quadratic_bound"},
    "backtrack_fn": {"name": "backtrack", "beta": 0.8},
    "step_size_update": {
        "name": "lassplore",
        "alpha": 1.25,
        "threshold": 2.0,
    },
    "init_step_size": 1.0,
    "term_criterion": {"name": "grad_norm", "tol": 1e-6},
    "ls_type": "prox_path",
    "prox": {"name": "group_l1"},
    "max_iters": max_iters,
    "restart_rule": ["gradient_mapping"],
}

CONE_DECOMP = deepcopy(FISTA_GL1)
CONE_DECOMP["post_process"] = [
    {
        "name": [
            "min_l2_decomp",
        ],
        "solver": "mosek",
    },
    {
        "name": "approximate_decomp",
        "regularizer": {
            "name": "group_l1",
            "lambda": 1e-10,
        },
        "tol": 1e-10,
        "combined": True,
    },
    None,
]

AL = {
    "name": "augmented_lagrangian",
    "term_criterion": {
        "name": "constrained_opt",
        "grad_tol": 1e-6,
        "constraint_tol": 1e-6,
    },
    "use_delta_init": True,
    "subproblem_solver": FISTA_GL1,
    "max_iters": outer_iters,
    "max_total_iters": max_total_iters,
}

ConvexRelu_GL1 = {
    "name": "al_mlp",
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": 1000,
        "seed": pattern_seed,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": lambda_to_try,
    },
    "initializer": {"name": "zero"},
    "delta": 100,
}

ConvexGated_GL1 = {
    "name": "convex_mlp",
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": 1000,
        "seed": pattern_seed,
    },
    "regularizer": {"name": "group_l1", "lambda": lambda_to_try},
    "initializer": {"name": "zero"},
}

uci_data = {
    "name": cone_decomp,
    "split_seed": 1995,
    "use_valid": False,
}

metrics = (
    [
        "objective",
        "nc_objective",
        "base_objective",
        "grad_norm",
        "constraint_gaps",
        "accuracy",
        "nc_accuracy",
    ],
    ["accuracy", "nc_accuracy"],
    [
        "active_neurons",
        "group_sparsity",
        "group_norms",
    ],
)

final_metrics = (
    [
        "objective",
        "nc_objective",
        "base_objective",
        "grad_norm",
        "constraint_gaps",
        "accuracy",
        "nc_accuracy",
    ],
    ["accuracy", "nc_accuracy"],
    [
        "group_sparsity",
        "active_neurons",
        "group_norms",
    ],
)

AL_EXPS = {
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

CONE_EXPS = {
    "method": CONE_DECOMP,
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

EXPERIMENTS: Dict[str, List] = {
    "table_1": [
        CONE_EXPS,
        AL_EXPS,
    ],
}
