from copy import deepcopy
from typing import Dict, List
import numpy as np

from scaffold.uci_names import HYPERPLANE_ABLATIONS

max_iters = max_epochs = 1000
n_samples = [10, 100, 1000]
seeds = (650 + np.arange(10)).tolist()
lambda_to_try = np.logspace(-6, -2, 20).tolist() + np.logspace(-2, 0, 10).tolist()[1:]

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
    "term_criterion": {"name": "grad_norm"},
    "ls_type": "prox_path",
    "prox": {"name": "group_l1"},
    "max_iters": max_iters,
    "restart_rule": "gradient_mapping",
}

AL = {
    "name": "augmented_lagrangian",
    "term_criterion": {
        "name": "constrained_heuristic",
        "grad_tol": 1e-6,
        "constraint_tol": 1e-6,
    },
    "use_delta_init": True,
    "subproblem_solver": FISTA_GL1,
    "max_iters": 5000,
    "max_total_iters": 10000,
}

ConvexGated_GL1 = {
    "name": "convex_mlp",
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": n_samples,
        "seed": seeds,
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
        "n_samples": n_samples,
        "seed": seeds,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": lambda_to_try,
    },
    "initializer": {"name": ["zero"]},
    "delta": 100,
}

metrics = (
    ["objective", "squared_error", "grad_norm", "constraint_gaps", "accuracy"],
    ["squared_error", "accuracy"],
    [
        "group_sparsity",
        "active_neurons",
        "sp_success",
        "step_size",
        "num_backtracks",
    ],
)

uci_ablations = {
    "name": HYPERPLANE_ABLATIONS,
    "split_seed": 1995,
    "use_valid": False,
}


EXPERIMENTS: Dict[str, List] = {
    "figure_5": [
        {
            "method": AL,
            "model": ConvexRelu_GL1,
            "data": uci_ablations,
            "metrics": metrics,
            "seed": 778,
            "repeat": 1,
            "backend": "torch",
            "device": "cuda",
            "dtype": "float32",
        },
        {
            "method": FISTA_GL1,
            "model": ConvexGated_GL1,
            "data": uci_ablations,
            "metrics": metrics,
            "seed": 778,
            "repeat": 1,
            "backend": "torch",
            "device": "cuda",
            "dtype": "float32",
        },
    ]
}
