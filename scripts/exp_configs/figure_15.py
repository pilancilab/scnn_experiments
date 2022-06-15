from typing import Dict, List

import numpy as np

from scaffold.uci_names import PERFORMANCE_PROFILE

# global parameters
max_iters = 1000
# no reason not to increase this
outer_iters = 5000
max_total_iters = 10000
n_samples = 500
lambda_to_try = np.logspace(-5, -1, 6).tolist()

# PyTorch models and optimizers

FISTA_GL1 = {
    "name": "fista",
    "ls_cond": {"name": "quadratic_bound"},
    "backtrack_fn": {"name": "backtrack", "beta": 0.8},
    "step_size_update": {"name": "lassplore", "alpha": 1.25, "threshold": 5.0},
    "init_step_size": 1.0,
    "term_criterion": {"name": "grad_norm"},
    "prox": {"name": "group_l1"},
    "ls_type": "prox_path",
    "max_iters": max_iters,
    "restart_rule": "gradient_mapping",
}

# augmented lagrangian methods

AL = {
    "name": "augmented_lagrangian",
    "term_criterion": {
        "name": "constrained_heuristic",
        "grad_tol": 1e-6,
        "constraint_tol": 1e-6,
    },
    "subprob_tol": 1e-6,
    "use_delta_init": [True, False],
    "subproblem_solver": FISTA_GL1,
    "max_iters": outer_iters,
    "max_total_iters": max_total_iters,
}

ConvexRelu_GL1 = {
    "name": "al_mlp",
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": n_samples,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": lambda_to_try,
    },
    "initializer": {"name": ["zero"]},
    "delta": [1, 10, 100, 1000, 10000],
}

# metrics and datasets.

metrics = (
    ["objective", "squared_error", "accuracy", "grad_norm", "constraint_gaps"],
    ["squared_error", "accuracy"],
    [
        "group_sparsity",
        "active_neurons",
        "sp_success",
        "step_size",
        "num_backtracks",
    ],
)

datasets = {
    "name": PERFORMANCE_PROFILE,
    "split_seed": 1995,
    "use_valid": False,
}

EXPERIMENTS: Dict[str, List] = {
    "figure_15": [
        {
            "method": AL,
            "model": ConvexRelu_GL1,
            "data": datasets,
            "metrics": metrics,
            "seed": 778,
            "repeat": 1,
            "backend": "torch",
            "device": "cuda",
            "dtype": "float32",
        },
    ],
}
