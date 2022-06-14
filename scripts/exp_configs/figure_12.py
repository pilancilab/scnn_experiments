from typing import Dict, List

import numpy as np

from scaffold.uci_names import PERFORMANCE_PROFILE

max_iters = 2000
n_samples = 5000
lambda_to_try = np.logspace(-5, -1, 6).tolist()

FISTA_GL1 = {
    "name": "fista",
    "ls_cond": {"name": "quadratic_bound"},
    "backtrack_fn": {"name": "backtrack", "beta": 0.8},
    "step_size_update": [
        {
            "name": "keep_new",
        },
        {
            "name": "forward_track",
            "alpha": 1.25,
        },
        {
            "name": "lassplore",
            "alpha": 1.25,
            "threshold": [2, 5, 10],
        },
    ],
    "init_step_size": 1.0,
    "term_criterion": {"name": "grad_norm"},
    "prox": {"name": "group_l1"},
    "ls_type": ["prox_path"],
    "max_iters": max_iters,
    "restart_rule": "gradient_mapping",
}

ConvexGated_GL1 = {
    "name": "convex_mlp",
    "kernel": "einsum",
    "sign_patterns": {"name": "sampler", "n_samples": n_samples, "seed": 650},
    "regularizer": {"name": "group_l1", "lambda": lambda_to_try},
    "initializer": {"name": "zero"},
}

metrics = (
    ["objective", "squared_error", "accuracy", "grad_norm"],
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
    "figure_12": [
        {
            "method": FISTA_GL1,
            "model": ConvexGated_GL1,
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
