"""
Experiment configurations.
"""
from typing import Dict, List

max_iters = 500

FISTA = {
    "name": ["fista"],
    "ls_cond": {"name": "quadratic_bound"},
    "backtrack_fn": {"name": "backtrack", "beta": 0.8},
    "step_size_update": [
        {
            "name": "lassplore",
            "alpha": 1.2,
            "threshold": 5,
        },
    ],
    "init_step_size": 0.1,
    "term_criterion": {"name": "grad_norm", "tol": 1e-8},
    "prox": {"name": "group_l1"},
    "ls_type": ["prox_path_sm"],
    "max_iters": max_iters,
    "metric_freq": 10,
    "restart_rule": "gradient_mapping",
}


EXPERIMENTS: Dict[str, List] = {
    "mnist": [
        {
            "method": FISTA,
            "model": {
                "name": "convex_mlp",
                "kernel": "einsum",
                "sign_patterns": {"name": "sampler", "n_samples": 5000},
                "regularizer": {
                    "name": "group_l1",
                    "lambda": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                },
                "c": 10,
                "initializer": {"name": "zero"},
            },
            "data": {
                "name": "mnist",
                "transforms": [["to_tensor", "normalize", "flatten"]],
            },
            "metrics": (
                ["objective", "grad_norm"],
                [],
                [],
            ),
            "final_metrics": (
                ["objective", "accuracy", "squared_error", "grad_norm"],
                ["accuracy", "squared_error"],
                [
                    "group_sparsity",
                ],
            ),
            "seed": 778,
            "repeat": list(range(1)),
            "backend": "torch",
            "device": "cuda",
            "dtype": "float32",
        },
    ],
}
