"""
Compare optimization speed for convex and non_convex methods.
"""

from typing import Dict, List

import numpy as np

max_iters = 10000
gated_width = 2500
relu_width = 500
arrangement_seed = 779
lambda_to_try = [1e-7]

TorchGated = {
    "name": "torch_mlp_l1",
    "hidden_layers": [
        [
            {
                "name": "gated_relu",
                "sign_patterns": {
                    "name": "sampler",
                    "n_samples": gated_width,
                    "seed": arrangement_seed,
                },
            },
        ]
    ],
    "regularizer": {
        "name": "l2",
        "lambda": lambda_to_try,
    },
}

TorchReLU = {
    "name": "torch_mlp_l1",
    "hidden_layers": [
        [
            {
                "name": "relu",
                "p": relu_width,
            },
        ]
    ],
    "regularizer": {
        "name": "l2",
        "lambda": lambda_to_try,
    },
}

TorchOptim = {
    "name": ["torch_sgd"],
    "step_size": [0.001, 0.01, 0.1, 1, 5, 10, 25, 50, 100],
    "batch_size": 0.1,
    "max_epochs": max_iters,
    "term_criterion": {"name": "grad_norm", "tol": 1e-10},
    "scheduler": {"name": "step", "step_length": 100, "decay": 0.5},
    "metric_freq": 10,
}


FISTA = {
    "name": ["fista"],
    "ls_cond": {"name": "quadratic_bound"},
    "backtrack_fn": {"name": "backtrack", "beta": 0.8},
    "step_size_update": [
        {
            "name": "lassplore",
            "alpha": 1.25,
            "threshold": 5,
        },
    ],
    "init_step_size": 0.1,
    "term_criterion": {"name": "grad_norm", "tol": 1e-10},
    "prox": {"name": "group_l1"},
    "max_iters": max_iters,
    "restart_rule": "gradient_mapping",
    "metric_freq": 10,
}


GReLU_GL1 = {
    "name": "convex_mlp",
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": gated_width,
        "seed": arrangement_seed,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": lambda_to_try,
    },
    "c": 10,
    "initializer": {"name": "zero"},
}

data = {
    "name": "mnist",
    "transforms": [["to_tensor", "normalize", "flatten"]],
    "use_valid": False,
}

metrics = (
    ["objective", "grad_norm", "nc_accuracy"],
    ["nc_accuracy"],
    ["group_sparsity"],
)

final_metrics = (
    ["objective", "accuracy", "squared_error", "grad_norm"],
    ["nc_accuracy", "squared_error"],
    [
        "group_sparsity",
    ],
)

ConvexGated = {
    "method": FISTA,
    "model": GReLU_GL1,
    "data": data,
    "metrics": metrics,
    "final_metrics": final_metrics,
    "seed": 778,
    "repeat": list(range(1)),
    "backend": "torch",
    "device": "cuda",
    "dtype": "float32",
}

NonConvex = {
    "method": TorchOptim,
    "model": [TorchReLU, TorchGated],
    "data": data,
    "metrics": metrics,
    "final_metrics": final_metrics,
    "seed": 778,
    "repeat": list(range(1)),
    "backend": "torch",
    "device": "cuda",
    "dtype": "float32",
}

EXPERIMENTS: Dict[str, List] = {
    "mnist_timing_convex": [ConvexGated],
    "mnist_timing_non_convex": [NonConvex],
}
