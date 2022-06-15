"""
Testing new methods.
"""
from typing import Dict, List

import numpy as np

max_iters = 1000
lam = np.logspace(-5, -1, 5).tolist()

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
    "subprob_tol": 1e-6,
    "use_delta_init": True,
    "subproblem_solver": FISTA_GL1,
    "max_iters": 100,
    "batch_size": None,
}

ConvexGated_GL1 = {
    "name": "convex_mlp",
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": 1000,
        "seed": 650,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": lam,
    },
}


ConvexRelu_GL1 = {
    "name": "al_mlp",
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": 100,
        "seed": 650,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": lam,
    },
    "initializer": {"name": "zero"},
    "delta": 1000,
}


uci_data = {
    "name": [
        "breast-cancer",
        # "monks-1",
        # "planning",
        # "spect",
        # "haberman-survival",
        # "twonorm",
        # "trains",
        # "iris",
        # "ecoli"
    ],
    "split_seed": 1995,
    "use_valid": True,
}

metrics = (
    [
        "objective",
        "grad_norm",
        "nc_accuracy",
        "constraint_gaps",
    ],
    ["nc_accuracy"],
    [
        "group_sparsity",
    ],
)

Gated_EXPs = {
    "method": FISTA_GL1,
    "model": ConvexGated_GL1,
    "data": uci_data,
    "metrics": metrics,
    "final_metrics": metrics,
    "seed": 778,
    "repeat": 1,
    "backend": "numpy",
    "device": "cpu",
    "dtype": "float64",
}

Relu_EXPs = {
    "method": AL,
    "model": ConvexRelu_GL1,
    "data": uci_data,
    "metrics": metrics,
    "final_metrics": metrics,
    "seed": 778,
    "repeat": 1,
    "backend": "numpy",
    "device": "cpu",
    "dtype": "float64",
}

EXPERIMENTS: Dict[str, List] = {
    "test": [Gated_EXPs, Relu_EXPs],
}
