from copy import deepcopy
from typing import Dict, List
import numpy as np

max_iters = max_epochs = 1000
n_samples = [500, 1000, 10000]
d = 50
n_true_arrangements = 100
seeds = (650 + np.arange(10)).tolist()
lambda_to_try = 1e-4

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
    "max_iters": 1000,
    "max_total_iters": 10000,
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

# Non-Convex Formulations

torch_relu_mlp = {
    "name": "torch_mlp",
    "hidden_layers": [
        [
            {"name": "relu", "p": n_true_arrangements},
        ]
    ],
    "regularizer": {
        "name": "l2",
        "lambda": lambda_to_try,
    },
}

torch_sgd = {
    "name": "torch_sgd",
    "step_size": 10,
    "batch_size": 0.1,
    "max_epochs": max_epochs,
    "term_criterion": {"name": "grad_norm"},
    "scheduler": {"name": "step", "step_length": 100, "decay": 0.5},
}


synthetic_data = {
    "name": "synthetic_classification",
    "data_seed": 951,
    "n": 250,
    "n_test": 250,
    "d": d,
    "kappa": 10,
    "hidden_units": n_true_arrangements,
}

torch_metrics = (
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

convex_metrics = (
    [
        "objective",
        "base_objective",
        "nc_objective",
        "squared_error",
        "grad_norm",
        "constraint_gaps",
        "accuracy",
        "nc_accuracy",
    ],
    ["squared_error", "accuracy", "nc_accuracy"],
    [
        "group_sparsity",
        "active_neurons",
        "sp_success",
        "step_size",
        "num_backtracks",
    ],
)

torch_exp = {
    "method": torch_sgd,
    "model": torch_relu_mlp,
    "data": synthetic_data,
    "metrics": torch_metrics,
    "seed": 778,
    "repeat": 1,
    "backend": "torch",
    "device": "cuda",
    "dtype": "float32",
}

convex_exp = {
    "method": AL,
    "model": ConvexRelu_GL1,
    "data": synthetic_data,
    "metrics": convex_metrics,
    "seed": 778,
    "repeat": 1,
    "backend": "torch",
    "device": "cuda",
    "dtype": "float32",
}

seeded_torch_exps = []
active_set_exps = []
for seed in seeds:
    exp = deepcopy(torch_exp)
    # seed main experiment
    exp["seed"] = seed
    # seed layer initializations
    for layer_config in exp["model"]["hidden_layers"]:
        layer_config[0]["seed"] = seed
    seeded_torch_exps.append(exp)

    as_exp = deepcopy(convex_exp)
    as_exp["model"]["sign_patterns"]["seed"] = seed
    as_exp["model"]["sign_patterns"]["n_samples"] = n_true_arrangements
    as_exp["active_set"] = torch_exp
    active_set_exps.append(as_exp)


EXPERIMENTS: Dict[str, List] = {
    "figure_1": [
        {
            "method": AL,
            "model": ConvexRelu_GL1,
            "data": synthetic_data,
            "metrics": convex_metrics,
            "seed": 778,
            "repeat": 1,
            "backend": "torch",
            "device": "cuda",
            "dtype": "float32",
        },
    ]
    + seeded_torch_exps
    + active_set_exps,
}
