from copy import deepcopy
from typing import Dict, List
import numpy as np
from experiment_utils import configs

from scaffold.uci_names import PERFORMANCE_PROFILE, two_class


max_iters = 2000
gate_n_samples = 5000
relu_n_samples = 2500
outer_iters = 5000
max_total_iters = 10000
arrangement_seed = 650
lambda_to_try = np.logspace(-5, -1, 6).tolist()

torch_gated_mlp = {
    "name": "torch_mlp_l1",
    "hidden_layers": [
        [
            {
                "name": "gated_relu",
                "sign_patterns": {
                    "name": "sampler",
                    "n_samples": gate_n_samples,
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

torch_relu_mlp = {
    "name": "torch_mlp_l1",
    "hidden_layers": [
        [
            {
                "name": "relu",
                "p": "m_star",
            },
        ]
    ],
    "regularizer": {
        "name": "l2",
        "lambda": lambda_to_try,
    },
}

torch_optimizers = {
    "name": ["torch_adam", "torch_sgd"],
    "step_size": [10, 5, 1, 0.5, 0.1, 0.01, 0.001],
    "batch_size": 0.1,
    "max_epochs": max_iters,
    "term_criterion": {"name": "grad_norm"},
    "scheduler": {"name": "step", "step_length": 100, "decay": 0.5},
}


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
        "name": "constrained_opt",
        "grad_tol": 1e-6,
        "constraint_tol": 1e-6,
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
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": lambda_to_try,
    },
    "initializer": {"name": ["zero"]},
    "delta": 100,
}

CVXPY_Relu_MLP = {
    "name": "ineq_lagrangian",
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": relu_n_samples,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": lambda_to_try,
    },
    "initializer": {"name": ["zero"]},
    "delta": 100,
}

CVXPY = {"name": "cvxpy", "solver": "mosek"}


uci_data = {
    "name": PERFORMANCE_PROFILE,
    "split_seed": 1995,
    "use_valid": False,
}
two_class = {"name": two_class, "split_seed": 1995, "use_valid": False}

metrics = (
    ["objective", "grad_norm"],
    [],
    ["active_neurons", "num_backtracks"],
)

relu_metrics = (
    ["objective", "base_objective", "nc_objective", "grad_norm"],
    [],
    ["active_neurons", "num_backtracks"],
)

cvxpy_metrics = (
    ["objective", "grad_norm", "accuracy", "squared_error"],
    ["accuracy", "squared_error"],
    ["active_neurons", "num_backtracks"],
)

gated_final_metrics = (
    ["accuracy", "squared_error"],
    ["accuracy", "squared_error"],
    [],
)

final_metrics = (
    ["accuracy", "squared_error", "constraint_gaps"],
    ["accuracy", "squared_error"],
    [],
)

AL_EXPS = {
    "method": AL,
    "model": ConvexRelu_GL1,
    "data": uci_data,
    "metrics": relu_metrics,
    "final_metrics": final_metrics,
    "seed": 778,
    "repeat": 1,
    "backend": "torch",
    "device": "cuda",
    "dtype": "float32",
}

expanded_al = configs.expand_config(AL_EXPS)
non_convex: List[Dict] = []

for config in expanded_al:
    config_copy = deepcopy(config)
    config_copy["src_hash"] = configs.hash_dict(config)

    # Note: must be updated to your own machine-specific path.
    config_copy["src_dir"] = "/scratch/users/mishkin/results/figure_4_relu"

    config_copy["method"] = torch_optimizers
    torch_relu_copy = deepcopy(torch_relu_mlp)
    torch_relu_copy["regularizer"]["lambda"] = config_copy["model"][
        "regularizer"
    ]["lambda"]
    config_copy["model"] = torch_relu_copy
    config_copy["repeat"] = list(range(3))

    non_convex.append(config_copy)

EXPERIMENTS: Dict[str, List] = {
    "figure_4_relu": expanded_al,
    "figure_4_nc_relu": non_convex,
    "figure_4_grelu": [
        {
            "method": FISTA_GL1,
            "model": ConvexGated_GL1,
            "data": uci_data,
            "metrics": metrics,
            "final_metrics": gated_final_metrics,
            "seed": 778,
            "repeat": 1,
            "backend": "torch",
            "device": "cuda",
            "dtype": "float32",
        },
        {
            "method": torch_optimizers,
            "model": torch_gated_mlp,
            "data": uci_data,
            "metrics": metrics,
            "final_metrics": gated_final_metrics,
            "seed": 778,
            "repeat": list(range(3)),
            "backend": "torch",
            "device": "cuda",
            "dtype": "float32",
        },
    ],
    "figure_4_grelu_cvxpy": [
        {
            "method": CVXPY,
            "model": ConvexGated_GL1,
            "data": uci_data,
            "metrics": cvxpy_metrics,
            "seed": 778,
            "repeat": 1,
            "backend": "numpy",
            "device": "cpu",
            "dtype": "float32",
        },
    ],
    "figure_4_relu_cvxpy": [
        {
            "method": CVXPY,
            "model": CVXPY_Relu_MLP,
            "data": uci_data,
            "metrics": cvxpy_metrics,
            "seed": 778,
            "repeat": 1,
            "backend": "numpy",
            "device": "cpu",
            "dtype": "float32",
        },
    ],
}
