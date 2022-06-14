"""
Initial attempt at performance profiles.
"""
from copy import deepcopy
from typing import Dict, List
import numpy as np
from experiment_utils import configs

from scaffold.uci_names import REGULARIZATION_PATH


max_iters = 2000
gate_n_samples = 5000
relu_n_samples = 2500
outer_iters = 5000
max_total_iters = 10000
arrangement_seed = 650
lambda_to_try = np.logspace(-6, -2, 10).tolist()

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

uci_data = {
    "name": REGULARIZATION_PATH,
    "split_seed": 1995,
    "n_folds": 5,
    "fold_index": list(range(5)),
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

expanded_al = configs.expand_config(AL_EXPS)
non_convex: List[Dict] = []

for config in expanded_al:
    config_copy = deepcopy(config)
    config_copy["src_hash"] = configs.hash_dict(config)
    # Note: must be updated to your system-specific path
    config_copy["src_dir"] = "/scratch/users/mishkin/results/table_3_relu_gs"

    config_copy["method"] = torch_optimizers
    torch_relu_copy = deepcopy(torch_relu_mlp)
    torch_relu_copy["regularizer"]["lambda"] = config_copy["model"][
        "regularizer"
    ]["lambda"]
    config_copy["model"] = torch_relu_copy

    non_convex.append(config_copy)

EXPERIMENTS: Dict[str, List] = {
    "table_3_relu_gs": [AL_EXPS],
    "table_3_nc_relu_gs": non_convex,
    "table_3_grelu_gs": [
        {
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
        },
        {
            "method": torch_optimizers,
            "model": torch_gated_mlp,
            "data": uci_data,
            "metrics": metrics,
            "final_metrics": final_metrics,
            "seed": 778,
            "repeat": 1,
            "backend": "torch",
            "device": "cuda",
            "dtype": "float32",
        },
    ],
}
