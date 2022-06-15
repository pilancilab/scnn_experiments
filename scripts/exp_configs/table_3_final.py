"""
Initial attempt at performance profiles.
"""
from copy import deepcopy
from typing import Dict, List
import numpy as np
import pickle as pkl
from experiment_utils import configs

from scaffold.uci_names import REGULARIZATION_PATH


max_iters = 2000
gate_n_samples = 5000
relu_n_samples = 2500
outer_iters = 5000
max_total_iters = 10000
arrangement_seed = (650 + np.array([0, 1, 2, 3, 4])).tolist()

torch_gated_mlp = {
    "name": "torch_mlp_l1",
    "hidden_layers": [
        [
            [
                {
                    "name": "gated_relu",
                    "sign_patterns": {
                        "name": "sampler",
                        "n_samples": gate_n_samples,
                        "seed": seed,
                    },
                },
            ]
        ]
        for seed in arrangement_seed
    ],
    "regularizer": {
        "name": "l2",
        "lambda": None,
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
        "lambda": None,
    },
}

torch_optimizers_template = {
    "name": None,
    "step_size": None,
    "batch_size": 0.1,
    "max_epochs": max_iters,
    "term_criterion": {"name": "grad_norm"},
    "scheduler": {"name": "step", "step_length": 100, "decay": 0.5},
}

torch_optimizers = {
    "name": ["torch_sgd", "torch_adam"],
    "step_size": None,
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
        "name": "constrained_heuristic",
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
        "lambda": None,
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
        "seed": arrangement_seed,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": None,
    },
    "initializer": {"name": ["zero"]},
    "delta": 100,
}

uci_data = {
    "name": REGULARIZATION_PATH,
    "split_seed": 1995,
    "use_valid": False,
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

try:
    with open("scripts/exp_configs/table_3_best_params.pkl", "rb") as f:
        best_params = pkl.load(f)

    convex_relu = {
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

    expanded_al = configs.expand_config(convex_relu)
    al_sgd_configs = []
    al_adam_configs = []
    al_configs = []

    nc_relu: List[Dict] = []

    for config in expanded_al:
        best_al = deepcopy(config)
        best_al["model"]["regularizer"]["lambda"] = best_params[
            config["data"]["name"]
        ]["augmented_lagrangian"]["key"][1]
        al_configs.append(best_al)

        adam_params = best_params[config["data"]["name"]]["torch_adam_relu"]
        adam_al = deepcopy(config)
        adam_al["model"]["regularizer"]["lambda"] = adam_params["key"][1]
        al_adam_configs.append(adam_al)

        sgd_params = best_params[config["data"]["name"]]["torch_sgd_relu"]
        sgd_al = deepcopy(config)
        sgd_al["model"]["regularizer"]["lambda"] = sgd_params["key"][1]
        al_sgd_configs.append(sgd_al)

        adam_nc = deepcopy(adam_al)
        adam_nc["src_hash"] = configs.hash_dict(adam_al)
        # Note: must be updated to your system-specific path
        adam_nc[
            "src_dir"
        ] = "/scratch/users/mishkin/results/table_3_relu_support"

        adam_nc["method"] = deepcopy(torch_optimizers_template)
        adam_nc["method"]["name"] = "torch_adam"
        adam_nc["method"]["step_size"] = adam_params["key"][0]
        torch_relu_copy = deepcopy(torch_relu_mlp)
        torch_relu_copy["regularizer"]["lambda"] = adam_params["key"][1]
        adam_nc["model"] = torch_relu_copy
        nc_relu.append(adam_nc)

        sgd_nc = deepcopy(sgd_al)
        sgd_nc["src_hash"] = configs.hash_dict(sgd_al)
        # Note: must be updated to your system-specific path
        sgd_nc[
            "src_dir"
        ] = "/scratch/users/mishkin/results/table_3_relu_support"
        sgd_nc["method"] = deepcopy(torch_optimizers_template)
        sgd_nc["method"]["name"] = "torch_sgd"
        sgd_nc["method"]["step_size"] = sgd_params["key"][0]
        torch_relu_copy = deepcopy(torch_relu_mlp)
        torch_relu_copy["regularizer"]["lambda"] = sgd_params["key"][1]
        sgd_nc["model"] = torch_relu_copy
        nc_relu.append(sgd_nc)

    convex_gated = {
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
    }

    nc_gated = {
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
    }

    fista_configs = configs.expand_config(convex_gated)
    for config in fista_configs:
        config["model"]["regularizer"]["lambda"] = best_params[
            config["data"]["name"]
        ]["fista"]["key"][1]

    nc_gated_configs = configs.expand_config(nc_gated)

    for config in nc_gated_configs:
        method_key = config["method"]["name"] + "_gated_relu"
        params = best_params[config["data"]["name"]][method_key]["key"]
        config["model"]["regularizer"]["lambda"] = params[1]
        config["method"]["step_size"] = params[0]

    EXPERIMENTS: Dict[str, List] = {
        "table_3_grelu_final": nc_gated_configs + fista_configs,
        "table_3_relu_final": al_configs,
        "table_3_relu_support": al_adam_configs + al_sgd_configs,
        "table_3_nc_relu_final": nc_relu,
    }
except:
    pass
