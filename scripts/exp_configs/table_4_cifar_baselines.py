"""
Configurations for experiments with pytorch training methods.
"""

from copy import deepcopy
from typing import Dict, List

import numpy as np

from experiment_utils import configs

from scaffold.uci_names import UCI_BIN_SUBSET

# global parameters
max_epochs = 1000
n_samples = 4000
lambda_to_try = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

# PyTorch models and optimizers

torch_gated_mlp = {
    "name": "torch_mlp_l1",
    "hidden_layers": [
        [
            {
                "name": "gated_relu",
                "sign_patterns": {"name": "sampler", "n_samples": n_samples, "conv_patterns": True},
            },
        ]
    ],
    "regularizer": {
        "name": "l2",
        "lambda": lambda_to_try,
    },
}


torch_optimizers = [
    {
        "name": ["torch_sgd"],
        "step_size": [1.0, 0.1, 0.01],
        "batch_size": 0.1,
        "max_epochs": max_epochs,
        "term_criterion": {"name": "grad_norm"},
        "scheduler": [
            {"name": "step", "step_length": 200, "decay": 0.5},
        ],
        "metric_freq": 10,
        "momentum": [0.9],
    },
    {
        "name": ["torch_adam"],
        "step_size": [1.0, 0.1, 0.01],
        "batch_size": 0.1,
        "max_epochs": max_epochs,
        "term_criterion": {"name": "grad_norm"},
        "scheduler": [
            {"name": "step", "step_length": 200, "decay": 0.5},
        ],
        "metric_freq": 10,
    },
    {
        "name": ["torch_adagrad"],
        "step_size": [1.0, 0.1, 1e-2],
        "batch_size": 0.1,
        "max_epochs": max_epochs,
        "term_criterion": {"name": "grad_norm"},
        "scheduler": [
            {"name": "step", "step_length": 200, "decay": 0.5},
        ],
        "metric_freq": 10,
    },
]


EXPERIMENTS: Dict[str, List] = {
    "cifar-baseline-updated": [
        {
            "method": torch_optimizers,
            "model": [torch_gated_mlp],
            "data": {
                "name": "cifar_10",
                "transforms": [["to_tensor", "normalize", "flatten"]],
            },
            "metrics":(
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
            "repeat": 1,
            "backend": "torch",
            "device": "cuda",
            "dtype": "float32",
        },
    ],
}
