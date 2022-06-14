"""
Default/base configuration objects for experiments.
"""
from copy import deepcopy

from scaffold.uci_names import BINARY_SMALL_UCI_DATASETS, UCI_BIN_SUBSET


# ====================== #
# ===== Optimizers ===== #
# ====================== #

# ===== Least-Squares Solvers ===== #
LSMR = {
    "name": "lsmr",
    "preconditioner": None,
}

LSQR = {
    "name": "lsqr",
    "preconditioner": None,
}

# ===== CVXPY Solvers ===== #

# choose one or more solvers before using. Note that MOSEK and Gurobi require licenses.
CVXPY = {"name": "cvxpy", "solver": ["ecos", "mosek", "gurobi"]}

# ===== (Sub)gradient Descent ===== #

GD = {
    "name": "gd",
    "step_size_update": {"name": "keep_old"},
    "term_criterion": {"name": "grad_norm"},
    "step_size": None,  # set before using.
    "max_iters": None,  # set before using.
}


# ===== (Sub)gradient Descent w/ Line Search ===== #

GD_LS = {
    "name": "gd_ls",
    "ls_cond": {"name": "armijo", "rho": 0.5},
    "backtrack_fn": {"name": "backtrack", "beta": 0.8},
    "step_size_update": {
        "name": "lassplore",
        "alpha": 1.25,
        "threshold": 5.0,
    },
    "init_step_size": 1.0,
    "term_criterion": {"name": "grad_norm"},
    "max_iters": None,  # set before using
}

# ===== Proximal Gradient ===== #

PGD = {
    "name": "proximal_gd",
    "step_size_update": {"name": "keep_old"},
    "term_criterion": {"name": "grad_norm"},
    "prox": {"name": "group_l1"},  # default is group l1.
    "step_size": None,  # set before using.
    "max_iters": None,  # set before using.
}

# ===== Proximal Gradient w/ Line search ===== #

PGD_LS = {
    "name": "proximal_gd_ls",
    "ls_cond": {"name": "quadratic_bound"},
    "backtrack_fn": {"name": "backtrack", "beta": 0.8},
    "step_size_update": {
        "name": "lassplore",
        "alpha": 1.25,
        "threshold": 5.0,
    },
    "init_step_size": 1e-2,
    "term_criterion": {"name": "grad_norm"},
    "ls_type": "prox_path",
    "prox": {"name": "group_l1"},  # default is group l1
    "max_iters": None,  # set before using.
}

# ===== FISTA (e.g. accelerated proximal-gradient) ===== #

# base version of FISTA. The "prox" field must be set before use.
FISTA_BASE = {
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
    "max_iters": None,  # set before using.
    "prox": None,  # set before using.
}

# FISTA for group-l1 problems.
FISTA_GL1 = deepcopy(FISTA_BASE)
FISTA_GL1["prox"] = {"name": "group_l1"}


# ===== Preconditioned FISTA ===== #

P_FISTA = {
    "name": "p_fista",
    "ls_cond": {"name": "precon_quadratic_bound"},
    "backtrack_fn": {"name": "backtrack", "beta": 0.8},
    "step_size_update": {
        "name": "forward_track",
        "alpha": 1.25,
    },
    "init_step_size": 1.0,
    "term_criterion": {"name": "grad_norm"},
    "ls_type": ["prox_path"],
    "preconditioner": "H_diag",
    "prox": {
        "name": "diag_group_l1",
        "tol": 1e-6,
        "max_iters": 10000,
    },  # only support this prox for now.
    "max_iters": None,  # set before using.
}

# ===== SAGA ===== #


SAGA = {
    "name": "saga",
    "step_size": "saga",  # set size suggested by SAGA theory.
    "step_size_update": {"name": "keep_old"},
    "term_criterion": {"name": "grad_norm"},
    "prox": {"name": "group_l1"},
    "sampling_scheme": ["shuffle"],
    "batch_size": None,  # set before using.
    "max_epochs": None,  # set before using.
}

# ===== Augmented Lagrangian ===== #

# "regular" augmented Lagrangian method
AL = {
    "name": "augmented_lagrangian",
    "term_criterion": {
        "name": "constrained_opt",
        "grad_tol": 1e-6,
        "constraint_tol": 1e-6,
    },
    "use_delta_init": True,  # use heuristic to set delta
    "subproblem_solver": FISTA_GL1,  # defaults to FISTA w/ correct prox.
    "max_iters": None,  # set before using
    #
    # additional configuration parameters for augmented Lagrangian method
    "subprob_tol": 1e-6,
    "omega": 1e-2,
    "eta_upper": 1e-1,
    "eta_lower": 1e-3,
    "tau": 2,
}

#  dual accelerated augmented Lagrangian method.
accAL = {
    "name": "acc_augmented_lagrangian",
    "term_criterion": {
        "name": "constrained_opt",
        "grad_tol": 1e-6,
        "constraint_tol": 1e-6,
    },
    "use_delta_init": True,  # use heuristic to set delta
    "subproblem_solver": FISTA_GL1,  # defaults to FISTA w/ correct prox.
    "max_iters": None,  # set before using
}


# PyTorch Training Methods #

SGD_Torch = {
    "name": "torch_sgd",
    "step_size": None,  # must set before use
    "batch_size": None,  # must set before use; can be specified as a integer or a proportion of the dataset.
    "max_iters": None,  # must set before use
    "term_criterion": {"name": "grad_norm"},
}

# ===================== #
# ===== Callbacks ===== #
# ===================== #

# Expand the set of hyperplane arrangements by simultaneously training an MLP.
MLPActiveSet = {
    "name": "mlp_active_set",
    "model": {
        "name": "relu_mlp",
        "p": None,  # will be inferred from the convex MLP
        "regularizer": None,  # set before using.
    },
    "method": None,  # choose an optimizer for the ReLU MLP.
}

# ================== #
# ===== Models ===== #
# ================== #

# Convex MLPs with "gated" Relu Activations #
ConvexGated_L2 = (
    {
        "name": "convex_mlp",
        "kernel": "einsum",
        "sign_patterns": {
            "name": "sampler",
            "n_samples": None,  # number of patterns to sample should be set before use.
            "seed": 650,
        },
        "regularizer": {
            "name": "l2",
            "lambda": None,  # lambda should be set before use.
        },
    },
)

ConvexGated_GL1 = {
    "name": "convex_mlp",
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": None,  # number of patterns to sample should be set before use.
        "seed": 650,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": None,  # lambda should be set before use.
    },
}

# Convex MLPs with Relu Activations #

ConvexRelu_GL1 = {
    "name": ["al_mlp"],
    "kernel": "einsum",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": None,  # number of patterns to sample should be set before use.
        "seed": 650,
    },
    "regularizer": {
        "name": "group_l1",
        "lambda": None,  # lambda should be set before use.
    },
    "initializer": {"name": ["least_squares"]},
    "delta": 100,
}


# Non-convex MLPs #

# Models Implemented in PyTorch #
# Note: you must use PyTorch optimizers with these models.

NonConvexGated_Torch = {
    "name": "torch_mlp",
    "layers": [
        [
            {
                "name": "gated_relu",
                "sign_patterns": {
                    "name": "sampler",
                    "seed": 650,
                    "n_samples": None,  # must set before use.
                },
            },
            {"name": "linear", "width": 1},
        ]
    ],
    "weight_decay": None,  # Must set before use.
}

NonConvexRelu_Torch = {
    "name": "torch_mlp",
    "layers": [
        [
            {
                "name": "relu",
                "p": None,  # Must specify number of units before use.
            },
            {"name": "linear", "width": 1},
        ]
    ],
    "weight_decay": None,  # weight decay parameter must be set before use.
}

# Manually Implemented MLPs #

NonConvexGated_Manual = {
    "name": "relu_mlp",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": None,  # number of patterns to sample should be set before use.
        "seed": 650,
    },
    "regularizer": {
        "name": "l2",
        "lambda": None,  # set before use.
    },
    "initializer": None,  # set before use
}

NonConvexReLU_Manual = {
    "name": "relu_mlp",
    "p": None,  # must specify number of hidden units before use.
    "regularizer": {
        "name": "l2",
        "lambda": None,  # set before use.
    },
    "initializer": None,  # set before use
}

# ======================== #
# ===== Initializers ===== #
# ======================== #


# initialize at zero
ZERO = {"name": "zero"}
# initialize at a vector drawn from standard normal distribution.
ZERO = {"name": "random"}
# initialize at the (regularized) least-squares solution.
LEAST_SQUARES = {"name": "least_squares"}
# initialize at the vectors generating the hyperplane arrangements.
GATES = {
    "name": "gates",
    "sign_patterns": {
        "name": "sampler",
        "n_samples": 50,
        "seed": 650,
    },
}


# ==================== #
# ===== Datasets ===== #
# ==================== #

synthetic_data = {
    "name": "synthetic_regression",
    "data_seed": 951,
    "n": 1000,
    "n_test": 1000,
    "d": 50,
    "kappa": [1, 10],
    "sigma": 0.1,
}

binary_uci_data = {
    "name": BINARY_SMALL_UCI_DATASETS,
    "split_seed": 1995,
    "test_prop": 0.2,
    "valid_prop": 0.2,
    "use_valid": True,
}

subset_binary_uci_data = {
    "name": UCI_BIN_SUBSET,
    "split_seed": 1995,
    "test_prop": 0.2,
    "valid_prop": 0.2,
    "use_valid": True,
}

# =================== #
# ===== Metrics ===== #
# =================== #

# binary classification metrics.

metrics = (
    ["objective", "squared_error", "binary_accuracy", "grad_norm"],
    ["squared_error", "binary_accuracy"],
    [
        "group_sparsity",
        "active_neurons",
        "sp_success",
        "step_size",
        "num_backtracks",
        "subproblem_metrics",
    ],
)


# =========================== #
# ===== Example Configs ===== #
# =========================== #

# This is *only* an example experiment config;
# it won't run until lambda, max_iters, etc. are instantiated.

EXPERIMENTS = {
    "uci-l2": [
        {
            "method": [LSMR, LSQR],
            "model": ConvexGated_L2,
            "metrics": metrics,
            "data": binary_uci_data,
            "seed": 778,
            "repeat": 1,
            "backend": "torch",  # Linear algebra backend (choose 'torch' or 'numpy')
            "device": "cuda",  # Device to run on (choose 'cuda' or 'cpu')
            "dtype": "float32",  # precision (choose 'float32' or 'float34')
        },
    ],
    "uci-gl1": [
        {
            "method": FISTA_GL1,
            "model": ConvexGated_GL1,
            "metrics": metrics,
            "data": binary_uci_data,
            "seed": 778,
            "repeat": 1,
            "backend": "numpy",  # Linear algebra backend (choose 'torch' or 'numpy')
            "device": "cpu",  # Device to run on (choose 'cuda' or 'cpu')
            "dtype": "float64",  # precision (choose 'float32' or 'float34')
        }
    ],
}
