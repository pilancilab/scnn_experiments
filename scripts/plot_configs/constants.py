"""Constants for creating plots.
"""

PLOTS = {}

line_colors = [
    "#000000",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#8c564b",
    "#17becf",
    "#556B2F",
    "#FFFF00",
    "#191970",
]

log_scale = {
    "train_objective": "log-linear",
    "train_squared_error": "log-linear",
    "test_squared_error": "log-linear",
    "train_grad_norm": "log-linear",
    "time": "linear-linear",
    "train_constraint_gaps": "log-linear",
}

y_labels = {
    "train_base_objective": "base objective",
    "train_objective": "objective",
    "train_nc_objective": "nc_objective",
    "train_squared_error": "train squared error",
    "test_squared_error": "test squared error",
    "test_nc_squared_error": "test squared error",
    "train_grad_norm": "gradient norm",
    "train_binary_accuracy": "train accuracy",
    "test_binary_accuracy": "test accuracy",
    "train_accuracy": "train accuracy",
    "train_nc_accuracy": "train nc accuracy",
    "test_accuracy": "test accuracy",
    "test_nc_accuracy": "test nc accuracy",
    "train_binned_accuracy": "train accuracy",
    "test_binned_accuracy": "test accuracy",
    "time": "time",
    "group_sparsity": "group sparsity",
    "feature_sparsity": "feature sparsity",
    "sparsity": "sparsity",
    "active_neurons": "active neurons",
    "active_features": "active features",
    "step_size": "step size",
    "num_backtracks": "backtracks",
    "sp_success": "successful line search",
    "train_constraint_gaps": "constraint gap",
}

x_labels = {
    "train_objective": "time",
    "train_squared_error": "time",
    "test_squared_error": "time",
    "train_grad_norm": "time",
    "train_binary_accuracy": "time",
    "test_binary_accuracy": "time",
    "train_accuracy": "train accuracy",
    "test_accuracy": "test accuracy",
    "time": "iterations",
    "group_sparsity": "time",
    "sparsity": "time",
    "active_neurons": "time",
    "step_size": "time",
    "num_backtracks": "time",
    "sp_success": "time",
    "train_constraint_gaps": "time",
}

iterations_x_labels = {
    "train_objective": "iterations",
    "train_squared_error": "iterations",
    "test_squared_error": "iterations",
    "train_grad_norm": "iterations",
    "train_binary_accuracy": "iterations",
    "test_binary_accuracy": "iterations",
    "train_accuracy": "train accuracy",
    "test_accuracy": "test accuracy",
    "time": "iterations",
    "group_sparsity": "iterations",
    "sparsity": "iterations",
    "active_neurons": "iterations",
    "step_size": "iterations",
    "num_backtracks": "iterations",
    "sp_success": "iterations",
    "train_constraint_gaps": "iterations",
}

limits = {
    "train_objective": (None, None),
    "train_squared_error": (None, None),
    "train_grad_norm": (None, None),
    "test_squared_error": (None, None),
    "train_binary_accuracy": (None, None),
    "test_binary_accuracy": (None, None),
    "time": (None, None),
}

line_kwargs = {
    "proximal_gd_ls_bb_prox_path_sm": {
        "c": line_colors[0],
        "label": "PGD-LS (BB, SM)",
    },
    "proximal_gd_ls_bb_grad_path": {
        "c": line_colors[1],
        "label": "PGD-LS (BB, GP)",
    },
    "proximal_gd_ls_forward_track_prox_path_sm": {
        "c": line_colors[2],
        "label": "PGD-LS (FT, SM)",
    },
    "proximal_gd_ls_forward_track_grad_path": {
        "c": line_colors[3],
        "label": "PGD-LS (FT, GP)",
    },
    "proximal_gd_ls_keep_old_prox_path_sm": {
        "c": line_colors[4],
        "label": "PGD-LS (Reset, SM)",
    },
    "proximal_gd_ls_keep_old_grad_path": {
        "c": line_colors[5],
        "label": "PGD-LS (Reset, GP)",
    },
    "fista_bb_prox_path_sm": {
        "c": line_colors[0],
        "label": "FISTA (BB, SM)",
    },
    "fista_bb_grad_path": {
        "c": line_colors[1],
        "label": "FISTA (BB, GP)",
    },
    "fista_forward_track_prox_path_sm": {
        "c": line_colors[2],
        "label": "FISTA (FT, SM)",
    },
    "fista_forward_track_grad_path": {
        "c": line_colors[3],
        "label": "FISTA (FT, GP)",
    },
    "fista_keep_old_prox_path_sm": {
        "c": line_colors[4],
        "label": "FISTA (Reset, SM)",
    },
    "fista_keep_old_grad_path": {
        "c": line_colors[5],
        "label": "FISTA (Reset, GP)",
    },
    "proximal_gd_0.001": {
        "c": line_colors[0],
        "label": "PGD (1e-3)",
    },
    "proximal_gd_0.0001": {
        "c": line_colors[1],
        "label": "PGD (1e-4)",
    },
    "proximal_gd_1e-05": {
        "c": line_colors[2],
        "label": "PGD (1e-5)",
    },
    "proximal_gd_1e-06": {
        "c": line_colors[3],
        "label": "PGD (1e-6)",
    },
    "proximal_gd_L_bar": {
        "c": line_colors[0],
        "label": "PGD (L)",
    },
    "proximal_gd_saga": {
        "c": line_colors[1],
        "label": "PGD (T)",
    },
    "saga_L_bar_0.01_shuffle": {
        "c": line_colors[0],
        "label": "SAGA (L_bar, 0.01, S)",
    },
    "saga_L_bar_0.05_shuffle": {
        "c": line_colors[1],
        "label": "SAGA (L_bar, 0.05, S)",
    },
    "saga_L_bar_0.1_shuffle": {
        "c": line_colors[2],
        "label": "SAGA (L_bar, 0.1, S)",
    },
    "saga_L_bar_0.01_lipschitz": {
        "c": line_colors[3],
        "label": "SAGA (L_bar, 0.01, L)",
    },
    "saga_L_bar_0.05_lipschitz": {
        "c": line_colors[4],
        "label": "SAGA (L_bar, 0.05, L)",
    },
    "saga_L_bar_0.1_lipschitz": {
        "c": line_colors[5],
        "label": "SAGA (L_bar, 0.1, L)",
    },
    "saga_saga_0.01_shuffle": {
        "c": line_colors[6],
        "label": "SAGA (T, 0.01, S)",
    },
    "saga_saga_0.05_shuffle": {
        "c": line_colors[7],
        "label": "SAGA (T, 0.05, S)",
    },
    "saga_saga_0.1_shuffle": {
        "c": line_colors[8],
        "label": "SAGA (T, 0.1, S)",
    },
    "saga_saga_0.01_lipschitz": {
        "c": line_colors[9],
        "label": "SAGA (T, 0.01, L)",
    },
    "saga_saga_0.05_lipschitz": {
        "c": line_colors[10],
        "label": "SAGA (T, 0.05, L)",
    },
    "saga_saga_0.1_lipschitz": {
        "c": line_colors[11],
        "label": "SAGA (T, 0.1, L)",
    },
    "fista_": {"c": line_colors[13], "label": "FISTA"},
    "fista_H_diag": {"c": line_colors[11], "label": "P FISTA"},
}
