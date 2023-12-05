"""
Loaders for Methods.
"""

from logging import Logger
from copy import deepcopy
from typing import Dict, Any, Tuple, Union, Optional, cast, Callable

import numpy as np
import torch

import lab

from scnn.private.models import (
    Model,
    ConvexMLP,
    AL_MLP,
)

from scnn.private.prox import ProximalOperator

from scnn.private.utils.linear import iterative_solvers
from scnn.private.methods.method_utils import (
    init_batch_size,
    init_max_epochs,
)

from scnn.private.methods.optimization_procedures import (
    METRIC_FREQ,
    ITER_LOG_FREQ,
    EPOCH_LOG_FREQ,
)

from scnn.private.methods import (
    ConeDecomposition,
    ApproximateConeDecomposition,
    OptimizationProcedure,
    IterativeOptimizationProcedure,
    TorchLoop,
    DoubleLoopProcedure,
    ObservedSignPatterns,
    ls,
    gd_ls,
    proximal_gradient_step,
    proximal_gradient_ls,
    fista_step,
    fista_ls,
    update_multipliers,
    acc_update_multipliers,
    Optimizer,
    ProximalOptimizer,
    ProximalLSOptimizer,
    MetaOptimizer,
    GD,
    GDLS,
    PGD,
    PGDLS,
    FISTA,
    AugmentedLagrangian,
    LinearSolver,
    CVXPYSolver,
    CVXPYGatedReLUSolver,
    CVXPYReLUSolver,
    MinL2Decomposition,
    MinL1Decomposition,
    FeasibleDecomposition,
    MinRelaxedL2Decomposition,
    SOCPDecomposition,
    Backtracker,
    MultiplicativeBacktracker,
    LSCondition,
    QuadraticBound,
    DiagQB,
    Armijo,
    StepSizeUpdater,
    KeepNew,
    KeepOld,
    ForwardTrack,
    Lassplore,
    GradientNorm,
    StepLength,
)

# load scaffolding

from .prox import get_proximal_op
from .models import get_model, get_regularizer

from .line_search import (
    get_update_fn,
    get_ls_condition,
    get_backtrack_fn,
)

from .termination_criteria import get_criterion

MetaOptimizers = ["augmented_lagrangian"]


def get_method(
    logger: Logger,
    rng: np.random.Generator,
    model: Model,
    train_set,
    method_config: Dict[str, Any],
) -> OptimizationProcedure:
    """Load an optimization method by name using the passed configuration parameters.
    :param logger: a logger instance.
    :param rng: a NumPy random number generator.
    :param model: the model to be optimized.
    :param train_set: an (X,y) tuple containing the training set.
    :param method_config: configuration object specifying the optimization method to load.
    :returns: an optimization procedure that can be called to fit a model.
    """

    method_name = method_config.get("name", None)
    log_freq = method_config.get("log_freq", ITER_LOG_FREQ)
    metric_freq = method_config.get("metric_freq", METRIC_FREQ)
    batch_size = method_config.get("batch_size", None)

    # load ridge-regression solver.
    if method_name in iterative_solvers.SOLVERS:
        preconditioner = method_config.get("preconditioner", None)
        max_iters = method_config.get("max_iters", 1000)
        tol = method_config.get("tol", iterative_solvers.TOL)

        linear_solver = LinearSolver(
            method_name, max_iters, tol, preconditioner
        )

        return OptimizationProcedure(linear_solver)
    # load CVXPY-based solver
    elif method_name == "cvxpy":
        solver_name = method_config.get("solver", "ecos")
        cvxpy_solver: CVXPYSolver

        if isinstance(model, AL_MLP):
            cvxpy_solver = CVXPYReLUSolver(solver_name)
        elif isinstance(model, ConvexMLP):
            cvxpy_solver = CVXPYGatedReLUSolver(solver_name)
        else:
            raise ValueError(
                f"Model type {type(model)} not supported by CVXPY solvers."
            )

        return OptimizationProcedure(cvxpy_solver)

    # load meta optimization procedure.
    elif method_name in MetaOptimizers:
        outer_optimizer = get_optimizer(
            rng, model, train_set, None, method_config
        )
        outer_term_config = method_config.get("term_criterion", None)
        outer_term_criterion = get_criterion(outer_term_config)

        subproblem_solver = get_method(
            logger, rng, model, train_set, method_config["subproblem_solver"]
        )
        inner_optimizer = subproblem_solver.optimizer
        inner_term_criterion = subproblem_solver.term_criterion
        inner_max_iters = subproblem_solver.max_iters
        outer_max_iters = method_config.get("max_iters", 250)
        max_total_iters = method_config.get("max_total_iters", None)

        return DoubleLoopProcedure(
            inner_optimizer,
            outer_optimizer,
            inner_max_iters,
            outer_max_iters,
            inner_term_criterion,
            outer_term_criterion,
            method_name,
            divergence_check=False,
            batch_size=batch_size,
            log_freq=log_freq,
            metric_freq=metric_freq,
            max_total_iters=max_total_iters,
        )

    else:
        if "mini_batch_size" in method_config:
            batch_size = init_batch_size(
                train_set, method_config["mini_batch_size"]
            )

        step_size = method_config.get(
            "step_size", method_config.get("init_step_size", 1.0)
        )

        optimizer = get_optimizer(
            rng, model, train_set, step_size, method_config
        )

        term_config = method_config.get("term_criterion", None)
        term_criterion = get_criterion(term_config)
        divergence_check = method_config.get("divergence_check", True)

        if isinstance(model, torch.nn.Module):  # use PyTorch training loop
            batch_size = init_batch_size(train_set, batch_size)

            max_epochs = init_max_epochs(
                train_set,
                method_config.get("max_epochs", None),
                method_config.get("max_iters", None),
                batch_size=batch_size,
            )

            scheduler = get_scheduler(
                method_config.get("scheduler", None), optimizer
            )

            return TorchLoop(
                optimizer,
                max_epochs,
                batch_size,
                term_criterion,
                name=method_name,
                divergence_check=divergence_check,
                log_freq=method_config.get("log_freq", EPOCH_LOG_FREQ),
                metric_freq=metric_freq,
                scheduler=scheduler,
            )
        else:
            # use a deterministic training loop.
            max_iters = method_config.get("max_iters", 250)

            post_process = None
            if method_config.get("post_process", None) is not None:
                post_process = get_callback(method_config["post_process"])

            return IterativeOptimizationProcedure(
                optimizer,
                max_iters,
                term_criterion,
                name=method_name,
                divergence_check=divergence_check,
                batch_size=batch_size,
                log_freq=method_config.get("log_freq", ITER_LOG_FREQ),
                metric_freq=metric_freq,
                post_process=post_process,
            )


def get_optimizer(
    rng: np.random.Generator,
    model: Model,
    train_set: Tuple[lab.Tensor, lab.Tensor],
    step_size: float,
    method_config: Dict[str, Any],
) -> Optimizer:
    """Lookup and initialize an optimization method using a configuration dictionary.
    :param model: the model to be optimized.
    :param train_set: (X, y) tuple with the training data.
    :param step_size: the initial step-size for the optimizer.
    :param method_config: the configuration options for the optimization method.
    :returns: an Optimizer instance.
    """

    # Unpack configuration.
    d = train_set[0].shape[1]

    name = method_config.get("name", None)
    update_step_size = get_update_fn(
        method_config.get("step_size_update", {"name": "keep_old"})
    )

    prox_config = method_config.get("prox", None)
    backtrack_config = method_config.get("backtrack_fn", None)
    ls_config = method_config.get("ls_cond", None)

    prox = load_proximal_op(model, d, prox_config)

    if backtrack_config is not None:
        backtrack_fn = get_backtrack_fn(backtrack_config)

    if ls_config is not None:
        ls_cond = get_ls_condition(ls_config)

    # ===== load optimizers ===== #

    if name is None:
        raise ValueError(
            "The method configuration must include a method name!"
        )
        assert prox is not None

    elif name == "gd":
        return GD(
            step_size=step_size,
            update_step_size=update_step_size,
        )

    elif name == "gd_ls":
        return GDLS(
            step_size,
            ls_cond,
            backtrack_fn,
            update_step_size,
        )

    elif name == "proximal_gd":
        return PGD(
            step_size=step_size,
            prox=prox,
            update_step_size=update_step_size,
        )

    elif name == "proximal_gd_ls":
        return PGDLS(
            step_size,
            ls_cond,
            backtrack_fn,
            update_step_size,
            prox=prox,
        )

    elif name == "fista":
        return FISTA(
            step_size,
            ls_cond,
            backtrack_fn,
            update_step_size,
            prox=prox,
            restart_rule=method_config.get("restart_rule", None),
        )

    elif name == "augmented_lagrangian":
        return AugmentedLagrangian(
            use_delta_init=method_config.get("use_delta_init", True),
            subprob_tol=method_config.get("subprob_tol", 1e-6),
            omega=method_config.get("omega", 1e-3),
            eta_upper=method_config.get("eta_upper", 1e-2),
            eta_lower=method_config.get("eta_lower", 1e-3),
            tau=method_config.get("tau", 2),
        )

    elif name == "torch_sgd":
        assert isinstance(model, torch.nn.Module)
        momentum = method_config.get("momentum", 0.0)

        return torch.optim.SGD(
            model.parameters(), lr=step_size, momentum=momentum
        )
    elif name == "torch_adam":
        assert isinstance(model, torch.nn.Module)
        betas = method_config.get("betas", (0.9, 0.999))
        eps = method_config.get("eps", 1e-8)

        return torch.optim.Adam(
            model.parameters(), lr=step_size, betas=betas, eps=eps
        )
    elif name == "torch_amsgrad":
        assert isinstance(model, torch.nn.Module)
        # use PyTorch defaults as defaults.
        betas = method_config.get("betas", (0.9, 0.999))
        eps = method_config.get("eps", 1e-8)

        return torch.optim.Adam(
            model.parameters(),
            lr=step_size,
            betas=betas,
            eps=eps,
            amsgrad=True,
        )
    elif name == "torch_adagrad":
        assert isinstance(model, torch.nn.Module)

        eps = method_config.get("eps", 1e-10)

        return torch.optim.Adagrad(model.parameters(), lr=step_size, eps=eps)
    elif name == "torch_lbfgs":
        # use default PyTorch parameters.
        max_iters_per_step = method_config.get("iters_per_step", 20)
        max_eval = method_config.get("max_eval", None)
        tolerance_grad = method_config.get("tolerance_grad", 1e-5)
        tolerance_change = method_config.get("tolerance_change", 1e-9)
        history_size = method_config.get("history_size", 100)
        line_search_fn = method_config.get("line_search_function", None)

        return torch.optim.LBFGS(
            model.parameters(),
            lr=step_size,
            max_iters=max_iters_per_step,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )

    else:
        raise ValueError(
            f"Method {name} not recognized! Please add it to 'methods/index.py'."
        )


def load_proximal_op(
    model: Model,
    d: int,
    prox_config: Optional[Dict] = None,
) -> Optional[ProximalOperator]:
    # synchronize regularization parameters.
    lam: Union[float, lab.Tensor] = 0.0
    M = 1.0
    D = None
    prox = None

    if prox_config is not None:
        assert model.regularizer is not None
        lam = model.regularizer.lam
        if hasattr(model.regularizer, "M"):
            M = model.regularizer.M

        if isinstance(model, ConvexMLP):
            D = model.D

        prox = get_proximal_op(prox_config, lam=lam, d=d, D=D, M=M)

    return prox


def get_callback(config: Dict[str, Any]) -> Callable:
    name = config.get("name", None)
    kwargs = config.get("kwargs", {})
    solver: CVXPYSolver

    if name == "min_l2_decomp":
        solver_name = config.get("solver", "ecos")
        solver = MinL2Decomposition(solver_name, kwargs)
        callback = ConeDecomposition(solver)
    elif name == "min_l1_decomp":
        solver_name = config.get("solver", "ecos")
        solver = MinL1Decomposition(solver_name, kwargs)
        callback = ConeDecomposition(solver)
    elif name == "feasible_decomp":
        solver_name = config.get("solver", "ecos")
        solver = FeasibleDecomposition(solver_name, kwargs)
        callback = ConeDecomposition(solver)
    elif name == "min_relaxed_l2_decomp":
        solver_name = config.get("solver", "ecos")
        solver = MinRelaxedL2Decomposition(solver_name, kwargs)
        callback = ConeDecomposition(solver)
    elif name == "socp_decomp":
        solver_name = config.get("solver", "ecos")
        solver = SOCPDecomposition(solver_name, kwargs)
        callback = ConeDecomposition(solver)
    elif name == "approximate_decomp":
        regularizer_config = config.get("regularizer", None)

        regularizer = None
        if regularizer_config is not None:
            regularizer = get_regularizer(regularizer_config)
            prox = get_proximal_op(regularizer_config, regularizer.lam)
        else:
            prox = get_proximal_op({"name": "identity"})

        callback = ApproximateConeDecomposition(
            regularizer,
            prox,
            config.get("max_iters", 1000),
            config.get("tol", 1e-6),
            combined=config.get("combined", True),
        )

    else:
        raise ValueError(f"Callback with name {name} not recognized.")

    return callback


def get_scheduler(config: Dict[str, Any], optimizer: torch.optim.Optimizer):
    if config is None:
        return None

    name = config.get("name")

    if name == "robbins_monro":
        decay_rate = config.get("rate", 1)

        def decay(epoch):
            return 1 / ((epoch + 1) ** decay_rate)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, decay)

    elif name == "step":
        decay = config.get("decay", 0.1)
        step_length = config.get("step_length", 100)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_length, gamma=decay
        )
