"""
Functions for running experiments.
"""
from typing import Dict

from logging import Logger

import lab
from convex_nn.private.methods.callbacks import ObservedSignPatterns
from convex_nn.private.initializers import get_initializer

from scaffold.models import get_model
from scaffold.methods import get_method
from scaffold.datasets import load_dataset
from scaffold.support import get_support


def run_experiment(
    logger: Logger, exp_dict: Dict, data_dir: str = "data", src_model=None
):
    seed = exp_dict["seed"] + exp_dict["repeat"]

    # linear algebra backend
    lab.set_backend(exp_dict.get("backend", lab.NUMPY))
    device = exp_dict.get("device", lab.CPU)
    dtype = exp_dict.get("dtype", lab.FLOAT32)
    lab.set_device(device)
    lab.set_dtype(dtype)
    lab.set_seeds(seed)

    rng = lab.np_rng

    logger.info(f"Loading dataset {exp_dict['data']['name']}.")
    train_set, test_set = load_dataset(exp_dict["data"], data_dir)

    if "support" in exp_dict:
        support = get_support(
            logger,
            exp_dict["model"],
            train_set,
            exp_dict["support"],
            device,
            dtype,
            seed,
        )

        # drop extra features from dataset
        train_set = train_set[0][:, support], train_set[1]
        test_set = test_set[0][:, support], test_set[1]

    logger.info(f"Constructing model {exp_dict['model']['name']}.")
    model = get_model(logger, rng, train_set, exp_dict["model"], src_model)
    logger.info(
        f"Initializing model at: {exp_dict['model'].get('initializer', {'name': 'zero'})['name']}."
    )
    initializer = get_initializer(
        logger, rng, train_set, exp_dict["model"].get("initializer", {})
    )

    logger.info(f"Preparing optimizer {exp_dict['method']['name']}.")
    opt_procedure = get_method(
        logger, rng, model, train_set, exp_dict["method"]
    )
    logger.info("Optimizing model.")
    exit_status, model, metrics = opt_procedure(
        logger,
        model,
        initializer,
        train_set,
        test_set,
        exp_dict["metrics"],
        exp_dict.get("final_metrics", None),
    )

    return exit_status, model, metrics


def run_reg_path_experiment(
    logger: Logger, exp_dict: Dict, data_dir: str = "data", src_model=None
):
    seed = exp_dict["seed"] + exp_dict["repeat"]

    # linear algebra backend
    lab.set_backend(exp_dict.get("backend", lab.NUMPY))
    lab.set_device(exp_dict.get("device", lab.CPU))
    lab.set_dtype(exp_dict.get("dtype", lab.FLOAT32))
    lab.set_seeds(seed)

    rng = lab.np_rng

    logger.info(f"Loading dataset {exp_dict['data']['name']}.")
    train_set, test_set = load_dataset(exp_dict["data"], data_dir)

    logger.info(f"Constructing model {exp_dict['model']['name']}.")
    model = get_model(logger, rng, train_set, exp_dict["model"], src_model)
    logger.info(
        f"Initializing model at: {exp_dict['model'].get('initializer', {'name': 'zero'})['name']}."
    )
    initializer = get_initializer(
        logger, rng, train_set, exp_dict["model"].get("initializer", {})
    )

    logger.info(f"Preparing optimizer {exp_dict['method']['name']}.")
    opt_procedure = get_method(
        logger, rng, model, train_set, exp_dict["method"]
    )

    regularization_path = exp_dict["regularization_path"]
    path_metrics = {}

    for i, lam in enumerate(regularization_path):
        opt_procedure.reset()
        model.regularizer.lam = lam
        if hasattr(opt_procedure, "optimizer") and hasattr(
            opt_procedure.optimizer, "prox"
        ):
            opt_procedure.optimizer.prox.lam = lam
        elif hasattr(opt_procedure, "inner_optimizer") and hasattr(
            opt_procedure.inner_optimizer, "prox"
        ):
            opt_procedure.inner_optimizer.prox.lam = lam
            # warm-start dual parameters only.
            model.weights = lab.zeros_like(model.weights)

        if i > 0:
            initializer = lambda new_model: model

        logger.info(
            f"Optimizing model {i+1}/{len(regularization_path)} using regularization parameter {lam}."
        )
        exit_status, model, metrics = opt_procedure(
            logger,
            model,
            initializer,
            train_set,
            test_set,
            exp_dict["metrics"],
            exp_dict.get("final_metrics", None),
        )
        path_metrics[lam] = metrics

    return exit_status, model, path_metrics


def run_active_set_experiment(
    logger: Logger, exp_dict: Dict, data_dir: str = "data", src_model=None
):
    seed = exp_dict["seed"] + exp_dict["repeat"]

    # linear algebra backend
    lab.set_backend(exp_dict.get("backend", lab.NUMPY))
    lab.set_device(exp_dict.get("device", lab.CPU))
    lab.set_dtype(exp_dict.get("dtype", lab.FLOAT32))
    lab.set_seeds(seed)

    rng = lab.np_rng

    logger.info(f"Loading dataset {exp_dict['data']['name']}.")
    train_set, test_set, _ = load_dataset(exp_dict["data"], data_dir)

    # ===  Active Set === #
    logger.info(f"Preparing active set method.")
    active_set_conf = exp_dict["active_set"]

    nc_model = get_model(
        logger, rng, train_set, active_set_conf["model"], src_model
    )
    nc_initializer = get_initializer(
        logger, rng, train_set, active_set_conf["model"].get("initializer", {})
    )
    nc_opt_procedure = get_method(
        logger, rng, nc_model, train_set, active_set_conf["method"]
    )
    logger.info("Optimizing non-convex model for active set discovery.")
    exit_status, nc_model, metrics = nc_opt_procedure(
        logger,
        nc_model,
        nc_initializer,
        train_set,
        test_set,
        (["objective", "grad_norm"], [], []),
        final_metrics=None,
        callback=ObservedSignPatterns(),
    )
    D, W = nc_model.activation_history, nc_model.weight_history

    logger.info(f"Constructing model {exp_dict['model']['name']}.")
    model = get_model(logger, rng, train_set, exp_dict["model"], src_model)

    max_new_patterns = active_set_conf.get("max_new_patterns", None)
    if max_new_patterns is not None and D.shape[1] > max_new_patterns:
        # take the most recent patterns
        D, W = D[:, -max_new_patterns:], W[-max_new_patterns:]

    model.add_new_patterns(D, W, remove_zero=True)

    logger.info(
        f"Initializing model at: {exp_dict['model'].get('initializer', {'name': 'zero'})['name']}."
    )
    initializer = get_initializer(
        logger, rng, train_set, exp_dict["model"].get("initializer", {})
    )

    logger.info(f"Preparing optimizer {exp_dict['method']['name']}.")
    opt_procedure = get_method(
        logger, rng, model, train_set, exp_dict["method"]
    )
    logger.info("Optimizing model.")
    exit_status, model, metrics = opt_procedure(
        logger,
        model,
        initializer,
        train_set,
        test_set,
        exp_dict["metrics"],
        exp_dict.get("final_metrics", None),
    )

    return exit_status, model, metrics
