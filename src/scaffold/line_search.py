"""
Scaffolding for loading line-search components.
"""

from convex_nn.private.methods.line_search import (
    Backtracker,
    MultiplicativeBacktracker,
    LSCondition,
    FSS,
    QuadraticBound,
    DiagQB,
    Armijo,
    StepSizeUpdater,
    KeepNew,
    KeepOld,
    ForwardTrack,
    Lassplore,
)

from typing import Dict, Any


def get_ls_condition(config: Dict[str, Any]) -> LSCondition:
    """Load a line-search condition by name using the passed configuration parameters.
    :param config: configuration object specifying the line-search condition to load.
    :returns: a LSCondition instance that can be used to evaluate the line-search condition.
    """
    cond: LSCondition
    name = config.get("name", None)
    if name is None:
        raise ValueError(
            "The line-search condition configuration must include a condition name!"
        )
    elif name == "quadratic_bound":
        cond = QuadraticBound()
    elif name == "diag_qb":
        raise NotImplementedError("Not implemented yet!")
    elif name == "fss":
        cond = FSS()
    elif name == "armijo":
        cond = Armijo(rho=config["rho"])
    else:
        raise ValueError(
            f"Line-search condition {name} not recognized. Please add to 'methods/line_search/conditions.py'."
        )

    return cond


def get_backtrack_fn(config: Dict[str, Any]) -> Backtracker:

    name = config.get("name", None)

    if name is None:
        raise ValueError(
            "The backtrack function configuration must include a condition name!"
        )
    elif name == "backtrack":
        backtrack_fn = MultiplicativeBacktracker(beta=config["beta"])
    else:
        raise ValueError(
            f"Backtrack function {name} not recognized. Please add to 'methods/line_search/backtrack.py'."
        )

    return backtrack_fn


def get_update_fn(config: Dict[str, Any]) -> StepSizeUpdater:
    """Load a method for updating the step-size between iterations by name using the passed
    configuration parameters.
    :param config: configuration object specifying the line-search condition to load.
    :returns: a StepSizeUpdater instance that can be used to update the step-size.
    """

    name = config.get("name", None)

    if name is None:
        raise ValueError(
            "The line-search condition configuration must include a condition name!"
        )
    elif name == "forward_track":
        update_fn = ForwardTrack(alpha=config["alpha"])
    elif name == "lassplore":
        update_fn = Lassplore(
            alpha=config["alpha"], threshold=config["threshold"]
        )
    elif name == "keep_new":
        update_fn = KeepNew()
    elif name == "keep_old":
        update_fn = KeepOld()
    else:
        raise ValueError(
            f"Step-size update {name} not recognized. Please add to 'methods/line_search/step_size_updates.py'."
        )

    return update_fn
