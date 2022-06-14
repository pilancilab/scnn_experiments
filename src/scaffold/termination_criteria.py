"""
Scaffolding for loading termination criteria.
"""
from typing import Dict, Any

from scnn.private.methods.termination_criteria import (
    TerminationCriterion,
    GradientNorm,
    StepLength,
    ConstrainedHeuristic,
    LagrangianGradNorm,
)

TOL = 1e-6


def get_criterion(config: Dict[str, Any]) -> TerminationCriterion:
    """Load a termination criterion by name using the passed configuration parameters.
    :param config: configuration object specifying the termination criterion.
    :returns: an instance of TerminationCriterion which can be used to determine if an optimizer as converged.
    """
    name = config.get("name", None)

    if name is None:
        raise ValueError("Termination criterion must have name!")
    elif name == "grad_norm":
        return GradientNorm(config.get("tol", TOL))
    elif name == "step_length":
        return StepLength(config.get("tol", TOL))
    elif name == "constrained_heuristic":
        return ConstrainedHeuristic(
            config.get("grad_tol", TOL), config.get("constraint_tol", TOL)
        )
    elif name == "lagrangian_grad_norm":
        return LagrangianGradNorm(
            config.get("grad_tol", TOL), config.get("constraint_tol", TOL)
        )
    else:
        raise ValueError(f"Termination criterion {name} not recognized!")
