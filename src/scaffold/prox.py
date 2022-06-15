"""
Loaders for proximal operators.
"""

from typing import Optional, Dict, Any

import lab

from scnn.private.prox import (
    ProximalOperator,
    Identity,
    L1,
    L2,
    GroupL1,
    FeatureGroupL1,
    Orthant,
    GroupL1Orthant,
)


def get_proximal_op(
    prox_config: Dict[str, Any],
    lam: float = 0.0,
    D: Optional[lab.Tensor] = None,
    d: Optional[int] = None,
    M: float = 1.0,
) -> ProximalOperator:
    """Lookup and initialize proximal operator using a configuration dictionary.
    :param prox_config: the configuration options for the proximal operator.
    :param lam: (optional) the strength of the regularization.
    :param D: (optional) the activation matrix for the convex neural network.
    :param M: (optional) constraint "strength".
    :returns: a function which evaluates the proximal operator.
    """

    name = prox_config.get("name", None)

    if name is None:
        raise ValueError(
            "The proximal operator configuration must include a method name!"
        )
    elif name == "identity":
        return Identity()
    elif name == "l2":
        return L2(lam=lam)
    elif name == "l1":
        return L1(lam=lam)
    elif name == "group_l1":
        return GroupL1(lam=lam)
    elif name == "feature_gl1":
        return FeatureGroupL1(lam=lam)
    elif name == "orthant":
        assert D is not None and d is not None
        A = 2 * D - lab.ones_like(D)
        return Orthant(A)
    elif name == "group_l1_orthant":
        assert D is not None and d is not None
        A = 2 * D - lab.ones_like(D)
        return GroupL1Orthant(d, lam=lam, A=A)

    else:
        raise ValueError(
            f"Proximal operator with name {name} not recognized! Please register it in 'proximal_ops.py'."
        )
