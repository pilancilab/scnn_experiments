"""
Loaders for models.
"""

from typing import Dict, Any, Tuple, Union, cast, Optional
from logging import Logger

import numpy as np
import torch

import lab

import convex_nn.activations as activations
from convex_nn.private.models.convex import operators

from convex_nn.private.models import (
    Model,
    LinearRegression,
    LogisticRegression,
    ReLUMLP,
    GatedReLUMLP,
    ConvexMLP,
    AL_MLP,
    Regularizer,
    GroupL1Regularizer,
    FeatureGroupL1Regularizer,
    GroupL1Orthant,
    L2Regularizer,
    L1Regularizer,
    L1SquaredRegularizer,
    OrthantConstraint,
)


from .torch_models import get_torch_mlp


REQURIES_GATES = [
    "convex_mlp",
    "al_mlp",
]


def get_model(
    logger: Logger,
    rng: np.random.Generator,
    train_set: Tuple[lab.Tensor, lab.Tensor],
    model_config: Dict[str, Any],
    src_model: Optional[Model] = None,
) -> Union[Model, torch.nn.Module]:
    """Construct and return the model specified by the supplied dictionary.
    :param logger: a logger instance.
    :param rng: a random number generator.
    :param train_set: the training dataset.
    :param model_config: a dictionary object specifying the desired
        model and its arguments.
    :param src_model: the output of a previous experiment for use creating/initializing the current model.
    :returns: a Model instance.
    """

    name = model_config.get("name", None)
    model: Union[torch.nn.Module, Model]

    # load regularizer if specified.

    regularizer, D, U, c = None, None, None, 1
    X, y = train_set
    n, d = X.shape
    c = y.shape[1]

    use_bias: bool = lab.to_np(lab.all(X[:, -1] == X[0, -1]))

    if name in REQURIES_GATES:
        gate_config = model_config.get("sign_patterns", {})

        U = activations.sample_gate_vectors(
            gate_config.get("seed", 123),
            d - use_bias,
            gate_config.get("n_samples", 100),
            gate_config.get("gate_type", "dense"),
            gate_config.get("order", 1),
        )

        active_prop = gate_config.get("active_proportion", None)
        if active_prop == "min_class_prop":
            if c == 1:
                active_prop = lab.sum(y.ravel() == 1) / n
                active_prop = (
                    1 - active_prop if active_prop > 0.5 else active_prop
                )
            else:
                active_prop = lab.min(lab.sum(y, axis=0) / n)
                active_prop = lab.to_scalar(lab.smax(active_prop, 0.05))

        D, U = activations.compute_activation_patterns(
            lab.to_np(X),
            U,
            bias=use_bias,
            active_proportion=active_prop,
        )

        D, U = lab.tensor(D, dtype=lab.get_dtype()), lab.tensor(
            U, dtype=lab.get_dtype()
        )

    if "regularizer" in model_config:
        regularizer = get_regularizer(model_config["regularizer"], D)

    # target shape
    c = train_set[1].shape[1]

    if name is None:
        raise ValueError(
            "The model configuration must contain a name for the model to create!"
        )
    elif name in REQURIES_GATES:
        assert D is not None
        assert U is not None

        kernel = model_config.get("kernel", operators.EINSUM)
        assert kernel in operators.KERNELS

        if name == "convex_mlp":
            model = ConvexMLP(d, D, U, kernel, regularizer=regularizer, c=c)

        elif name == "al_mlp":
            delta = model_config.get("delta", 1000.0)

            model = AL_MLP(
                d,
                D,
                U,
                kernel,
                delta,
                regularizer=regularizer,
                c=c,
            )

    elif name.startswith("torch_mlp"):
        model = get_torch_mlp(
            rng, model_config, train_set, src_model, regularizer
        )
        # move model to correct device
        model.to(lab.get_device())

    elif name == "relu_mlp":
        model = ReLUMLP(d, model_config["p"], regularizer, c=c)
    elif name == "gated_relu_mlp":
        assert U is not None
        model = GatedReLUMLP(d, U, regularizer, c=c)
    elif name == "linear_regression":
        model = LinearRegression(d, c=c, regularizer=regularizer)
    else:
        raise ValueError(
            f"Model name {name} not recognized! Please add it to the model \
            index in 'models/index.py'!"
        )

    return model


def get_regularizer(
    config: Dict[str, Any], D: Optional[lab.Tensor] = None
) -> Regularizer:
    """Construct and return the regularizer specified by the supplied dictionary.
    :param config: the dictionary providing configuration parameters for the regularizer.
    :param D: sign patterns.
    :returns: the regularizer.
    """

    name = config.get("name", None)

    if name is None:
        raise ValueError(
            "The model configuration must contain a name for the model to create!"
        )
    elif name == "l2":
        lam = config.get("lambda", 0.0)

        return L2Regularizer(lam)
    elif name == "l1":
        lam = config.get("lambda", 0.0)

        return L1Regularizer(lam)
    elif name == "group_l1":
        lam = config.get("lambda", 0.0)

        return GroupL1Regularizer(lam)

    elif name == "feature_gl1":
        lam = config.get("lambda", 0.0)

        return FeatureGroupL1Regularizer(lam)

    elif name == "orthant":
        assert D is not None
        A = 2 * D - lab.ones_like(D)
        return OrthantConstraint(A)

    elif name == "group_l1_orthant":
        assert D is not None
        lam = config.get("lambda", 0.0)
        A = 2 * D - lab.ones_like(D)

        return GroupL1Orthant(A, lam)

    elif name == "l1_squared":
        lam = config.get("lambda", 0.0)

        return L1SquaredRegularizer(lam)

    else:
        raise ValueError(
            f"Regularizer name {name} not recognized! Please add it to the regularizer index in 'models/index.py'!"
        )
