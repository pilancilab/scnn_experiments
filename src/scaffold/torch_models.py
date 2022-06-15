"""
Loaders for PyTorch models.
"""
from typing import Any, Tuple, Dict, Union, Optional, cast
import math

import numpy as np
import torch

import lab

import scnn.private.loss_functions as loss_fns
from scnn.private.models import (
    Model,
    Regularizer,
    L1SquaredRegularizer,
    SequentialWrapper,
    LayerWrapper,
    GatedReLULayer,
)
import scnn.activations as activations


def get_torch_mlp(
    rng: np.random.Generator,
    model_config: Dict[str, Any],
    train_set: Tuple[torch.Tensor, torch.Tensor],
    src_model: Model,
    regularizer: Optional[Regularizer] = None,
) -> Union[torch.nn.Module, Model]:
    """
    Construct a PyTorch multi-layer perceptron (MLP).
    :param rng: a seeded numpy random number generator.
    :param model_config: the configuration object for the MLP.
    :param train_set: the training set as a tuple of tensors.
    :param src_model: the output of a previous experiment for use creating/initializing the current model.
    :returns: model --- the model as a PyTorch module.
    """

    layer_configs = model_config["hidden_layers"]
    name = model_config.get("name", "")

    torch_layers = []
    in_features = train_set[0].shape[1]
    c = train_set[1].shape[1]

    if regularizer is not None:
        lam = regularizer.lam
    else:
        lam = 0

    for config in layer_configs:
        layer, in_features = get_torch_layer(
            rng, config, in_features, train_set, src_model, lam
        )
        torch_layers.append(layer)

    # allow for l1-squared penalty of second layer weights
    if not name.endswith("l1"):
        # output layer.
        torch_layers.append(
            LayerWrapper(
                torch.nn.Linear(
                    in_features=in_features, out_features=c, bias=False
                ),
                None,
            )
        )
    else:
        curr_regularizer = L1SquaredRegularizer(lam)
        # output layer.
        torch_layers.append(
            LayerWrapper(
                torch.nn.Linear(
                    in_features=in_features, out_features=c, bias=False
                ),
                curr_regularizer,
            )
        )

    # construct sequential container for layers
    model = SequentialWrapper(
        *torch_layers, loss_fn=loss_fns.squared_error, regularizer=regularizer
    )

    return model


def get_torch_layer(
    rng: np.random.Generator,
    layer_config: Dict[str, Any],
    in_features: int,
    train_set: Tuple[torch.Tensor, torch.Tensor],
    src_model: Model,
    lam: int = 0,
) -> Tuple[torch.nn.Module, int]:
    """
    Construct PyTorch layers for a neural network.
    :param rng: a seeded numpy random number generator.
    :param layer_config: the configuration object for the layer.
    :param U: a tensor of vectors for a gated ReLU activation function (see GatedReLULayer below).
    :param train_set: the training set as a tuple of tensors.
    :param src_model: the output of a previous experiment for use creating/initializing the current model.
    :returns: (layer, output_dimension) --- the layer as a PyTorch module and its output dimension.
    """

    name = layer_config.get("name", None)
    bias = layer_config.get("bias", False)
    seed = layer_config.get("seed", None)
    layer: torch.nn.Module
    X, y = train_set

    if name is None:
        raise ValueError(
            "A PyTorch layer configuration must have a layer name."
        )
    elif name.startswith("relu"):
        out_features = layer_config["p"]

        # load number of output features from src model.
        if out_features == "m_star":
            assert src_model is not None
            out_features = lab.to_scalar(
                lab.sum(lab.sum(src_model.weights, axis=-1) != 0.0)
            )
            # use a minimum of 50 features.
            out_features = max(out_features, 50)

        linear_layer = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )

        if seed is not None:
            # use a specific initialization.
            torch_rng = torch.get_rng_state()
            torch.manual_seed(seed)
            torch.nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5))
            torch.set_rng_state(torch_rng)

        regularizer = None
        if name.endswith("l1"):
            regularizer = L1SquaredRegularizer(lam)

        layer = LayerWrapper(
            torch.nn.Sequential(
                linear_layer,
                torch.nn.ReLU(),
            ),
            regularizer,
        )

    elif name.startswith("linear"):
        out_features = layer_config["p"]

        regularizer = None
        if name.endswith("l1"):
            regularizer = L1SquaredRegularizer(lam)

        layer = LayerWrapper(
            torch.nn.Linear(
                in_features=in_features, out_features=out_features, bias=bias
            ),
            regularizer,
        )

    elif name.startswith("gated_relu"):
        U = activations.sample_gate_vectors(
            layer_config.get("seed", 123),
            in_features,
            layer_config.get("n_samples", 100),
            layer_config.get("gate_type", "dense"),
            layer_config.get("order", 1),
        )

        D, U = activations.compute_activation_patterns(lab.to_np(X), U)
        D, U = lab.tensor(D, dtype=lab.get_dtype()), lab.tensor(
            U, dtype=lab.get_dtype()
        )

        U = cast(torch.Tensor, U)
        regularizer = None
        if name.endswith("l1"):
            regularizer = L1SquaredRegularizer(lam)

        layer = LayerWrapper(GatedReLULayer(U), regularizer)
        out_features = layer.layer.out_features

    else:
        raise ValueError(
            f"PyTorch layer {name} not recognized. Please register it in 'models.pytorch'."
        )

    return layer, out_features
