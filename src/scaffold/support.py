"""
Helpers for finding feature support.
"""
from logging import Logger
from typing import Dict, Any, Tuple, Union, Optional, cast, Callable, List

import lab

from scnn.support import LinearSupportFinder, ForwardBackward
from scnn.solvers import RFISTA, AL
from scnn.models import ConvexReLU, ConvexGatedReLU
from scnn.regularizers import NeuronGL1
from scnn.activations import sample_gate_vectors

from scnn.private.models import Model, ConvexMLP, AL_MLP


def get_support(
    logger: Logger,
    model_config: Dict[str, Any],
    train_set,
    support_config: Dict[str, Any],
    device: str,
    dtype: str,
    seed: int,
) -> lab.Tensor:

    name = support_config.get("name", None)
    X, y = train_set
    _, d = X.shape
    c = y.shape[1]

    if name is None:
        # return full support
        support = lab.arange(d).tolist()

    elif name == "linear":
        # find the support by fitting sparse linear models
        lambda_path = support_config["lambda_path"]
        valid_prop = support_config.get("valid_prop", 0.2)
        support_finder = LinearSupportFinder(lambda_path, valid_prop)

        support = support_finder(
            None, None, None, train_set, device, dtype, seed
        )

    elif name == "forward" or name == "backward":
        model_name = model_config["name"]
        if model_name not in ["convex_mlp", "al_mlp"]:
            raise ValueError(
                "Forward-backward only supports convex_mlp or al_mlp."
            )

        valid_prop = support_config.get("valid_prop", 0.2)
        support_finder = ForwardBackward(name == "forward", valid_prop)

        sign_patterns = model_config["sign_patterns"]
        G = sample_gate_vectors(
            sign_patterns.get("seed", 123),
            d,
            sign_patterns.get("n_samples", 100),
        )
        if model_name == "al_mlp":
            convex_model = ConvexReLU(G, c)
            solver = AL(convex_model)
        else:
            convex_model = ConvexGatedReLU(G, c)
            solver = RFISTA(convex_model)

        if "regularizer" in model_config:
            # only support group L1 at the moment.
            assert model_config["regularizer"]["name"] == "group_l1"

            regularizer = NeuronGL1(model_config["regularizer"]["lambda"])

        support = support_finder(
            convex_model, regularizer, solver, train_set, device, dtype, seed
        )

    else:
        raise ValueError(f"Support method {name} not registered.")

    return support
