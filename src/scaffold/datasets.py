"""
Datasets and related utilities.
"""
import os
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
from sklearn.model_selection import KFold  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore

import torch
from torch.utils import data
from torchvision import datasets, transforms  # type: ignore

import lab
from scnn.private.utils.data import (
    gen_regression_data,
    gen_classification_data,
    gen_sparse_regression_problem,
    unitize_columns,
    train_test_split,
    add_bias_col,
)

from .uci_names import (
    UCI_DATASETS,
    SMALL_UCI_DATASETS,
    BINARY_SMALL_UCI_DATASETS,
)


# types

Dataset = Tuple[lab.Tensor, lab.Tensor]

# constants

PYTORCH_DATASETS = ["mnist", "cifar_10", "cifar_100"]


# loaders


def load_dataset(dataset_config: Dict[str, Any], data_dir: str = "data"):
    """Load a dataset by name using the passed configuration parameters.
    :param rng: a random number generator.
    :param dataset_config: configuration object specifying the dataset to load.
    :param data_dir: the base directory to look for datasets.
    :returns: (X, y), where X is the feature matrix, y are the target
    """
    train_data: Dataset
    test_data: Dataset

    name = dataset_config.get("name", None)
    valid_prop = dataset_config.get("valid_prop", 0.2)
    test_prop = dataset_config.get("test_prop", 0.2)
    use_valid = dataset_config.get("use_valid", True)
    split_seed = dataset_config.get("split_seed", 1995)
    n_folds = dataset_config.get("n_folds", None)
    fold_index = dataset_config.get("fold_index", None)
    unitize_data_cols = dataset_config.get("unitize_data_cols", True)
    add_bias = dataset_config.get("add_bias", False)

    if name is None:
        raise ValueError("Dataset configuration must have a name parameter!")
    elif name in [
        "synthetic_regression",
        "synthetic_classification",
        "sparse_regression",
    ]:
        data_seed, n, n_test, d, c, kappa = (
            dataset_config.get("data_seed", 951),
            dataset_config["n"],
            dataset_config["n_test"],
            dataset_config["d"],
            dataset_config.get("c", 1),
            dataset_config.get("kappa", 1.0),
        )

        if name == "synthetic_regression":
            sigma = dataset_config.get("sigma", 0.0)
            train_data, test_data, _ = gen_regression_data(
                data_seed,
                n,
                n_test,
                d,
                c,
                sigma,
                kappa,
            )
        elif name == "synthetic_classification":
            hidden_units = dataset_config.get("hidden_units", 50)
            train_data, test_data = gen_classification_data(
                data_seed,
                n,
                n_test,
                d,
                hidden_units,
                kappa,
            )
        elif name == "sparse_regression":
            sigma = dataset_config.get("sigma", 0.0)
            train_data, test_data, _ = gen_sparse_regression_problem(
                data_seed,
                n,
                n_test,
                d,
                sigma,
                kappa,
                dataset_config.get("num_zeros", 0),
                dataset_config.get("transform", None),
            )
        train_data = (train_data[0], train_data[1].reshape(-1, 1))
        test_data = (test_data[0], test_data[1].reshape(-1, 1))

        train_data = lab.all_to_tensor(train_data, dtype=lab.get_dtype())
        test_data = lab.all_to_tensor(test_data, dtype=lab.get_dtype())

    elif name in PYTORCH_DATASETS:
        pytorch_src = os.path.join(data_dir, "pytorch")
        transform = load_transforms(
            dataset_config.get("transforms", None), name
        )
        train_data = load_pytorch_dataset(
            name,
            pytorch_src,
            train=True,
            transform=transform,
            valid_prop=valid_prop,
            use_valid=use_valid,
            split_seed=split_seed,
            n_folds=n_folds,
            fold_index=fold_index,
        )

        test_data = load_pytorch_dataset(
            name,
            pytorch_src,
            train=False,
            transform=transform,
            valid_prop=valid_prop,
            use_valid=use_valid,
            split_seed=split_seed,
            n_folds=n_folds,
            fold_index=fold_index,
        )

    elif name in UCI_DATASETS:
        uci_src = os.path.join(data_dir, "uci", "datasets")
        train_data, test_data = load_uci_dataset(
            name,
            uci_src,
            test_prop,
            valid_prop,
            use_valid,
            split_seed,
            n_folds,
            fold_index,
        )

    else:
        raise ValueError(
            f"Dataset with name {name} not recognized! Please configure it first."
        )

    if add_bias:
        train_data = add_bias_col(train_data)
        test_data = add_bias_col(test_data)

    if unitize_data_cols:
        train_data, test_data, _ = unitize_columns(train_data, test_data)

    return train_data, test_data


def load_uci_dataset(
    name: str,
    src: str = "data/uci/datasets",
    test_prop: float = 0.2,
    valid_prop: float = 0.2,
    use_valid: bool = True,
    split_seed: int = 1995,
    n_folds: Optional[int] = None,
    fold_index: Optional[int] = None,
    unitize_data_cols: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Load one of the UCI datasets by name.
    :param name: the name of the dataset to load.
    :param src: base path for pytorch datasets..
    :param train: whether to load the training set (True) or test set (False)
    :param transform: torchvision transform for processing the image features.
    :param use_valid: whether or not to use a train/validation split of the training set.
    :param split_seed: the seed to use when constructing the train/validation split.
    :param n_folds: the number of cross-validation folds to use. Defaults to 'None',
        ie. no cross validation is performed.
    :param fold_index: the particular cross validation split to load. This must be provided
        when 'n_folds' is not None.
    :returns: training and test sets.
    """

    data_dict = {}
    for k, v in map(
        lambda x: x.split(),
        open(os.path.join(src, name, name + ".txt"), "r").readlines(),
    ):
        data_dict[k] = v

    # load data
    f = open(os.path.join(src, name, data_dict["fich1="]), "r").readlines()[1:]
    full_X = np.asarray(
        list(map(lambda x: list(map(float, x.split()[1:-1])), f))
    )
    full_y = np.asarray(list(map(lambda x: int(x.split()[-1]), f))).squeeze()

    classes = np.unique(full_y)
    if len(classes) == 2:
        full_y[full_y == classes[0]] = -1
        full_y[full_y == classes[1]] = 1
        full_y = np.expand_dims(full_y, 1)
    else:
        # use one-hot encoding for multi-class problems.
        full_y = np.expand_dims(full_y, 1)
        encoder = OneHotEncoder()
        encoder.fit(full_y)
        full_y = encoder.transform(full_y).toarray()

    # for vector-outputs

    # split dataset
    train_set, test_set = train_test_split(
        full_X, full_y, test_prop, split_seed=split_seed
    )
    if n_folds is not None:
        assert fold_index is not None

        train_set, test_set = cv_split(
            train_set[0],
            train_set[1],
            fold_index,
            n_folds,
            split_seed=split_seed,
        )

    elif use_valid:
        train_set, test_set = train_test_split(
            train_set[0], train_set[1], valid_prop, split_seed=split_seed
        )

    return lab.all_to_tensor(
        train_set, dtype=lab.get_dtype()
    ), lab.all_to_tensor(test_set, dtype=lab.get_dtype())


def load_pytorch_dataset(
    name: str,
    src: str = "data/pytorch",
    train: bool = True,
    transform: Optional[Any] = None,
    valid_prop: float = 0.2,
    use_valid: bool = True,
    split_seed: int = 1995,
    n_folds: Optional[int] = None,
    fold_index: Optional[int] = None,
) -> Any:
    """Load TorchVision dataset.
    :param name: the name of the dataset to load.
    :param src: base path for pytorch datasets..
    :param train: whether to load the training set (True) or test set (False)
    :param transform: torchvision transform for processing the image features.
    :param use_valid: whether or not to use a train/validation split of the training set.
    :param split_seed: the seed to use when constructing the train/validation split.
    :param n_folds: the number of cross-validation folds to use. Defaults to 'None',
        ie. no cross validation is performed.
    :param fold_index: the particular cross validation split to load. This must be provided
        when 'n_folds' is not None.
    :returns: torch.utils.data.Dataset.
    """

    if name == "cifar_100":
        cls = datasets.CIFAR100
        num_classes = 100
    elif name == "cifar_10":
        cls = datasets.CIFAR10
        num_classes = 10
    elif name == "mnist":
        cls = datasets.MNIST
        num_classes = 10
    else:
        raise ValueError(
            f"PyTorch dataset with name '{name}' not recognized! Please register it in 'datasets.py'."
        )

    fetch_train = train or use_valid
    # avoid annoying download message by first trying to load the dataset without downloading.
    try:
        dataset = cls(
            root=src,
            transform=transform,
            download=False,
            train=fetch_train,
        )
    except Exception:
        dataset = cls(
            root=src,
            transform=transform,
            download=True,
            train=fetch_train,
        )

    X = []
    y = []
    # iterate through dataset to obtain transformed NumPy arrays
    # one-hot encoding for multi-class labels
    for X_batch, y_batch in data.DataLoader(dataset, batch_size=256):
        X.append(X_batch.numpy())
        y.append(torch.nn.functional.one_hot(y_batch, num_classes).numpy())

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    if n_folds is not None:
        assert fold_index is not None

        train_set, valid_set = cv_split(
            X, y, fold_index, n_folds, split_seed=split_seed
        )

        if train:
            return lab.all_to_tensor(train_set, dtype=lab.get_dtype())
        else:
            return lab.all_to_tensor(valid_set, dtype=lab.get_dtype())

    elif use_valid:

        train_set, valid_set = train_test_split(X, y, valid_prop, split_seed)

        if train:
            return lab.all_to_tensor(train_set, dtype=lab.get_dtype())
        else:
            return lab.all_to_tensor(valid_set, dtype=lab.get_dtype())

    else:
        return lab.all_to_tensor((X, y), dtype=lab.get_dtype())


def cv_split(X, y, fold_index, n_folds=5, split_seed=1995):
    """ """
    kf = KFold(n_folds, shuffle=True, random_state=split_seed)

    train_indices, valid_indices = list(kf.split(X))[fold_index]

    return (X[train_indices], y[train_indices]), (
        X[valid_indices],
        y[valid_indices],
    )


def load_transforms(
    transform_names: Optional[List[str]], dataset_name: str
) -> Any:
    """Load transformations for a PyTorch dataset.
    :param transform_list: a list of transformations to apply to the dataset.
    :param dataset_name: name of the pytorch dataset (used for normalization)
        Order *matters* as it determines the order in which the transforms are applied.
    :returns: transform
    """

    # no transformations to load.
    if transform_names is None or len(transform_names) == 0:
        return None

    transform_list = []
    for name in transform_names:
        if name == "to_tensor":
            transform_list.append(transforms.ToTensor())
        elif name == "flatten":
            transform_list.append(
                transforms.Lambda(lambd=lambda x: x.reshape(-1))
            )
        elif name == "normalize":
            if dataset_name == "mnist":
                transform_list.append(
                    transforms.Normalize((0.1307,), (0.3081,))
                )
            elif dataset_name == "cifar_10":
                transform_list.append(
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    )
                )
            elif dataset_name == "cifar_100":
                transform_list.append(
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    )
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} not recognized for normalization! Please register it in 'datasets.py'"
                )

        else:
            raise ValueError(
                f"Transform {name} not recognized! Please register it 'datasets.py'"
            )

    return transforms.Compose(transform_list)
