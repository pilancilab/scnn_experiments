from itertools import product
from functools import reduce, partial
from collections import defaultdict
import pickle as pkl
import os

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
import numpy as np

from scaffold import datasets
from scaffold.uci_names import BINARY_SMALL_UCI_DATASETS
from experiment_utils import utils, files
from exp_configs import EXPERIMENTS  # type: ignore

# load neural network results
metric_grid = files.load_and_clean_experiments(
    EXPERIMENTS["table_2_final"] + EXPERIMENTS["table_2_final_deep"],
    ["results/table_2_final", "results/table_2_final_deep"],
    ["test_nc_accuracy"],
    ("data", "name"),
    ("method", "name"),
    ("model", "sign_patterns", "seed"),
    partial(utils.quantile_metrics, quantiles=(0.0, 1.0)),
    keep=[],
    remove=[],
    filter_fn=None,
    processing_fns=[],
    x_key=None,
    x_vals=None,
)

flipped_grid = defaultdict(lambda: defaultdict(dict))

for dataset in metric_grid.keys():
    for metric in metric_grid[dataset].keys():
        for line in metric_grid[dataset][metric].keys():
            flipped_grid[dataset][line][metric] = metric_grid[dataset][metric][
                line
            ]


def accuracy(y_hat, y):
    return np.sum(y_hat == y) / len(y)


def find_best(acc, new):
    if new[1] >= acc[1]:
        return new
    else:
        return acc


def load_data(data_config):
    train, valid = datasets.load_dataset(data_config, "data")

    if train[1].shape[1] == 1:
        train = train[0], train[1].ravel()
        valid = valid[0], valid[1].ravel()
    else:
        train = train[0], np.argmax(train[1], axis=1)
        valid = valid[0], np.argmax(valid[1], axis=1)

    return train, valid


seed = 650
verbose = False


def cv_experiment(classifier, hyperparameters, dataset_name):
    uci_data = {
        "name": dataset_name,
        "split_seed": 1995,
        "n_folds": 5,
        "fold_index": 0,
    }

    results = []
    for params in hyperparameters:
        print(f"Trying {params}...")
        validation_accuracies = []
        for fold in range(5):
            data_config = uci_data.copy()
            data_config["fold_index"] = fold

            train, valid = load_data(data_config)

            clf = classifier(random_state=seed, **params)
            clf.fit(train[0], train[1])

            validation_accuracy = accuracy(clf.predict(valid[0]), valid[1])

            if verbose:
                print(params)
                print(
                    "Train Accuracy:",
                    accuracy(clf.predict(train[0]), train[1]),
                )
                print("Valid Accuracy:", validation_accuracy)

            validation_accuracies.append(validation_accuracy)
        results.append((params, np.mean(validation_accuracies)))

    best_params = dict(reduce(find_best, results)[0])

    # compute test metrics
    test_config = {
        "name": dataset_name,
        "split_seed": 1995,
        "use_valid": False,
    }
    train, test = load_data(test_config)

    clf = classifier(random_state=seed, **best_params)
    clf.fit(train[0], train[1])

    train_acc = accuracy(clf.predict(train[0]), train[1])
    test_acc = accuracy(clf.predict(test[0]), test[1])

    return best_params, train_acc, test_acc


all_results = {}

# random forests
depths = [2, 4, 10, 25, 50]
sizes = [5, 10, 100, 1000]
rf_params = [
    {"n_estimators": size, "max_depth": depth}
    for size, depth in product(sizes, depths)
]

# linear SVMs
lambda_to_try = np.logspace(-5, 0, 6).tolist()
linear_svm_params = [
    {"kernel": "linear", "C": 1.0 / lam} for lam in lambda_to_try
]

# kernel SVMs
lambda_to_try = np.logspace(-5, 0, 5).tolist()
bandwidths = np.logspace(-4, 2, 6).tolist()
kernel_svm_params = [
    {"kernel": "rbf", "C": 1.0 / lam, "gamma": gamma}
    for lam, gamma in product(lambda_to_try, bandwidths)
]

dataset_list = BINARY_SMALL_UCI_DATASETS

for dataset_name in dataset_list:
    # for dataset_name in ["car"]:
    print(f"Running dataset {dataset_name}.")

    if os.path.exists(f"results/binary_baselines/{dataset_name}.pkl"):
        print("Loading results...")

        with open(f"results/binary_baselines/{dataset_name}.pkl", "rb") as f:
            res = pkl.load(f)
    else:
        print("Running RF")
        baseline_res = cv_experiment(
            RandomForestClassifier, rf_params, dataset_name
        )
        print("Running L-SVM")
        linear_svm_res = cv_experiment(SVC, linear_svm_params, dataset_name)
        print("Running K-SVM")
        kernel_svm_res = cv_experiment(SVC, kernel_svm_params, dataset_name)

        res = (baseline_res, linear_svm_res, kernel_svm_res)

        # save result
        with open(f"results/binary_baselines/{dataset_name}.pkl", "wb") as f:
            pkl.dump(res, f)

    all_results[dataset_name] = res

result_str = (
    "Dataset, C-GReLU, C-ReLU, Random Forest, Linear SVM, Kernel SVM \n"
)

result_keys = list(all_results.keys())
result_keys.sort()

for key in result_keys:
    value = all_results[key]
    result_str = result_str + key

    grelu = flipped_grid[key]["fista"]["test_nc_accuracy"]["upper"][-1]
    relu = flipped_grid[key]["augmented_lagrangian"]["test_nc_accuracy"][
        "upper"
    ][-1]

    result_str = result_str + ", " + str(round(grelu * 100, 1))
    result_str = result_str + ", " + str(round(relu * 100, 1))
    result_str = result_str + ", " + str(round(value[0][2] * 100, 1))
    result_str = result_str + ", " + str(round(value[1][2] * 100, 1))
    result_str = result_str + ", " + str(round(value[2][2] * 100, 1))

    result_str = result_str + "\n"

with open("tables/table_2.csv", "w") as f:
    f.write(result_str)
