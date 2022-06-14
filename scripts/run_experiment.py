"""
Run an experiment.
"""
import os
from warnings import warn
import math
import subprocess
from collections import defaultdict

import numpy as np

from experiment_utils import experiments, configs, command_line, utils
from scaffold.runners import (
    run_experiment,
    run_active_set_experiment,
    run_reg_path_experiment,
)

from exp_configs import EXPERIMENTS  # type: ignore


def compute_dataset_blocks(experiment_list):
    dataset_blocks = defaultdict(list)

    for i, exp_config in enumerate(experiment_list):
        dataset_name = exp_config["data"]["name"]
        dataset_blocks[dataset_name].append(i)

    return list(dataset_blocks.values())


def compute_index_blocks(nodes, experiment_list):
    n = len(experiment_list)
    indices = list(range(n))

    # run all jobs in the same node.
    if nodes == 1:
        return [indices]

    block_size = math.floor(n / nodes)
    blocks = []
    for i in range(nodes - 1):
        blocks.append(indices[block_size * i : block_size * (i + 1)])
    # handle uneven division.
    blocks.append(indices[block_size * (i + 1) :])

    return blocks


def build_job_string(
    exp_id,
    data_dir,
    results_dir,
    force_rerun,
    save_results,
    verbose,
    debug,
    log_file,
    timed,
    shuffle,
    indices,
):
    job_string = (
        f"python3 scripts/run_experiment.py -E {exp_id} -D {data_dir} -R {results_dir}"
    )

    if force_rerun:
        job_string = job_string + " -F"

    if save_results:
        job_string = job_string + " -S 1"

    if verbose:
        job_string = job_string + " -V"

    if debug:
        job_string = job_string + " --debug"

    if log_file is not None:
        job_string = job_string + f" -L {log_file}"

    if timed:
        job_string = job_string + " -T"

    if shuffle:
        job_string = job_string + " --shuffle"

    if indices is not None:
        job_string = job_string + " -I " + " ".join([str(i) for i in indices])

    return job_string


# Script #

if __name__ == "__main__":
    (
        exp_id,
        data_dir,
        results_dir,
        force_rerun,
        save_results,
        verbose,
        debug,
        log_file,
        timed,
        nodes,
        indices,
        sbatch,
        shuffle,
        group_by_dataset,
    ), _ = command_line.get_experiment_arguments()

    logger = utils.get_logger(exp_id, verbose, debug, log_file)

    # lookup experiment #
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Experiment id {exp_id} is not in the experiment list!")
    config = EXPERIMENTS[exp_id]
    logger.warning(f"\n\n====== Running {exp_id} ======\n")

    experiment_list = configs.expand_config_list(config)

    # avoid clustering hard experiments!
    if shuffle:
        rng = np.random.default_rng(seed=123)
        rng.shuffle(experiment_list)

    if nodes is not None or group_by_dataset:
        # Break experiment up across desired number of nodes.
        if group_by_dataset:
            index_blocks = compute_dataset_blocks(experiment_list)
        else:
            index_blocks = compute_index_blocks(nodes, experiment_list)
        # run sub-processes to batch submit the experiments.
        logger.warning("Submitting Jobs.")
        for i, block in enumerate(index_blocks):
            # create string for the job.
            job_string = build_job_string(
                exp_id,
                data_dir,
                results_dir,
                force_rerun,
                save_results,
                verbose,
                debug,
                log_file,
                timed,
                shuffle,
                block,
            )

            if sbatch is None:
                # run multiple sub-processes locally.
                subprocess.Popen(
                    job_string,
                    shell=True,
                    stdin=None,
                    stdout=None,
                    stderr=None,
                    close_fds=True,
                )
            else:
                # run the appropriate slurm job.
                subprocess.run(
                    f"sbatch --export=ALL,JOB_STR='{job_string}' {sbatch} ", shell=True
                )

            logger.warning(f"Submitted job {i+1}/{len(index_blocks)}.")

    else:
        if indices is not None:
            # run only a subset of the experiments
            experiment_list = [experiment_list[i] for i in indices]

        # run experiments
        logger.warning("Starting experiments.")

        results_dir = os.path.join(results_dir, exp_id)
        for i, exp_dict in enumerate(experiment_list):
            num_repeats = 10
            logger.warning(f"Running Experiment: {i+1}/{len(experiment_list)}.")
            logger.info(f"Method: {exp_dict['method']['name']}")
            # wrap in try-except block to prevent a single failure from crashing all experiments.
            runner = run_experiment
            if "active_set" in exp_dict and exp_dict["active_set"] is not None:
                runner = run_active_set_experiment
            elif (
                "regularization_path" in exp_dict
                and exp_dict["regularization_path"] is not None
            ):
                runner = run_reg_path_experiment

            try:
                experiments.run_or_load(
                    logger,
                    exp_dict,
                    runner,
                    data_dir,
                    results_dir,
                    force_rerun,
                )
            except Exception as e:
                if debug:
                    raise
                else:
                    # log the error
                    logger.error(
                        f"Exception {e} encountered while running experiment with configuration {exp_dict}."
                    )
                    # output the error to the user.
                    warn(
                        f"Exception {e} encountered while running experiment with configuration {exp_dict}."
                    )

        logger.warning("Experiments complete.")
