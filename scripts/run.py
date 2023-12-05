"""
Run an experiment.
"""
import os
from warnings import warn
import math
import subprocess
from collections import defaultdict
from logging import Logger

import numpy as np
import torch

from scaffold.runners import (
    run_experiment,
    run_active_set_experiment,
    run_reg_path_experiment,
)

from experiment_utils import experiments, configs, command_line, utils
from exp_configs import EXPERIMENTS  # type: ignore


def compute_index_blocks(nodes, exp_list):
    n = len(exp_list)
    indices = list(range(n))

    # run all jobs in the same node.
    if nodes == 1:
        return [indices]

    block_size = math.ceil(n / nodes)
    blocks = []
    for i in range(nodes - 1):
        blocks.append(indices[block_size * i : block_size * (i + 1)])

    # handle uneven division.
    blocks.append(indices[block_size * (nodes - 1) :])

    # check no experiments were lost.
    total = []
    for b in blocks:
        total += b

    assert len(total) == n

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
    nodes,
    shuffle,
):
    job_string = (
        f"python3 scripts/run.py "
        f"-E {exp_id} -D {data_dir} -R {results_dir}"
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

    if nodes is not None:
        job_string = job_string + f" -N {nodes}"

    if shuffle:
        job_string = job_string + " --shuffle"

    return job_string


# Script #

if __name__ == "__main__":
    (
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
            index,
            sbatch,
            shuffle,
        ),
        _,
        _,
    ) = command_line.get_experiment_arguments()

    logger = utils.get_logger(exp_id, verbose, debug, log_file)

    # lookup experiment #
    if exp_id not in EXPERIMENTS:
        raise ValueError(
            f"Experiment id {exp_id} is not in the experiment list!"
        )
    config = EXPERIMENTS[exp_id]
    logger.warning(f"\n\n====== Running {exp_id} ======\n")

    experiment_list = configs.expand_config_list(config)

    save_path = os.path.join(results_dir, exp_id)

    # experiment_list = experiments.filter_experiment_list(
    #     experiment_list, save_path, force_rerun
    # )

    # avoid clustering hard experiments during array jobs!
    if shuffle:
        rng = np.random.default_rng(seed=123)
        rng.shuffle(experiment_list)

    if nodes is not None and index is None:
        # Break experiment up across desired number of nodes.
        if sbatch is None:
            raise ValueError(
                (
                    "An sbatch file must be specified when "
                    "running on multiple nodes!"
                )
            )

        # run sub-processes to batch submit the experiments.
        logger.warning("Submitting array of jobs.")
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
            nodes,
            shuffle,
        )

        # run the appropriate slurm job.
        subprocess.run(
            (
                f"sbatch --array=0-{nodes-1} --export=ALL,"
                f"JOB_STR='{job_string}' {sbatch} "
            ),
            shell=True,
        )

        logger.warning("Submitted job array!")

    else:
        if index is not None:
            blocks = compute_index_blocks(nodes, experiment_list)

            # run the desired block of experiments
            experiment_list = [experiment_list[i] for i in blocks[index]]

        # run experiments
        logger.warning("Starting experiments.")

        for i, exp_dict in enumerate(experiment_list):
            num_repeats = 10
            logger.warning(
                f"Running Experiment: {i+1}/{len(experiment_list)}."
            )

            # prevent a single failure from crashing all experiments.
            try:
                experiments.run_or_load(
                    logger,
                    exp_dict,
                    run_experiment,
                    data_dir,
                    save_path,
                    force_rerun,
                )
            except Exception as e:
                # only propagate if debugging.
                if debug:
                    raise e

                # log the error
                logger.error(
                    f"Exception {e} encountered while running experiment with configuration {exp_dict}."
                )
                # output the error to the user.
                warn(
                    f"Exception {e} encountered while running experiment with configuration {exp_dict}."
                )

        logger.warning("Experiments complete.")
