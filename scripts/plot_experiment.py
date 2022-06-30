"""Plot experiment results.
"""
import os
from typing import List, Dict
from functools import reduce

from experiment_utils import configs, utils, files, command_line
from experiment_utils.plotting import plot_grid, plot_cell, defaults

# merge experiment dictionaries.
from exp_configs import EXPERIMENTS
from plot_configs import PLOTS

# Script #

if __name__ == "__main__":
    (
        exp_ids,
        plot_names,
        figures_dir,
        results_dir_base,
        verbose,
        debug,
        log_file,
    ), _ = command_line.get_plotting_arguments()

    logger_name = reduce(lambda acc, x: f"{acc}{x}_", exp_ids, "")
    logger = utils.get_logger(logger_name, verbose, debug, log_file)

    # lookup experiment #
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            raise ValueError(
                f"Experiment id {exp_id} is not in the experiment list!"
            )

    config_list: List[Dict] = reduce(
        lambda acc, eid: acc + EXPERIMENTS[eid], exp_ids, []
    )
    results_dir = [os.path.join(results_dir_base, eid) for eid in exp_ids]

    logger.info(
        f"\n\n====== Making {plot_names} from {exp_ids} results ======\n"
    )

    for plot_name in plot_names:
        dest = os.path.join(figures_dir, "_".join(exp_ids))
        plot_config_list = PLOTS[plot_name]

        for i, plot_config in enumerate(plot_config_list):
            logger.info(f"\n\n Creating {i+1}/{len(plot_config_list)} \n")

            metric_grid = files.load_and_clean_experiments(
                config_list,
                results_dir,
                metrics=plot_config["metrics"],
                row_key=plot_config["row_key"],
                line_key=plot_config["line_key"],
                repeat_key=plot_config["repeat_key"],
                metric_fn=plot_config["metrics_fn"],
                keep=plot_config.get("keep", []),
                remove=plot_config.get("remove", []),
                filter_fn=plot_config.get("filter_fn", None),
                transform_fn=plot_config.get("transform_fn", None),
                processing_fns=plot_config.get("processing_fns", None),
                x_key=plot_config.get("x_key", None),
                x_vals=plot_config.get("x_vals", None),
            )

            plot_grid.plot_grid(
                plot_fn=plot_config.get(
                    "plot_fn", plot_cell.make_convergence_plot
                ),
                results=metric_grid,
                figure_labels=plot_config["figure_labels"],
                line_kwargs=plot_config["line_kwargs"],
                limits=plot_config["limits"],
                log_scale=plot_config["log_scale"],
                base_dir=os.path.join(
                    dest, f"{plot_name}", f"{plot_config['name']}.pdf"
                ),
                settings=plot_config["settings"],
            )

    logger.info("Plotting done.")
