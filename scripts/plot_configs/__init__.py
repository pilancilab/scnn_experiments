"""
Experiment configurations.
"""

import os
from importlib import util
from functools import reduce

plot_list = []

# terrifying dynamic import to find experiment dictionaries.
local_dir = os.path.dirname(__file__)
for module_name in os.listdir(local_dir):
    if module_name == '__init__.py' or module_name[-3:] != '.py':
        continue
    spec = util.spec_from_file_location("module.name", os.path.join(local_dir, module_name))
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        plot_list.append(module.PLOTS)
    except:
        raise ValueError(f"Module {module} must export an experiment list in the variable 'EXPERIMENTS'.")
del module

# merge and export experiment list
PLOTS = reduce(lambda acc, x: {**acc, **x}, plot_list, {})

__all__ = ["PLOTS"]

