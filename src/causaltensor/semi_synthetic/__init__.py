"""
causaltensor.semi_synthetic
---------------------------
User-facing API for running semi-synthetic causal inference experiments on
custom panel data.

Usage
-----
>>> import numpy as np
>>> from causaltensor.semi_synthetic import run_experiment
>>>
>>> # Your own outcome panel and treatment mask
>>> O = np.random.randn(20, 40)
>>> Z = np.zeros((20, 40)); Z[0, 20:] = 1   # one treated unit, block pattern
>>>
>>> df = run_experiment(
...     O, Z,
...     methods=["OLS_DID", "SDID", "DCPR"],
...     patterns=["Block", "Staggered"],
...     baseline_type="control",
...     treatment_levels=[0.2, 0.1],
...     n_trials=5,
... )
>>> df.groupby(["method", "pattern"])["error"].mean()
"""

from causaltensor.semi_synthetic.aa_test import plot_aa_null_figure, run_aa_test
from causaltensor.semi_synthetic.empirical_power import (
    empirical_critical_abs_tau,
    plot_empirical_power_figure,
    run_empirical_power_grid,
)
from causaltensor.semi_synthetic.experiment import run_experiment

__all__ = [
    "run_experiment",
    "run_aa_test",
    "plot_aa_null_figure",
    "empirical_critical_abs_tau",
    "run_empirical_power_grid",
    "plot_empirical_power_figure",
]
