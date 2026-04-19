"""
causaltensor
============
Causal inference for panel data.

User-facing modules
-------------------
real
    Estimate treatment effects on observed panel data.

    >>> from causaltensor.real import estimate
    >>> tau = estimate(O, Z, "DID")

semi_synthetic
    Benchmark estimators on your own data with injected effects.

    >>> from causaltensor.semi_synthetic import run_experiment, run_aa_test
    >>> df = run_experiment(O, Z, methods=["DID", "SDID"], treatment_levels=[0.1, 0.2])

synthetic
    Generate fully synthetic (N, T) panels for benchmarking and testing.

    >>> from causaltensor.synthetic import generate
    >>> O, Z, tau_true = generate(N=50, T=100, treatment_pattern="Block", seed=0)

analysis
    Batch reports and load tests (CLI scripts, save CSV results).

    >>> from causaltensor.analysis import run_load_test, run_real_data_report

datasets
    Built-in panel datasets (smoking, card, basque, …).

    >>> from causaltensor.datasets import load_dataset, available_datasets

Low-level estimators are in ``causaltensor.cauest``.
"""

from causaltensor import real
from causaltensor import semi_synthetic
from causaltensor import synthetic
from causaltensor import analysis
from causaltensor import datasets
from causaltensor import cauest

from causaltensor.real import estimate
from causaltensor.semi_synthetic import run_experiment, run_aa_test
from causaltensor.synthetic import generate

__all__ = [
    # sub-packages
    "real",
    "semi_synthetic",
    "synthetic",
    "analysis",
    "datasets",
    "cauest",
    # top-level convenience re-exports
    "estimate",
    "run_experiment",
    "run_aa_test",
    "generate",
]
