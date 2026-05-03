"""
causaltensor.synthetic
----------------------
Synthetic panel data generator for benchmarking and testing.

Usage
-----
>>> from causaltensor.synthetic import generate

>>> # Gaussian panel, block treatment (tau_true=0 — inject via run_experiment)
>>> O, Z, _ = generate(N=30, T=50, rank=3, treatment_pattern="Block", seed=0)

>>> # Single known treatment level (compare tau_hat to tau_true)
>>> O, Z, tau_true = generate(N=30, T=50, treatment_pattern="Block",
...                            treatment_level=0.2, seed=0)

>>> # Non-negative panel (Gamma factors + Poisson noise)
>>> O, Z, _ = generate(N=30, T=50, M_type="nonneg", noise_type="poisson",
...                    mean_M=10, treatment_pattern="Staggered", seed=1)

>>> # No treatment — tau_true=0.0 (load testing / A/A tests)
>>> O, Z, tau_true = generate(N=20, T=40, treatment_pattern=None, seed=2)

Available patterns
------------------
'IID', 'Block', 'Staggered', 'Adaptive'
"""

from causaltensor.synthetic.dgp import VALID_PATTERNS, generate

__all__ = ["generate", "VALID_PATTERNS"]
