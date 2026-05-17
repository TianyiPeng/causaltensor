"""
causaltensor.real
-----------------
User-facing interface for estimating treatment effects on real (observed) panel data.

Usage
-----
>>> import numpy as np
>>> from causaltensor.real import estimate

>>> O = ...  # your outcome panel (n x T)
>>> Z = ...  # your binary treatment mask (n x T)

# Single method -> returns a float
>>> tau = estimate(O, Z, "OLS_DID")

# Multiple methods -> returns {method: tau_hat}
>>> results = estimate(O, Z, ["OLS_DID", "SDID", "DCPR"])

Available methods
-----------------
'DCPR', 'MC_NNM_CV', 'CovPCA', 'OLS_DID', 'SDID', 'SC', 'RSC'
"""

from causaltensor.real.estimate import VALID_METHODS, estimate

__all__ = ["estimate", "VALID_METHODS"]
