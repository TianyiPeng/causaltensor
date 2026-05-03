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
>>> tau = estimate(O, Z, "DID")

# Multiple methods -> returns {method: tau_hat}
>>> results = estimate(O, Z, ["DID", "SDID", "DC_PR_auto_rank"])

Available methods
-----------------
'DC_PR_auto_rank', 'MC_NNM_CV', 'CovariancePCA',
'DID', 'SDID', 'SC', 'RobustSyntheticControl'
"""

from causaltensor.real.estimate import VALID_METHODS, estimate

__all__ = ["estimate", "VALID_METHODS"]
