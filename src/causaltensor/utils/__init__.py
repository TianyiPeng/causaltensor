"""
causaltensor.utils
------------------
Shared utilities used across analysis, semi_synthetic, and real modules.

- ``common``              : estimator dispatch and treatment-info extraction
- ``treatment_patterns``  : Z_iid, Z_block, Z_stagger, Z_adaptive
"""

from causaltensor.utils.common import (
    extract_treatment_info_from_Z,
    get_tau_from_method,
    get_tau_from_method_with_error,
)

__all__ = [
    "extract_treatment_info_from_Z",
    "get_tau_from_method",
    "get_tau_from_method_with_error",
]
