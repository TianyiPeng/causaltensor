"""
causaltensor.analysis
---------------------
Modules that run experiments and generate reports:

- semi_synthetic      : semi-synthetic power/accuracy experiments on one dataset (inject known
                        treatment into real or generated panels, compare estimators)
- real_dataset_report : run each estimator on one real dataset (observed ``Z``)
                        and tabulate the results
- aa_tests            : A/A power analysis (zero-treatment verification) on one
                        dataset
- load_tests          : load tests — wall-clock time and peak memory across
                        N×T grids for all estimators

Shared utilities live in ``causaltensor.utils``:
- utils.common               : estimator dispatch (get_tau_from_method*),
                               treatment-info extraction
- semi_synthetic.utils       : baseline building, treatment injection
- utils.treatment_patterns   : Z_iid, Z_block, Z_stagger, Z_adaptive
"""

from causaltensor.analysis.semi_synthetic import main as run_semi_synthetic
from causaltensor.analysis.real_dataset_report import run_real_data_report, save_report
from causaltensor.analysis.load_tests import run_load_test, save_load_test, print_load_test_table

__all__ = [
    "run_semi_synthetic",
    "run_real_data_report",
    "save_report",
    "run_load_test",
    "save_load_test",
    "print_load_test_table",
]
