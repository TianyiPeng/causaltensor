"""
causaltensor.analysis
---------------------
Modules that run experiments and generate reports:

- semi_synthetic      : semi-synthetic power/accuracy experiments on one dataset (inject known
                        treatment into real or generated panels, compare estimators)
- real_dataset_report : run each estimator on one real dataset (observed ``Z``)
                        and tabulate the results
- power_analysis        : A/A null + empirical power + PNGs + CSVs under
                          ``results/power_analysis/<dataset>/``
- load_tests          : Windows load tests — ``rss_fit_peak_mb``, time,
                        ATT error, CSV+PNG bundles

Shared utilities live in ``causaltensor.utils``:
- utils.common               : estimator dispatch (get_tau_from_method*),
                               treatment-info extraction
- semi_synthetic.utils       : baseline building, treatment injection
- utils.treatment_patterns   : Z_iid, Z_block, Z_stagger, Z_adaptive
"""

from causaltensor.analysis.semi_synthetic import main as run_semi_synthetic
from causaltensor.analysis.real_dataset_report import run_real_data_report, save_report
from causaltensor.analysis.load_tests import (
    aggregate_trials,
    print_load_test_table,
    run_load_test,
    save_load_test,
    save_load_test_bundle,
)

__all__ = [
    "run_semi_synthetic",
    "run_real_data_report",
    "save_report",
    "aggregate_trials",
    "run_load_test",
    "save_load_test",
    "save_load_test_bundle",
    "print_load_test_table",
]
