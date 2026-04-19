"""
causaltensor.analysis
---------------------
Modules that run experiments and generate reports:

- semi_synthetic      : semi-synthetic power/accuracy experiments (inject known
                        treatment into real or generated panels, compare estimators)
- real_dataset_report : run each estimator on real datasets that include an
                        observed treatment matrix and tabulate the results

Shared utilities live in ``causaltensor.utils``:
- utils.common               : estimator dispatch (get_tau_from_method*),
                               treatment-info extraction
- semi_synthetic.utils       : baseline building, treatment injection
- utils.treatment_patterns   : Z_iid, Z_block, Z_stagger, Z_adaptive

Future additions (same package):
- aa_tests   : A/A power analysis (zero-treatment verification)
- load_tests : scalability benchmarks across N, T sizes
"""

from causaltensor.analysis.semi_synthetic import main as run_semi_synthetic
from causaltensor.analysis.real_dataset_report import run_real_data_report, save_report

__all__ = [
    "run_semi_synthetic",
    "run_real_data_report",
    "save_report",
]
