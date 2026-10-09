"""
PWT Block CATE runner. The experiment itself is
:func:`causaltensor.semi_synthetic.cate.run_cate`. This script loads the panel,
writes CSVs, and leaves figures to ``causaltensor.analysis.plot``.

CLI::

    python -m causaltensor.analysis.cate
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from causaltensor.datasets.dataset_loader import load_dataset
from causaltensor.semi_synthetic.aa_test import DEFAULT_METHODS
from causaltensor.semi_synthetic.cate import run_cate
from causaltensor.utils.panel import default_raw_datasets_path, prepare_panel

logger = logging.getLogger(__name__)

_RESULTS_SUBDIR = "cate"
COVARIATES = ("hc", "csh_i", "openness")


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / _RESULTS_SUBDIR


def _load_pwt():
    Y, Z_df, X = load_dataset("pwt", default_raw_datasets_path())
    X = X.reindex(Y.index)
    keep = X.loc[:, list(COVARIATES)].notna().all(axis=1)
    Y = Y.loc[keep]
    X = X.loc[keep]
    Z_df = None if Z_df is None else Z_df.reindex(index=Y.index, columns=Y.columns)
    O, Z = prepare_panel(Y, Z_df)
    covariates = X.loc[:, list(COVARIATES)].to_numpy(dtype=float)
    return O, Z, covariates


def run_and_save(
    out_dir: Optional[Path] = None,
    *,
    treatment_level: float = 0.1,
    trials: int = 100,
    seed: int = 0,
    methods: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    out = Path(out_dir) if out_dir is not None else default_output_dir()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"cate_pwt_block_delta{treatment_level:g}_trials{trials}"
    O, Z, covariates = _load_pwt()
    metrics, gates = run_cate(
        O,
        Z,
        covariates,
        covariate_names=COVARIATES,
        treatment_level=treatment_level,
        n_trials=trials,
        seed=seed,
        methods=methods,
        verbose=verbose,
    )
    path_metrics = out / f"{stem}_metrics.csv"
    path_gates = out / f"{stem}_gates.csv"
    metrics.to_csv(path_metrics, index=False)
    gates.to_csv(path_gates, index=False)
    logger.info("Wrote %s", path_metrics)
    logger.info("Wrote %s", path_gates)
    return {"metrics": metrics, "gates": gates, "paths": {"metrics": path_metrics, "gates": path_gates}}


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="PWT Block semi-synthetic CATE experiment.")
    parser.add_argument("--trials", type=int, default=100, help="Monte Carlo trials (default: 100).")
    parser.add_argument(
        "--treatment-level",
        type=float,
        default=0.1,
        help="tau* = treatment_level * mean(|M|) (default: 0.1).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed (default: 0).")
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Output folder (default: analysis/results/{_RESULTS_SUBDIR}/).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        choices=tuple(DEFAULT_METHODS.keys()),
        help="Estimator keys (default: all seven).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_and_save(
        Path(args.out_dir) if args.out_dir else None,
        treatment_level=args.treatment_level,
        trials=args.trials,
        seed=args.seed,
        methods=args.methods,
        verbose=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = main()
    print("\nSaved:")
    for k, p in res["paths"].items():
        print(f"  {k}: {p}")
