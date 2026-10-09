"""
Compact synthetic robustness check at one treatment level.

The reference DGP is the usual Gaussian low-rank panel with iid Gaussian noise
and an additive heterogeneous effect. Each other setting changes one piece:

- ``ar1``: noise is Gaussian AR(1) with correlation 0.5, same marginal variance
- ``student_t``: noise is Student-t with 3 degrees of freedom, same variance
- ``poisson``: Gamma-factor nonnegative baseline and Poisson noise
- ``baseline``: the cell effect is ``gamma * |M|``, with ``gamma`` set so the
  treated-cell average still equals the reference ATT

Assignment stays IID, Block, Staggered, or Adaptive, and each estimator is run
only on patterns where it is valid.

CLI::

    python -m causaltensor.analysis.synthetic_robustness
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from causaltensor.semi_synthetic.aa_test import (
    DEFAULT_METHODS,
    VALID_PATTERNS,
)
from causaltensor.synthetic.dgp import generate
from causaltensor.utils.common import get_tau_from_method

logger = logging.getLogger(__name__)

_RESULTS_SUBDIR = "synthetic_robustness"

DGPS: tuple[tuple[str, dict], ...] = (
    ("reference", {"noise_type": "normal"}),
    ("ar1", {"noise_type": "ar1"}),
    ("student_t", {"noise_type": "student_t"}),
    ("poisson", {"M_type": "nonneg", "noise_type": "poisson", "mean_M": 1.0}),
    ("baseline", {"noise_type": "normal", "baseline_dependent": True}),
)

_DGP_LABELS = {
    "reference": "Reference",
    "ar1": "AR(1) noise",
    "student_t": "Student-t noise",
    "poisson": "Gamma M, Poisson",
    "baseline": r"Effect $\propto |M|$",
}


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / _RESULTS_SUBDIR


def _relative_error(tau_star: float, tau_hat: float) -> float:
    if tau_star == 0 or np.isnan(tau_hat):
        return float("nan")
    return float(abs(tau_star - tau_hat) / abs(tau_star))


def run_robustness(
    *,
    N: int = 200,
    T: int = 50,
    treatment_level: float = 0.1,
    trials: int = 30,
    seed: int = 0,
    methods: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-trial relative errors for each DGP, pattern, and valid estimator."""
    method_names = list(DEFAULT_METHODS if methods is None else methods)
    rows: List[Dict[str, Any]] = []

    for dgp_idx, (dgp_name, dgp_kwargs) in enumerate(DGPS):
        for pattern_idx, pattern in enumerate(VALID_PATTERNS):
            valid = [
                m
                for m in method_names
                if pattern in DEFAULT_METHODS[m]
            ]
            if not valid:
                continue
            for trial in range(trials):
                trial_seed = int(
                    np.random.SeedSequence(
                        [seed, dgp_idx, pattern_idx, trial]
                    ).generate_state(1)[0]
                )
                O, Z, tau_star = generate(
                    N,
                    T,
                    treatment_pattern=pattern,
                    treatment_level=treatment_level,
                    seed=trial_seed,
                    **dgp_kwargs,
                )
                Zf = np.asarray(Z, dtype=float)
                for method_name in valid:
                    tau_hat = get_tau_from_method(method_name, O, Zf)
                    rows.append(
                        {
                            "dgp": dgp_name,
                            "pattern": pattern,
                            "method": method_name,
                            "trial": trial,
                            "tau_star": tau_star,
                            "tau_hat": tau_hat,
                            "error": _relative_error(tau_star, tau_hat),
                        }
                    )
            if verbose:
                logger.info("Finished dgp=%s pattern=%s", dgp_name, pattern)

    return pd.DataFrame(rows)

def run_and_save(
    out_dir: Optional[Path] = None,
    *,
    N: int = 200,
    T: int = 50,
    treatment_level: float = 0.1,
    trials: int = 30,
    seed: int = 0,
    methods: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    out = Path(out_dir) if out_dir is not None else default_output_dir()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"robustness_N{N}_T{T}_delta{treatment_level:g}_trials{trials}"

    df = run_robustness(
        N=N,
        T=T,
        treatment_level=treatment_level,
        trials=trials,
        seed=seed,
        methods=methods,
        verbose=verbose,
    )
    path_csv = out / f"{stem}_trials.csv"
    df.to_csv(path_csv, index=False)
    summary = (
        df.groupby(["dgp", "pattern", "method"])["error"]
        .agg(mean_error="mean", std_error="std")
        .reset_index()
    )
    path_agg = out / f"{stem}_summary.csv"
    summary.to_csv(path_agg, index=False)
    logger.info("Wrote %s", path_agg)
    return {
        "df": df,
        "summary": summary,
        "paths": {"trials": path_csv, "summary": path_agg},
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Synthetic robustness: AR(1) noise, heavy tails, baseline-dependent effect."
    )
    parser.add_argument("--N", type=int, default=200, help="Units (default: 200).")
    parser.add_argument("--T", type=int, default=50, help="Periods (default: 50).")
    parser.add_argument("--trials", type=int, default=30, help="MC trials per DGP and pattern (default: 30).")
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
        help="Estimator keys (default: all valid for each pattern).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    return run_and_save(
        Path(args.out_dir) if args.out_dir else None,
        N=args.N,
        T=args.T,
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
