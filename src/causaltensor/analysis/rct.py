"""
Randomized earnings trials. The design-based estimate is the difference in
mean post-period earnings, with a 95% interval. The seven estimators are fit
on the same panel.

CLI::

    python -m causaltensor.analysis.rct
"""

from __future__ import annotations

import argparse
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from causaltensor.analysis.real_dataset_report import DEFAULT_METHODS
from causaltensor.datasets.dataset_loader import load_dataset
from causaltensor.utils.common import get_fit_result_from_method
from causaltensor.utils.panel import default_raw_datasets_path, prepare_panel

logger = logging.getLogger(__name__)

_RESULTS_SUBDIR = "rct"
_DATASETS = ("jsa_dc",)
_Z95 = statistics.NormalDist().inv_cdf(0.975)


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / _RESULTS_SUBDIR


def _scalar(value) -> float:
    if value is None:
        return float("nan")
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _difference_in_means(outcome: np.ndarray, treated: np.ndarray):
    a = outcome[treated]
    b = outcome[~treated]
    estimate = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
    return estimate, se, estimate - _Z95 * se, estimate + _Z95 * se


def run_rct(
    dataset: str,
    methods: Optional[Sequence[str]] = None,
    datasets_path: Optional[str] = None,
) -> pd.DataFrame:
    """One row for the randomized estimate and each fitted estimator."""
    path = default_raw_datasets_path() if datasets_path is None else datasets_path
    Y, Z_df, _X = load_dataset(dataset, datasets_path=path)
    O, Z = prepare_panel(Y, Z_df)
    treated = Z.sum(axis=1) > 0
    post = Z.sum(axis=0) > 0
    post_mean = O[:, post].mean(axis=1)

    estimate, se, lo, hi = _difference_in_means(post_mean, treated)
    rows = [{
        "method": "RCT",
        "estimate": estimate,
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        "ok": True,
        "error": "",
    }]

    for method in DEFAULT_METHODS if methods is None else methods:
        res, err = get_fit_result_from_method(method, O, Z)
        estimate = _scalar(res.tau) if res is not None else float("nan")
        ok = err is None and res is not None and np.isfinite(estimate)
        rows.append({
            "method": method,
            "estimate": estimate,
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ok": ok,
            "error": err or ("" if ok else "non-finite estimate"),
        })
        if err is not None:
            logger.info("%s failed: %s", method, err)
    return pd.DataFrame(rows)


def run_and_save(
    out_dir: Optional[Path] = None,
    *,
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    out = Path(out_dir) if out_dir is not None else default_output_dir()
    out.mkdir(parents=True, exist_ok=True)
    names = list(_DATASETS if datasets is None else datasets)
    tables: Dict[str, pd.DataFrame] = {}
    paths: Dict[str, Path] = {}
    for name in names:
        table = run_rct(name, methods=methods)
        path = out / f"{name}.csv"
        table.to_csv(path, index=False)
        logger.info("Wrote %s", path)
        tables[name] = table
        paths[name] = path
    return {"tables": tables, "paths": paths}


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Randomized earnings-trial comparisons.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=list(_DATASETS),
        choices=_DATASETS,
        help="RCT panel to fit (default: jsa_dc).",
    )
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
        choices=DEFAULT_METHODS,
        help="Estimator keys (default: all seven).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_and_save(
        Path(args.out_dir) if args.out_dir else None,
        datasets=args.dataset,
        methods=args.methods,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = main()
    for name, table in res["tables"].items():
        print(f"\n{name}")
        print(table.to_string(index=False))
    print("\nSaved:")
    for k, p in res["paths"].items():
        print(f"  {k}: {p}")
