"""
Standardized effect magnitudes for the semi-synthetic control baselines.

For each built-in panel, this uses the same control-unit baseline ``M`` as
``analysis.semi_synthetic`` and reports how the existing injection rule
``tau* = Delta * mean(|M|)`` compares with overall and residual variation.

CLI::

    python -m causaltensor.analysis.standardized_effects
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from causaltensor.datasets.dataset_loader import load_dataset
from causaltensor.semi_synthetic.utils import build_baseline_M
from causaltensor.utils.common import extract_treatment_info_from_Z
from causaltensor.utils.panel import default_raw_datasets_path

logger = logging.getLogger(__name__)

DEFAULT_DATASETS: tuple[str, ...] = (
    "smoking",
    "pwt",
    "movielens",
    "dunnhumby",
)
DEFAULT_DELTAS: tuple[float, ...] = (0.05, 0.1, 0.2)


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "results" / "datasets.csv"


def _two_way_residuals(M: np.ndarray) -> np.ndarray:
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def dataset_standardized_effects(
    dataset_name: str,
    *,
    datasets_path: Optional[str] = None,
    deltas: Sequence[float] = DEFAULT_DELTAS,
    baseline_type: str = "control",
) -> dict:
    if datasets_path is None:
        datasets_path = default_raw_datasets_path()
    Y_df, Z_df, _ = load_dataset(dataset_name, datasets_path=datasets_path)
    treated_states, treat_start_years = extract_treatment_info_from_Z(Y_df, Z_df)
    M, n, T = build_baseline_M(
        Y_df.to_numpy(dtype=float),
        treated_states,
        treat_start_years,
        baseline_type,
    )
    mean_abs = float(np.mean(np.abs(M)))
    sd_m = float(np.std(M, ddof=0))
    sd_r = float(np.std(_two_way_residuals(M), ddof=0))
    row = {
        "dataset": dataset_name,
        "n": n,
        "T": T,
        "mean_abs_baseline": mean_abs,
        "sd_baseline": sd_m,
        "sd_residual": sd_r,
    }
    for delta in deltas:
        tau_star = float(delta) * mean_abs
        row[f"tau_star_delta{delta:g}"] = tau_star
        row[f"std_effect_sdM_delta{delta:g}"] = tau_star / sd_m if sd_m else np.nan
        row[f"std_effect_sdR_delta{delta:g}"] = tau_star / sd_r if sd_r else np.nan
    return row


def run_standardized_effects(
    datasets: Sequence[str] = DEFAULT_DATASETS,
    *,
    datasets_path: Optional[str] = None,
    deltas: Sequence[float] = DEFAULT_DELTAS,
    baseline_type: str = "control",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    rows = [
        dataset_standardized_effects(
            name,
            datasets_path=datasets_path,
            deltas=deltas,
            baseline_type=baseline_type,
        )
        for name in datasets
    ]
    df = pd.DataFrame(rows)
    path = Path(output_path) if output_path is not None else default_output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote %s", path)
    print(df.to_string(index=False))
    print(f"\nSaved {path}")
    return df


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(
        description="Write standardized effect magnitudes for semi-synthetic baselines."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Built-in dataset names (default: smoking pwt movielens dunnhumby).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: analysis/results/datasets.csv).",
    )
    args = parser.parse_args(argv)
    return run_standardized_effects(
        datasets=args.datasets,
        output_path=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
