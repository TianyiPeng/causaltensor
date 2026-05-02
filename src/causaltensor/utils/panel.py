"""Shared helpers for loading observed panels as NumPy arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def default_raw_datasets_path() -> str:
    """Absolute path to ``causaltensor/datasets/raw`` (with trailing slash)."""
    return str(Path(__file__).resolve().parent.parent / "datasets" / "raw") + "/"


def prepare_panel(
    Y_df: pd.DataFrame, Z_df: Optional[pd.DataFrame]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Align and convert to float arrays; binarize ``Z`` for estimators."""
    O = Y_df.values.astype(float)
    if Z_df is None:
        return O, None
    Z_aligned = Z_df.reindex(index=Y_df.index, columns=Y_df.columns)
    Z = (Z_aligned.fillna(0).values > 0).astype(float)
    return O, Z
