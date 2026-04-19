"""
causaltensor.datasets
---------------------
Built-in panel datasets for causal inference.

Usage
-----
>>> from causaltensor.datasets import load_dataset, available_datasets

>>> print(available_datasets())
>>> Y_df, Z_df, X_df = load_dataset("smoking")
>>> O = Y_df.values   # (n_units, n_periods)
>>> Z = Z_df.values   # binary treatment mask
"""

from causaltensor.datasets.dataset_loader import available_datasets, load_dataset
from causaltensor.datasets.panel_dataset import PanelDataset

__all__ = ["load_dataset", "available_datasets", "PanelDataset"]
