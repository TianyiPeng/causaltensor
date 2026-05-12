import numpy as np
import numpy.testing as npt
import pandas as pd

from causaltensor.datasets.panel_dataset import PanelDataset


def test_from_arrays_preserves_order_and_casts_to_float():
    O = np.array([[1, 2, 3], [4, 5, 6]], dtype=int)
    unit_idx = np.array(["store_b", "store_a"])
    time_idx = np.array(["t1", "t0", "t2"])
    Z_df = pd.DataFrame(
        [[0, 1, 0], [1, 0, 1]],
        index=unit_idx,
        columns=time_idx,
    )
    X_df = pd.DataFrame(
        [[10, 100], [20, 200]],
        index=unit_idx,
        columns=["size", "sales"],
    )

    dataset = PanelDataset.from_arrays(
        O,
        Z=Z_df,
        X=X_df,
        unit_index=unit_idx,
        time_index=time_idx,
    )

    assert dataset.O.dtype == float
    npt.assert_array_equal(dataset.unit_index, unit_idx)
    npt.assert_array_equal(dataset.time_index, time_idx)
    npt.assert_allclose(dataset.Z, Z_df.to_numpy(dtype=float))
    npt.assert_allclose(dataset.X, X_df.to_numpy(dtype=float))


def test_from_dataframes_sorts_and_aligns_all_inputs():
    O_df = pd.DataFrame(
        [[30, 20], [10, 40]],
        index=pd.Index(["unit_b", "unit_a"], name="entity"),
        columns=pd.Index([2021, 2019], name="year"),
    )
    Z_df = pd.DataFrame(
        [[0, 1], [1, 0]],
        index=pd.Index(["unit_a", "unit_b"], name="entity"),
        columns=pd.Index([2019, 2021], name="year"),
    )
    X_df = pd.DataFrame(
        {"size": [5, 10], "sales": [100, 50]},
        index=pd.Index(["unit_b", "unit_a"], name="entity"),
    )

    dataset = PanelDataset.from_dataframes(O_df=O_df, Z_df=Z_df, X_entity_df=X_df)

    assert dataset.unit_name == "entity"
    assert dataset.time_name == "year"
    expected_units = np.array(["unit_a", "unit_b"])
    expected_times = np.array([2019, 2021])
    npt.assert_array_equal(dataset.unit_index, expected_units)
    npt.assert_array_equal(dataset.time_index, expected_times)
    npt.assert_allclose(dataset.O, O_df.sort_index().sort_index(axis=1).to_numpy(dtype=float))
    npt.assert_allclose(
        dataset.Z,
        Z_df.reindex(index=expected_units, columns=expected_times).to_numpy(dtype=float),
    )
    npt.assert_allclose(
        dataset.X,
        X_df.reindex(index=expected_units).to_numpy(dtype=float),
    )


def test_from_long_builds_panel_and_covariate_means():
    long_df = pd.DataFrame(
        {
            "region": ["b", "b", "a", "a"],
            "year": ["2019", "2018", "2019", "2018"],
            "outcome": [3.0, 2.0, 6.0, 4.0],
            "treat": [1, 0, 0, 1],
            "c1": [10.0, 12.0, 20.0, 22.0],
            "c2": [1.0, 2.0, 3.0, 5.0],
        }
    )

    dataset = PanelDataset.from_long(
        long_df,
        unit_col="region",
        time_col="year",
        outcome_col="outcome",
        treat_col="treat",
        covar_cols=["c1", "c2"],
    )

    assert dataset.unit_name == "region"
    assert dataset.time_name == "year"
    npt.assert_array_equal(dataset.unit_index, np.array(["a", "b"]))
    npt.assert_array_equal(dataset.time_index, np.array([2018, 2019]))
    npt.assert_allclose(dataset.O, np.array([[4.0, 6.0], [2.0, 3.0]]))
    npt.assert_allclose(dataset.Z, np.array([[1.0, 0.0], [0.0, 1.0]]))
    npt.assert_allclose(dataset.X, np.array([[21.0, 4.0], [11.0, 1.5]]))


def test_from_long_handles_whitespace_and_thousands_separators():
    # years include leading/trailing whitespace and comma as thousands separator
    long_df = pd.DataFrame(
        {
            "region": ["b", "b", "a", "a"],
            "year": [" 2,019", "2018 ", "2,019", " 2018"],
            "outcome": [3.0, 2.0, 6.0, 4.0],
            "treat": [1, 0, 0, 1],
            "c1": [10.0, 12.0, 20.0, 22.0],
            "c2": [1.0, 2.0, 3.0, 5.0],
        }
    )

    dataset = PanelDataset.from_long(
        long_df,
        unit_col="region",
        time_col="year",
        outcome_col="outcome",
        treat_col="treat",
        covar_cols=["c1", "c2"],
    )

    npt.assert_array_equal(dataset.time_index, np.array([2018, 2019]))


def test_from_long_handles_decimal_like_numeric_strings():
    # years as decimal-like strings ("2019.0") should also coerce to ints
    long_df = pd.DataFrame(
        {
            "region": ["b", "b", "a", "a"],
            "year": ["2019.0", "2018.0", "2019.0", "2018.0"],
            "outcome": [3.0, 2.0, 6.0, 4.0],
            "treat": [1, 0, 0, 1],
            "c1": [10.0, 12.0, 20.0, 22.0],
            "c2": [1.0, 2.0, 3.0, 5.0],
        }
    )

    dataset = PanelDataset.from_long(
        long_df,
        unit_col="region",
        time_col="year",
        outcome_col="outcome",
        treat_col="treat",
        covar_cols=["c1", "c2"],
    )

    npt.assert_array_equal(dataset.time_index, np.array([2018, 2019]))
