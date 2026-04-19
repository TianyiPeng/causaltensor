import warnings

import numpy as np
import pytest

from causaltensor.cauest.DebiasConvex import (
    DCPanelSolver,
    DC_PR_with_suggested_rank,
)


def make_positive_panel(shape=(6, 6)):
    M0 = np.outer(
        np.linspace(1.0, 2.0, shape[0]),
        np.linspace(0.5, 1.5, shape[1]),
    )
    Z = np.zeros_like(M0)
    Z[shape[0] // 2 :, shape[1] // 2 :] = 1
    O = M0 + 0.25 * Z
    return O, Z


def test_nonnegative_fit_warns_disables_std_and_adds_diagnostics():
    O, Z = make_positive_panel()

    solver = DCPanelSolver(Z=Z, O=O)
    with pytest.warns(RuntimeWarning, match="experimental"):
        res = solver.fit(suggest_r=1, method="non-convex", method_non_neg="svd")

    assert res.std is None
    assert res.inference_valid is False
    assert res.non_negative_method == "svd"
    assert res.diagnostics["std_available"] is False
    assert res.diagnostics["inference_valid"] is False
    assert res.diagnostics["method_non_neg"] == "svd"
    assert "projected_design_condition_number" in res.diagnostics
    assert "residual_frobenius_norm" in res.diagnostics


def test_raw_mode_keeps_inference_outputs():
    O, Z = make_positive_panel()

    solver = DCPanelSolver(Z=Z, O=O)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = solver.fit(suggest_r=1, method="non-convex")

    runtime_warnings = [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert res.std is not None
    assert np.isfinite(res.std)
    assert res.inference_valid is True
    assert res.non_negative_method is None
    assert res.diagnostics == {}


def test_nonnegative_fit_returns_nonnegative_baseline():
    O, Z = make_positive_panel()

    solver = DCPanelSolver(Z=Z, O=O)
    with pytest.warns(RuntimeWarning, match="experimental"):
        res = solver.fit(suggest_r=1, method="non-convex", method_non_neg="svd")

    assert np.min(res.baseline) >= -1e-10
    assert res.diagnostics["baseline_min"] >= -1e-10
    assert res.diagnostics["negative_fraction"] == 0


def test_nonnegative_wrapper_returns_none_std():
    O, Z = make_positive_panel()

    with pytest.warns(RuntimeWarning, match="experimental"):
        M, tau, std = DC_PR_with_suggested_rank(
            O,
            Z,
            suggest_r=1,
            method="non-convex",
            method_non_neg="svd",
        )

    assert M.shape == O.shape
    assert np.min(M) >= -1e-10
    assert np.isfinite(tau)
    assert std is None


@pytest.mark.parametrize("method_non_neg", ["clip", "nnmf"])
def test_rejects_unsupported_nonnegative_methods(method_non_neg):
    O, Z = make_positive_panel()

    solver = DCPanelSolver(Z=Z, O=O)
    with pytest.raises(ValueError, match="method_non_neg must be None or 'svd'"):
        solver.fit(suggest_r=1, method="non-convex", method_non_neg=method_non_neg)
