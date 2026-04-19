import numpy as np
import pytest
from types import SimpleNamespace

from causaltensor.cauest.MCNNM import MCNNMPanelSolver


def make_single_treated_cell(shape=(5, 5)):
    Z = np.zeros(shape)
    Z[-1, -1] = 1
    return Z


def test_baseline_projection_keeps_raw_outputs_and_adds_projected_outputs():
    O = -np.ones((5, 5))
    Z = make_single_treated_cell(O.shape)

    solver = MCNNMPanelSolver(Z=Z)
    res = solver.solve_with_regularizer(
        O=O,
        l=0.1,
        max_iter=2,
        baseline_projection="clip_nonnegative",
    )

    assert res.baseline_projected is not None
    assert np.min(res.baseline) < 0
    assert np.min(res.baseline_projected) >= 0
    assert res.tau == res.tau_raw
    assert res.baseline is res.baseline_raw
    assert res.tau_projected <= res.tau
    assert res.projection_diagnostics["clipped_fraction"] > 0
    assert res.projection_diagnostics["baseline_min_projected"] == 0


def test_fit_dispatches_regularizer_with_baseline_projection():
    O = -np.ones((5, 5))
    Z = make_single_treated_cell(O.shape)

    solver = MCNNMPanelSolver(Z=Z)
    res = solver.fit(
        O=O,
        l=0.1,
        max_iter=2,
        baseline_projection="clip_nonnegative",
    )

    assert res.baseline_projection == "clip_nonnegative"
    assert np.min(res.baseline_projected) >= 0
    assert res.projection_diagnostics["clipped_fraction"] > 0


def test_fit_requires_observations():
    solver = MCNNMPanelSolver(Z=make_single_treated_cell())

    with pytest.raises(ValueError, match="O must be provided"):
        solver.fit(l=0.1)


def test_baseline_projection_is_noop_when_raw_baseline_is_nonnegative():
    O = np.ones((5, 5))
    Z = make_single_treated_cell(O.shape)

    solver = MCNNMPanelSolver(Z=Z)
    res = solver.solve_with_regularizer(
        O=O,
        l=0.1,
        max_iter=2,
        baseline_projection="clip_nonnegative",
    )

    np.testing.assert_allclose(res.baseline_projected, res.baseline)
    assert res.tau_projected == pytest.approx(res.tau)
    assert res.projection_diagnostics["clipped_fraction"] == 0
    assert res.projection_diagnostics["clipped_mass"] == 0


def test_rejects_unknown_baseline_projection():
    O = np.ones((5, 5))
    Z = make_single_treated_cell(O.shape)

    solver = MCNNMPanelSolver(Z=Z)
    with pytest.raises(ValueError, match="baseline_projection"):
        solver.solve_with_regularizer(O=O, l=0.1, max_iter=1, baseline_projection="clip")


def projected_result(rank=1):
    M = np.eye(5)
    if rank > 1:
        M[1, 1] = 1
    else:
        M[1:, :] = 0
    baseline = -np.ones((5, 5))
    baseline_projected = np.zeros_like(baseline)
    return SimpleNamespace(
        M=M,
        baseline=baseline,
        baseline_projected=baseline_projected,
        tau=0.0,
        tau_projected=-1.0,
        baseline_projection="clip_nonnegative",
        projection_diagnostics={"clipped_fraction": 1.0},
    )


def test_suggested_rank_path_threads_baseline_projection(monkeypatch):
    O = -np.ones((5, 5))
    Z = make_single_treated_cell(O.shape)
    calls = []

    def fake_solve_with_regularizer(self, **kwargs):
        calls.append(kwargs["baseline_projection"])
        return projected_result(rank=1 if len(calls) == 1 else 2)

    monkeypatch.setattr(MCNNMPanelSolver, "solve_with_regularizer", fake_solve_with_regularizer)

    solver = MCNNMPanelSolver(Z=Z)
    res = solver.solve_with_suggested_rank(O=O, suggest_r=1, baseline_projection="clip_nonnegative")

    assert calls == ["clip_nonnegative", "clip_nonnegative"]
    assert res.baseline_projection == "clip_nonnegative"


def test_cross_validation_path_threads_baseline_projection(monkeypatch):
    O = -np.ones((5, 5))
    Z = make_single_treated_cell(O.shape)
    calls = []

    def fake_solve_with_regularizer(self, **kwargs):
        calls.append(kwargs.get("baseline_projection"))
        if kwargs.get("baseline_projection") == "clip_nonnegative":
            return projected_result(rank=1)
        return SimpleNamespace(M=np.zeros_like(O), baseline=np.zeros_like(O))

    monkeypatch.setattr(MCNNMPanelSolver, "solve_with_regularizer", fake_solve_with_regularizer)

    solver = MCNNMPanelSolver(Z=Z)
    res = solver.solve_with_cross_validation(
        O=O,
        K=2,
        list_l=[0.1],
        baseline_projection="clip_nonnegative",
    )

    assert calls[-1] == "clip_nonnegative"
    assert res.baseline_projection == "clip_nonnegative"


def test_projection_targets_full_baseline_with_fixed_effects_and_covariates():
    O = -np.ones((5, 5))
    Z = make_single_treated_cell(O.shape)
    X = np.arange(O.size, dtype=float).reshape(O.shape)

    solver = MCNNMPanelSolver(Z=Z, X=X)
    res = solver.solve_with_regularizer(
        O=O,
        l=0.1,
        max_iter=2,
        baseline_projection="clip_nonnegative",
    )

    assert res.beta is not None
    assert not np.allclose(res.baseline, res.M)
    assert np.min(res.baseline) < 0
    assert np.min(res.baseline_projected) >= 0
