import numpy as np
import pytest

from causaltensor.cauest.MCNNM import MCNNMPanelSolver


def make_solver():
    Z = np.zeros((3, 3), dtype=int)
    Z[2, 2] = 1
    return MCNNMPanelSolver(Z=Z)


def test_fit_uses_suggested_rank_first(monkeypatch):
    solver = make_solver()
    O = np.arange(9, dtype=float).reshape(3, 3)
    result = object()
    calls = {}

    def fake_suggested_rank(*, O=None, suggest_r=1):
        calls["O"] = O
        calls["suggest_r"] = suggest_r
        return result

    monkeypatch.setattr(solver, "solve_with_suggested_rank", fake_suggested_rank)

    assert solver.fit(O=O, suggest_r=2, l=0.5, K=3) is result
    assert calls["O"] is O
    assert calls["suggest_r"] == 2


def test_fit_uses_regularizer_when_l_is_provided(monkeypatch):
    solver = make_solver()
    O = np.arange(9, dtype=float).reshape(3, 3)
    M_init = np.ones_like(O)
    result = object()
    calls = {}

    def fake_regularizer(
        *,
        O=None,
        l=None,
        M_init=None,
        eps=1e-7,
        max_iter=2000,
    ):
        calls["O"] = O
        calls["l"] = l
        calls["M_init"] = M_init
        calls["eps"] = eps
        calls["max_iter"] = max_iter
        return result

    monkeypatch.setattr(solver, "solve_with_regularizer", fake_regularizer)

    assert (
        solver.fit(O=O, l=0.5, K=3, M_init=M_init, eps=1e-5, max_iter=7)
        is result
    )
    assert calls["O"] is O
    assert calls["l"] == 0.5
    assert calls["M_init"] is M_init
    assert calls["eps"] == 1e-5
    assert calls["max_iter"] == 7


def test_fit_uses_cross_validation_by_default(monkeypatch):
    solver = make_solver()
    O = np.arange(9, dtype=float).reshape(3, 3)
    list_l = [0.1, 0.2]
    result = object()
    calls = {}

    def fake_cross_validation(*, O=None, K=2, list_l=None):
        calls["O"] = O
        calls["K"] = K
        calls["list_l"] = list_l
        return result

    monkeypatch.setattr(solver, "solve_with_cross_validation", fake_cross_validation)

    assert solver.fit(O=O, list_l=list_l) is result
    assert calls["O"] is O
    assert calls["K"] == 2
    assert calls["list_l"] is list_l

    calls.clear()
    assert solver.fit(O=O, K=4, list_l=list_l) is result
    assert calls["O"] is O
    assert calls["K"] == 4
    assert calls["list_l"] is list_l


def test_fit_requires_observation_matrix():
    solver = make_solver()

    with pytest.raises(ValueError, match="O must be provided"):
        solver.fit()
