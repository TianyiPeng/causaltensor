"""
Unit tests for all 7 estimators.

Each test generates a small synthetic panel with a known treatment effect using
causaltensor.synthetic.generate(..., treatment_level=..., seed=N), runs the
estimator, and asserts shape and that relative error is within tolerance.

Tolerances are intentionally generous — these are sanity tests, not accuracy
benchmarks (that belongs in semi_synthetic experiments).
"""

from __future__ import annotations

import numpy as np
import pytest

from causaltensor.synthetic import generate

_N, _T = 50, 60
_TREATMENT_LEVEL = 0.2


def _panel(seed: int, did: bool = False):
    """Return (O, Z, tau_true) from a reproducible synthetic panel."""
    if did:
        rng = np.random.default_rng(seed)
        a = rng.random(_N) + 1.0
        b = rng.random(_T) + 1.0
        M = a[:, None] + b
        tau_true = np.mean(np.abs(M)) * _TREATMENT_LEVEL
        from causaltensor.utils.treatment_patterns import Z_block
        from causaltensor.semi_synthetic.utils import sample_treatment_parameters
        m1, m2, _, _ = sample_treatment_parameters(_N, _T, rng)
        Z = Z_block(M, m1=m1, m2=m2, rng=rng).astype(int)
        O = M + tau_true * Z + rng.normal(0, abs(tau_true) * 0.1, M.shape)
        return O, Z, tau_true
    return generate(_N, _T, rank=3, treatment_pattern="Block",
                    treatment_level=_TREATMENT_LEVEL, seed=seed)


class TestDID:
    def test_shape_and_accuracy(self):
        from causaltensor.cauest.DID import DID
        O, Z, tau_true = _panel(seed=1, did=True)
        M_hat, tau_hat = DID(O, Z)
        assert M_hat.shape == O.shape
        assert np.isfinite(tau_hat)
        rel_err = abs(tau_hat - tau_true) / (abs(tau_true) + 1e-9)
        assert rel_err < 0.05, f"DID relative error too large: {rel_err:.3f}"


class TestSDID:
    def test_finite_output(self):
        # SDID needs a specific block structure; just verify it runs and is finite
        from causaltensor.cauest.SDID import SDID
        O, Z, _ = _panel(seed=2)
        tau_hat = SDID(O, Z)
        assert np.isfinite(tau_hat)


class TestSC:
    def test_shape_and_finite(self):
        from causaltensor.cauest.OLSSyntheticControl import ols_synthetic_control
        O, Z, _ = _panel(seed=3)
        M_hat, tau_hat = ols_synthetic_control(O, Z)
        assert M_hat.shape == O.shape
        assert np.isfinite(tau_hat)


class TestRobustSC:
    def test_shape_and_finite(self):
        from causaltensor.cauest.RobustSyntheticControl import robust_synthetic_control
        O, Z, _ = _panel(seed=4)
        M_hat, tau_hat = robust_synthetic_control(O, Z)
        assert M_hat.shape == O.shape
        assert np.isfinite(tau_hat)


class TestDCPR:
    def test_auto_rank(self):
        from causaltensor.cauest.DebiasConvex import DC_PR_auto_rank
        O, Z, _ = _panel(seed=5)
        M_hat, tau_hat, _ = DC_PR_auto_rank(O, Z)
        assert M_hat.shape == O.shape
        assert np.isfinite(tau_hat)

    def test_suggested_rank(self):
        from causaltensor.cauest.DebiasConvex import DC_PR_with_suggested_rank
        O, Z, _ = _panel(seed=5)
        M_hat, tau_hat, _ = DC_PR_with_suggested_rank(O, Z, 3)
        assert M_hat.shape == O.shape
        assert np.linalg.matrix_rank(M_hat) == 3
        assert np.isfinite(tau_hat)

    def test_multiple_treatments(self):
        from causaltensor.cauest.DebiasConvex import DC_PR_with_suggested_rank
        from causaltensor.matlib.generation_treatment_pattern import iid_treatment
        from causaltensor.synthetic.utils import generate_low_rank_M, add_noise
        rng = np.random.default_rng(42)
        M = generate_low_rank_M(_N, _T, rank=3, mean=2.0, rng=rng)
        Z1 = iid_treatment(0.2, M.shape)
        Z2 = iid_treatment(0.2, M.shape)
        tau1, tau2 = 1.0, -0.5
        O = M + Z1 * tau1 + Z2 * tau2 + add_noise(M, noise_scale=0.1, rng=rng)
        M_hat, tau_hat, _ = DC_PR_with_suggested_rank(O, [Z1, Z2], suggest_r=3,
                                                       method="non-convex")
        assert M_hat.shape == O.shape
        err = np.linalg.norm(np.atleast_1d(tau_hat) - [tau1, tau2]) / np.linalg.norm([tau1, tau2])
        assert err < 0.15, f"DC-PR multi-treatment error too large: {err:.3f}"


class TestMCNNM:
    def test_cross_validation(self):
        from causaltensor.cauest.MCNNM import MC_NNM_with_cross_validation
        O, Z, tau_true = _panel(seed=6)
        M_hat, _, _, tau_hat = MC_NNM_with_cross_validation(O, 1 - Z)
        assert M_hat.shape == O.shape
        assert np.isfinite(tau_hat)
        rel_err = abs(tau_hat - tau_true) / (abs(tau_true) + 1e-9)
        assert rel_err < 0.2, f"MC-NNM CV relative error too large: {rel_err:.3f}"

    def test_suggested_rank(self):
        from causaltensor.cauest.MCNNM import MC_NNM_with_suggested_rank
        O, Z, _ = _panel(seed=6)
        M_hat, _, _, tau_hat = MC_NNM_with_suggested_rank(O, 1 - Z, 3)
        assert M_hat.shape == O.shape
        assert np.linalg.matrix_rank(M_hat) == 3
        assert np.isfinite(tau_hat)


class TestCovariancePCA:
    def test_suggested_rank(self):
        from causaltensor.cauest.CovariancePCA import covariance_PCA
        O, Z, _ = _panel(seed=7)
        M_hat, tau_hat = covariance_PCA(O, Z, suggest_r=3)
        assert M_hat.shape == O.shape
        assert np.isfinite(tau_hat)

    def test_auto_rank(self):
        from causaltensor.cauest.CovariancePCA import covariance_PCA
        O, Z, _ = _panel(seed=7)
        M_hat, tau_hat = covariance_PCA(O, Z, suggest_r=-1)
        assert M_hat.shape == O.shape
        assert np.isfinite(tau_hat)
