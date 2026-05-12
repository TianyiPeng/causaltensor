import numpy as np
import pytest
from causaltensor.cauest.OLSSyntheticControl import ols_synthetic_control
from causaltensor.cauest.DID import DID
from causaltensor.cauest.SDID import SDID
from causaltensor.cauest.DebiasConvex import DC_PR_with_suggested_rank
from causaltensor.cauest.MCNNM import MC_NNM_with_cross_validation, MC_NNM_with_suggested_rank
from causaltensor.cauest.RobustSyntheticControl import robust_synthetic_control
from causaltensor.cauest.CovariancePCA import covariance_PCA
from causaltensor.synthetic.utils import generate_low_rank_M, add_noise
from causaltensor.utils.treatment_patterns import Z_iid, Z_block

_RNG_SEED = 0


class TestSyntheticClass:
    @pytest.fixture
    def create_dataset_factory(self):
        N, T = 100, 50
        treatment_level = 0.1
        rng = np.random.default_rng(_RNG_SEED)

        def create_dataset(did=False, add_fixed_effects=False, r=3, iid=False):
            if did:
                a = rng.random(N)
                b = rng.random(T)
                M = a[:, None] + b
            else:
                M = generate_low_rank_M(N, T, rank=r, rng=rng)
                if add_fixed_effects:
                    a = rng.random(N)
                    b = rng.random(T)
                    M += a[:, None] + b

            self.tau = np.mean(np.abs(M)) * treatment_level

            if not iid:
                Z = Z_block(M, m1=N // 2, m2=T // 2, rng=rng)
            else:
                Z = Z_iid(M, p_treat=0.3, rng=rng)

            O = M + self.tau * Z + add_noise(M, noise_scale=np.abs(self.tau) * 0.1, rng=rng)
            return O, Z
        return create_dataset

    

    def test_did(self, create_dataset_factory):
        # Block pattern
        O, Z = create_dataset_factory(did=True)
        M, tau = DID(O, Z) 
        assert M.shape == O.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.01

        # IID Pattern
        O, Z = create_dataset_factory(did=True, iid=True)
        M, tau = DID(O, Z) 
        assert M.shape == O.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.01


    def test_sdid(self, create_dataset_factory):
        # Only Block pattern
        O, Z = create_dataset_factory()
        tau = SDID(O, Z)
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.02


    def test_synthetic_control(self, create_dataset_factory):
        # Only Block pattern
        O, Z = create_dataset_factory()
        M, tau = ols_synthetic_control(O, Z)
        assert M.shape == O.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.2

    def test_robust_synthetic_control(self, create_dataset_factory):
        O, Z = create_dataset_factory()
        Mhat, tau = robust_synthetic_control(O, Z)
        assert Mhat.shape == O.shape
        assert np.isfinite(tau)
        error = np.abs(self.tau - tau) / self.tau
        assert error <= 0.1

    def test_covariance_pca(self, create_dataset_factory):
        O, Z = create_dataset_factory(did=False, r=3)
        m_hat, tau = covariance_PCA(O, Z, suggest_r=3)
        assert m_hat.shape == O.shape
        assert np.isfinite(tau)
        error = np.abs(self.tau - tau) / self.tau
        assert error <= 0.2

        m_cv, tau_cv = covariance_PCA(O, Z, suggest_r=-1)
        assert m_cv.shape == O.shape
        assert np.isfinite(tau_cv)
        error = np.abs(self.tau - tau_cv) / self.tau
        assert error <= 0.2

    def test_mc(self, create_dataset_factory):
        r = 1
        # Block Pattern
        O, Z = create_dataset_factory(did=False, add_fixed_effects=True, r = r)
        M, a, b, tau = MC_NNM_with_cross_validation(O, 1-Z)
        assert M.shape == O.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.1

        suggest_r = r
        M, a, b, tau = MC_NNM_with_suggested_rank(O, 1-Z, suggest_r)
        assert M.shape == O.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.1
        assert np.linalg.matrix_rank(M) == suggest_r


    def test_dcpr(self, create_dataset_factory):
        suggest_r = 3

        # Block Pattern
        O, Z = create_dataset_factory()
        M, tau, std = DC_PR_with_suggested_rank(O, Z, suggest_r)
        assert np.linalg.matrix_rank(M) == suggest_r
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.05

        # IID Pattern
        O, Z = create_dataset_factory(iid=True)
        M, tau, std = DC_PR_with_suggested_rank(O, Z, suggest_r)
        assert np.linalg.matrix_rank(M) == suggest_r
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.05



    def test_dcpr_multiple(self):
        N, T, r = 100, 50, 3
        rng = np.random.default_rng(_RNG_SEED)

        M0 = generate_low_rank_M(N, T, rank=r, rng=rng)

        num_treat = 2
        taus = [rng.standard_normal() for _ in range(num_treat)]
        Zs = [Z_iid(M0, p_treat=0.3, rng=rng) for _ in range(num_treat)]

        # Heteroskedastic noise on baseline and treatment effects
        Sigma = rng.random(M0.shape)
        O = M0 + rng.standard_normal(M0.shape) * Sigma
        for k in range(num_treat):
            SigmaZ = rng.random(M0.shape)
            O += Zs[k] * taus[k] + Zs[k] * SigmaZ * rng.standard_normal(M0.shape)

        M, tau_hat, _ = DC_PR_with_suggested_rank(O, Zs, suggest_r=r, method="non-convex")
        error = np.linalg.norm(tau_hat - taus) / np.linalg.norm(taus)
        assert M.shape == O.shape
        assert error < 0.2



"""
Run the following to run all test cases:
    pytest
Run the following in the terminal to test and get coverage report:
    pytest --cov=./src/causaltensor/cauest --cov-report=term-missing

"""
