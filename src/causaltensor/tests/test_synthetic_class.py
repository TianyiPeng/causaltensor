import numpy as np
import pytest
from causaltensor.cauest.OLSSyntheticControl import ols_synthetic_control
from causaltensor.cauest.DID import DID
from causaltensor.cauest.SDID import SDID
from causaltensor.cauest.DebiasConvex import DC_PR_auto_rank, DC_PR_with_suggested_rank
from causaltensor.cauest.MCNNM import MC_NNM_with_cross_validation, MC_NNM_with_suggested_rank
from causaltensor.matlib import low_rank_M0_normal
from causaltensor.matlib import iid_treatment

np.random.seed(0)


class TestSyntheticClass:
    @pytest.fixture
    def create_dataset_factory(self):
        num_individuals = 100
        num_time_periods = 50
        Z = np.zeros((num_individuals, num_time_periods))
        # TODO: Check if we want multiple test stores for non dcpr?
        Z[-1,num_time_periods//2:] = 1
        self.tau = 0.05  # Assumed treatment effect
        error = np.random.normal(0, 1, (num_individuals, num_time_periods))

        def create_dataset(did=False):
            #TODO: Why are we not doing tau = (Y1.post-Y1.pre)-(Y0.post-Y0.pre)?
            if did:
                # Generating synthetic data               
                a = np.random.normal(0, 1, num_individuals)
                b = np.random.normal(0, 1, num_time_periods)
                O = a[:, None] + b + self.tau * Z + error
            else:
                r = 3
                # TODO: Is M low-rank for synthetic control and MCMM?
                U = np.random.rand(num_individuals, r)
                V = np.random.rand(r, num_time_periods)
                M = U @ V
                O = M + self.tau * Z + error
            return O, Z
        return create_dataset

    

    def test_did(self, create_dataset_factory):
        O, Z = create_dataset_factory(did=True)
        M, tau = DID(O, Z)
        # TODO: Check for better assertions  
        assert np.abs(self.tau-tau) <= 0.001


    def test_sdid(self, create_dataset_factory):
        # TODO: What is the equation of SDID?
        pass


    def test_synthetic_control(self, create_dataset_factory):
        O, Z = create_dataset_factory(did=False)
        M, tau = ols_synthetic_control(O.T, Z.T)
        # TODO: Check for better assertions
        assert M.shape == O.T.shape
        assert np.abs(self.tau-tau) <= 0.001

    def test_synthetic_control_feature_selection(self, create_dataset_factory):
        O, Z = create_dataset_factory(did=False)
        M, tau = ols_synthetic_control(O.T, Z.T, select_features = True)
        # TODO: Check for better assertions
        assert M.shape == O.T.shape
        assert np.abs(self.tau-tau) <= 0.001

    def test_dcpr(self, create_dataset_factory):
        O, Z = create_dataset_factory(did=False)
        M, tau, std = DC_PR_auto_rank(O, Z)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert np.abs(self.tau-tau) <= 0.001

        suggest_r = 3
        M, tau, std = DC_PR_with_suggested_rank(O, Z, suggest_r)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert np.linalg.matrix_rank(M) == suggest_r
        assert np.abs(self.tau-tau) <= 0.001



    # def test_dcpr_multiple(self):
    #     n1 = 100
    #     n2 = 100
    #     r = 2
    #     M0 = low_rank_M0_normal(n1 = n1, n2 = n2, r = r) #low rank baseline matrix

    #     num_treat = 5 #number of treatments
    #     prob = 0.2
    #     Z = []
    #     tau = []
    #     for k in range(num_treat):
    #         Z.append(iid_treatment(prob=prob, shape=M0.shape)) #treatment patterns
    #         tau.append(np.random.normal(loc=0, scale=1)) #treatment effects
        
    #     def adding_noise(M0, Z, tau, Sigma, SigmaZ):
    #         num_treat = len(Z)
    #         O = M0 + np.random.normal(loc=0, scale=1, size=M0.shape) * Sigma #add heterogenous noise to the baseline matrix
    #         for k in range(num_treat):
    #             O += Z[k] * tau[k] + Z[k] * SigmaZ[k] * np.random.normal(loc=0, scale=1, size=M0.shape) #add heterogeneous noise to the treatment effects
    #         return O
    #     Sigma = np.random.rand(M0.shape[0], M0.shape[1])
    #     SigmaZ = []
    #     for k in range(num_treat):
    #         SigmaZ.append(np.random.rand(M0.shape[0], M0.shape[1]))

    #     results = []
    #     for T in range(100):
    #         O = adding_noise(M0, Z, tau, Sigma, SigmaZ)
    #         M, tau_hat, standard_deviation = DC_PR_with_suggested_rank(O, Z, suggest_r=r, method="non-convex") #solving a non-convex optimization to obtain M and tau
    #         results.append(np.linalg.norm(tau_hat - tau) / np.linalg.norm(tau))     
    #     results = np.array(results)
    #     assert M.shape == O.shape
    #     assert np.mean(results) < 0.05

    #     results = []
    #     for T in range(30):
    #         O = adding_noise(M0, Z, tau, Sigma, SigmaZ)
    #         M, tau_hat, standard_deviation = DC_PR_with_suggested_rank(O, Z, suggest_r=r, method="convex") #solving a non-convex optimization to obtain M and tau
    #         results.append(np.linalg.norm(tau_hat - tau) / np.linalg.norm(tau))     
    #     results = np.array(results)
    #     assert M.shape == O.shape
    #     assert np.mean(results) < 0.1

    #     # TODO: Any checks on SD?


    def test_mc(self, create_dataset_factory):
        O, Z = create_dataset_factory(did=False)
        M, a, b, tau = MC_NNM_with_cross_validation(O, 1-Z)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert np.abs(self.tau-tau) <= 0.001

        suggest_r = 3
        M, a, b, tau = MC_NNM_with_suggested_rank(O, 1-Z, suggest_r)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert np.abs(self.tau-tau) <= 0.001
        assert np.linalg.matrix_rank(M) == suggest_r



"""
Run the following to run all test cases:
    pytest
Run the following in the terminal to test and get coverage report:
    pytest --cov=./src/causaltensor/cauest --cov-report=term-missing
"""

