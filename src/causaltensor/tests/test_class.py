import numpy as np
import pytest
from causaltensor.cauest.OLSSyntheticControl import ols_synthetic_control
from causaltensor.cauest.DID import DID
from causaltensor.cauest.SDID import SDID
from causaltensor.cauest.DebiasConvex import DC_PR_auto_rank, DC_PR_with_suggested_rank
from causaltensor.cauest.MCNNM import MC_NNM_with_cross_validation, MC_NNM_with_suggested_rank
from causaltensor.matlib import low_rank_M0_normal
from causaltensor.matlib import iid_treatment
import os

np.random.seed(0)


class TestClass:
    @pytest.fixture
    def create_dataset(self):
        file_path = os.path.join('tests', 'MLAB_data.txt')
        O_raw = np.loadtxt(file_path) # California Smoke Dataset
        O = O_raw[8:, :] ## remove features that are not relevant in this demo
        O = O.T
        Z = np.zeros_like(O) # Z has the same shape as O
        Z[-1, 19:] = 1
        return O, Z
    

    def test_did(self, create_dataset):
        O, Z = create_dataset
        M, tau = DID(O, Z)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert tau <= -20 and tau >= -30


    def test_sdid(self, create_dataset):
        O, Z = create_dataset
        tau = SDID(O, Z)
        # TODO: Check for better assertions
        assert tau <= -10 and tau >= -20


    def test_synthetic_control(self, create_dataset):
        O, Z = create_dataset
        M, tau = ols_synthetic_control(O.T, Z.T)
        # TODO: Check for better assertions
        assert M.shape == O.T.shape
        assert tau <= -10 and tau >= -20

    def test_synthetic_control_feature_selection(self, create_dataset):
        O, Z = create_dataset
        M, tau = ols_synthetic_control(O.T, Z.T, select_features = True)
        # TODO: Check for better assertions
        assert M.shape == O.T.shape
        assert tau <= -10 and tau >= -20

    def test_dcpr(self, create_dataset):
        O, Z = create_dataset
        M, tau, std = DC_PR_auto_rank(O, Z)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert tau <= -10 and tau >= -20

        suggest_r = 2
        M, tau, std = DC_PR_with_suggested_rank(O, Z, suggest_r)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert np.linalg.matrix_rank(M) == suggest_r
        assert tau <= -10 and tau >= -20



    def test_dcpr_multiple(self):
        n1 = 100
        n2 = 100
        r = 2
        M0 = low_rank_M0_normal(n1 = n1, n2 = n2, r = r) #low rank baseline matrix

        num_treat = 5 #number of treatments
        prob = 0.2
        Z = []
        tau = []
        for k in range(num_treat):
            Z.append(iid_treatment(prob=prob, shape=M0.shape)) #treatment patterns
            tau.append(np.random.normal(loc=0, scale=1)) #treatment effects
        
        def adding_noise(M0, Z, tau, Sigma, SigmaZ):
            num_treat = len(Z)
            O = M0 + np.random.normal(loc=0, scale=1, size=M0.shape) * Sigma #add heterogenous noise to the baseline matrix
            for k in range(num_treat):
                O += Z[k] * tau[k] + Z[k] * SigmaZ[k] * np.random.normal(loc=0, scale=1, size=M0.shape) #add heterogeneous noise to the treatment effects
            return O
        Sigma = np.random.rand(M0.shape[0], M0.shape[1])
        SigmaZ = []
        for k in range(num_treat):
            SigmaZ.append(np.random.rand(M0.shape[0], M0.shape[1]))

        results = []
        for T in range(100):
            O = adding_noise(M0, Z, tau, Sigma, SigmaZ)
            M, tau_hat, standard_deviation = DC_PR_with_suggested_rank(O, Z, suggest_r=r, method="non-convex") #solving a non-convex optimization to obtain M and tau
            results.append(np.linalg.norm(tau_hat - tau) / np.linalg.norm(tau))     
        results = np.array(results)
        assert M.shape == O.shape
        assert np.mean(results) < 0.05

        results = []
        for T in range(30):
            O = adding_noise(M0, Z, tau, Sigma, SigmaZ)
            M, tau_hat, standard_deviation = DC_PR_with_suggested_rank(O, Z, suggest_r=r, method="convex") #solving a non-convex optimization to obtain M and tau
            results.append(np.linalg.norm(tau_hat - tau) / np.linalg.norm(tau))     
        results = np.array(results)
        assert M.shape == O.shape
        assert np.mean(results) < 0.1

        # TODO: Any checks on SD?


    def test_mc(self, create_dataset):
        O, Z = create_dataset
        M, a, b, tau = MC_NNM_with_cross_validation(O, 1-Z)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert tau <= -15 and tau >= -25

        suggest_r = 2
        M, a, b, tau = MC_NNM_with_suggested_rank(O, 1-Z, suggest_r)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert tau <= -20 and tau >= -30
        assert np.linalg.matrix_rank(M) == suggest_r



"""
Run the following to run all test cases:
    pytest
Run the following in the terminal to test and get coverage report:
    pytest --cov=./src/causaltensor/cauest --cov-report=term-missing
"""

