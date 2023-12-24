import numpy as np
import pytest
from causaltensor.cauest.OLSSyntheticControl import ols_synthetic_control
from causaltensor.cauest.DID import DID
from causaltensor.cauest.SDID import SDID
from causaltensor.cauest.DebiasConvex import DC_PR_auto_rank, DC_PR_with_suggested_rank
from causaltensor.cauest.MCNNM import MC_NNM_with_cross_validation, MC_NNM_with_suggested_rank
from causaltensor.matlib.generation_treatment_pattern import iid_treatment, block_treatment_testone
from causaltensor.matlib.generation import low_rank_M0_normal

np.random.seed(0)


class TestSyntheticClass:
    @pytest.fixture
    def create_dataset_factory(self):
        num_individuals = 100
        num_time_periods = 50  
        treatment_level = 0.1 

        

        def create_dataset(did=False, iid=False):
            # Generating synthetic data               
            if did:
                a = np.random.rand(num_individuals)
                b = np.random.rand(num_time_periods)
                M = a[:, None] + b
            else:
                r = 3
                M = low_rank_M0_normal(num_individuals, num_time_periods, r)
            
            self.tau = np.mean(np.abs(M)) * treatment_level

            # Generating treatment pattern
            if not iid:
                Z = block_treatment_testone(num_individuals//2, num_time_periods//2, M)
            else:
                Z = iid_treatment(0.3, (num_individuals, num_time_periods))

            error = np.random.normal(0, np.abs(self.tau)*0.1, (num_individuals, num_time_periods))

            #TODO: Should error be added only to treatment???
            O = M + self.tau * Z + error
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
        M, tau = ols_synthetic_control(O.T, Z.T)
        assert M.shape == O.T.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.01


    def test_synthetic_control(self, create_dataset_factory):
        # Only Block pattern
        O, Z = create_dataset_factory()
        M, tau = ols_synthetic_control(O.T, Z.T)
        assert M.shape == O.T.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.01


        # Feature Selection
        M, tau = ols_synthetic_control(O.T, Z.T, select_features = True)
        assert M.shape == O.T.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.3

    

    def test_mc(self, create_dataset_factory):
        # Block Pattern
        O, Z = create_dataset_factory()
        M, a, b, tau = MC_NNM_with_cross_validation(O, 1-Z)
        assert M.shape == O.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.1

        suggest_r = 3
        M, a, b, tau = MC_NNM_with_suggested_rank(O, 1-Z, suggest_r)
        assert M.shape == O.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.01
        assert np.linalg.matrix_rank(M) == suggest_r

        # IID Pattern
        O, Z = create_dataset_factory(iid=True)
        M, a, b, tau = MC_NNM_with_cross_validation(O, 1-Z)
        assert M.shape == O.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.1

        suggest_r = 3
        M, a, b, tau = MC_NNM_with_suggested_rank(O, 1-Z, suggest_r)
        assert M.shape == O.shape
        error = np.abs(self.tau-tau)/self.tau
        assert error <= 0.01
        assert np.linalg.matrix_rank(M) == suggest_r


    def test_dcpr(self, create_dataset_factory):
        suggest_r = 3

        # Block Pattern
        results = []
        for T in range(3):
            O, Z = create_dataset_factory()
            M, tau, std = DC_PR_auto_rank(O, Z)
            error = np.abs(self.tau-tau)/self.tau
            results.append(error)
        assert np.min(results) <= 0.005

        results = []
        for T in range(3):
            O, Z = create_dataset_factory()
            M, tau, std = DC_PR_with_suggested_rank(O, Z, suggest_r)
            assert np.linalg.matrix_rank(M) == suggest_r
            error = np.abs(self.tau-tau)/self.tau
            results.append(error)
        assert np.min(results) <= 0.005

        # IID Pattern
        results = []
        for T in range(3):
            O, Z = create_dataset_factory(iid=True)
            M, tau, std = DC_PR_auto_rank(O, Z)
            error = np.abs(self.tau-tau)/self.tau
            results.append(error)
        assert np.min(results) <= 0.005

        results = []
        for T in range(3):
            O, Z = create_dataset_factory(iid=True)
            M, tau, std = DC_PR_with_suggested_rank(O, Z, suggest_r)
            assert np.linalg.matrix_rank(M) == suggest_r
            error = np.abs(self.tau-tau)/self.tau
            results.append(error)
        assert np.min(results) <= 0.005




    def test_dcpr_multiple(self):
        n1 = 100
        n2 = 50
        r = 3
        M0 = low_rank_M0_normal(n1 = n1, n2 = n2, r = r) #low rank baseline matrix

        num_treat = 5
        prob = 0.3
        Z = []
        tau = []
        for k in range(num_treat):
            # IID Pattern
            Z.append(iid_treatment(prob=prob, shape=M0.shape)) #treatment patterns
            tau.append(np.random.normal(loc=0, scale=1)) #treatment effects
        
        def adding_noise(M0, Z, tau, Sigma, SigmaZ):
            num_treat = len(Z)
            # TODO: Why are we multiplying normal & uniform noises?
            O = M0 + np.random.normal(loc=0, scale=1, size=M0.shape) * Sigma #add heterogenous noise to the baseline matrix
            for k in range(num_treat):
                # TODO: Why are we adding noise here?
                O += Z[k] * tau[k] + Z[k] * SigmaZ[k] * np.random.normal(loc=0, scale=1, size=M0.shape) #add heterogeneous noise to the treatment effects
            return O
        Sigma = np.random.rand(M0.shape[0], M0.shape[1])
        SigmaZ = []
        for k in range(num_treat):
            SigmaZ.append(np.random.rand(M0.shape[0], M0.shape[1]))

        results = []
        for T in range(5):
            O = adding_noise(M0, Z, tau, Sigma, SigmaZ)
            M, tau_hat, standard_deviation = DC_PR_with_suggested_rank(O, Z, suggest_r=r, method="non-convex") #solving a non-convex optimization to obtain M and tau
            results.append(np.linalg.norm(tau_hat - tau) / np.linalg.norm(tau))     
        results = np.array(results)
        assert M.shape == O.shape
        assert np.mean(results) < 0.08



"""
Run the following to run all test cases:
    pytest
Run the following in the terminal to test and get coverage report:
    pytest --cov=./src/causaltensor/cauest --cov-report=term-missing
"""
