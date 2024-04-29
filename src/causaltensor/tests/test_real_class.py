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


class TestRealClass:
    @pytest.fixture
    def create_dataset(self):
        file_path = os.path.join('tests', 'MLAB_data.txt')
        O_raw = np.loadtxt(file_path) # California Smoke Dataset
        X = O_raw[1:8, :]  ## predictors
        O = O_raw[8:, :] ## remove features that are not relevant in this demo
        O = O.T
        X = X.T
        Z = np.zeros_like(O) # Z has the same shape as O
        Z[-1, 19:] = 1
        return O, Z, X
    

    def test_did(self, create_dataset):
        O, Z, _ = create_dataset
        M, tau = DID(O, Z)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert tau <= -20 and tau >= -30


    def test_sdid(self, create_dataset):
        O, Z, _ = create_dataset
        tau = SDID(O, Z)
        # TODO: Check for better assertions
        assert tau <= -10 and tau >= -20


    def test_synthetic_control(self, create_dataset):
        O, Z, X = create_dataset
        # SC on only outcomes
        M, tau = ols_synthetic_control(O.T, Z.T)
        assert M.shape == O.T.shape
        assert tau <= -10 and tau >= -20
        # SC on predictors
        M, tau = ols_synthetic_control(O.T, Z.T, X.T)
        assert M.shape == O.T.shape
        assert tau <= -10 and tau >= -20

 

    def test_dcpr(self, create_dataset):
        O, Z, _ = create_dataset
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


    def test_mc(self, create_dataset):
        O, Z, _ = create_dataset
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

