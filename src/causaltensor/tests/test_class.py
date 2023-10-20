import numpy as np
import pytest
from causaltensor.cauest.OLSSyntheticControl import ols_synthetic_control
from causaltensor.cauest.DID import DID
from causaltensor.cauest.SDID import SDID
from causaltensor.cauest.DebiasConvex import DC_PR_auto_rank, DC_PR_with_suggested_rank
from causaltensor.cauest.MCNNM import MC_NNM_with_cross_validation


np.random.seed(0)


class TestClass:
    @pytest.fixture
    def create_dataset(self):
        O_raw = np.loadtxt('tests\\MLAB_data.txt')
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

    def test_dcpr_auto(self, create_dataset):
        O, Z = create_dataset
        M, tau, std = DC_PR_auto_rank(O, Z)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert tau <= -10 and tau >= -20

    def test_dcpr_rank(self, create_dataset):
        O, Z = create_dataset
        suggest_r = 2
        M, tau, std = DC_PR_with_suggested_rank(O, Z, suggest_r)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert np.linalg.matrix_rank(M) == suggest_r
        assert tau <= -10 and tau >= -20

    def test_mc(self, create_dataset):
        O, Z = create_dataset
        M, a, b, tau = MC_NNM_with_cross_validation(O, 1-Z)
        # TODO: Check for better assertions
        assert M.shape == O.shape
        assert tau <= -15 and tau >= -25




"""
Run the following to run all test cases:
    pytest
Run the following in the terminal to test and get coverage report:
    pytest --cov=.\src\causaltensor\cauest --cov-report=term-missing
"""

