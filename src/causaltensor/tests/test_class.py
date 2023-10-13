import numpy as np
import pytest
# TODO: Fix Relative Imports
import sys
sys.path.append('c:\\Users\\Arushi Jain\\Dropbox (MIT)\\RAship\\causaltensor')
from src.causaltensor.cauest.OLSSyntheticControl import ols_synthetic_control

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


"""
Run the following in the terminal to test and get coverage report
pytest --cov=.\src\causaltensor\cauest --cov-report=term-missing
"""

