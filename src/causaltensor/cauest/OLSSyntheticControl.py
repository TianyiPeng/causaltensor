import numpy as np
from scipy.optimize import fmin_slsqp
from toolz import partial
from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.result import Result


class OLSSCResult(Result):
    def __init__(self, baseline = None, tau=None, beta=None, return_tau_scalar=False, individual_te=None):
        super().__init__(baseline = baseline, tau = tau, return_tau_scalar = return_tau_scalar)
        self.beta = beta
        self.M = baseline # the counterfactual
        self.individual_te = individual_te


class OLSSCPanelSolver(PanelSolver):
    def __init__(self, X, Z, pval=False):
        """
        @param X: T x N matrix to be regressed
        @param Z: T x N intervention matrix of 0s and 1s
        """
        self.X = X
        self.T0, self.Y0, self.Y1, self.control_units, self.treatment_units = self.preprocess(Z)
        self.individual_te = np.zeros(len(self.treatment_units)) 
        self.pval = pval
   


    def preprocess(self, Z):
        """
        Split the observation matrix into Y0, Y1 and T0

        @param Z: T x N intervention matrix of 0s and 1s

        @return T0: number of pre-intervention (baseline) time periods 
        @return Y0: T x control_units observation matrix
        @return Y1: T x treatment_units observation matrix
        @return control_units: column indices of the control units in O
        """
        N = Z.shape[1]
        control_units = np.where(np.all(Z == 0, axis=0))[0]
        treatment_units = np.where(np.any(Z == 1, axis=0))[0]
        Y0 = self.X[:, control_units]
        Y1 = self.X[:, treatment_units]
        T0 = np.where(Z.any(axis=1))[0][0]
        return T0, Y0, Y1, control_units, treatment_units
    

    
    def ols_inference(self, Y1, Y0):
        """
        given some treatment outcome data as well as some control outcome data,
        create a synthetic control and estimate the average treatment effect of the intervention
        
        @param Y1: outcome data for treated unit (T x 1 vector)
        
        @return counterfactual: counterfactual predicted by synthetic control
        @return tau: average treatment effect on test unit redicted by synthetic control
        """
        def loss_w(W, X, y):
            return np.sqrt(np.mean((y - X.dot(W))**2))
        
        X = Y0[:self.T0] # control units in pre-intervention
        y = Y1[:self.T0] # treatment unit in pre-intervention        
        w_start = [1/X.shape[1]]*X.shape[1] # weights for each control units in synthetic control

        weights = fmin_slsqp(partial(loss_w, X=X, y=y),
                            np.array(w_start),
                            f_eqcons=lambda x: np.sum(x) - 1,
                            bounds=[(0.0, 1.0)]*len(w_start),
                            disp=False)
        
        M = Y0 @ weights
        tau = np.mean((Y1-M)[self.T0:])
        return M, tau
        

    def fit(self):
        T = len(self.Y1)
        tau = 0
        M = np.copy(self.X)
        self.individual_te = []
        for i, s in enumerate(self.treatment_units):
            Y1_s = self.Y1[:,i].reshape((T,))
            counterfactual_s, tau_s = self.ols_inference(Y1_s, self.Y0)
            tau += tau_s
            M[:, s] = counterfactual_s
            self.individual_te.append([s, tau_s])
        
        tau /= len(self.treatment_units)  
        
        if self.pval:
            self.individual_te = self.permutation_test()

        res = OLSSCResult(baseline = M, tau = tau, individual_te=self.individual_te)

        
        return res
    

    def permutation_test(self):
        T = len(self.Y1)
        individual_te_control = []
        for i, cu in enumerate(self.control_units):
            Y1_s = self.Y0[:,i].reshape((T,))
            # create a synthetic control for eahc control unit using other control units
            _, tau_s = self.ols_inference(Y1_s, np.hstack((self.Y0[:, :i], self.Y0[:, i+1:]))) 
            individual_te_control.append([cu, tau_s])
        # rank treatment effects for both treatment and control units based on magnitude of tretament effect
        sorted_te = sorted(self.individual_te + individual_te_control, key=lambda x: abs(x[1]), reverse=True)
        n = len(sorted_te)
        p_values = []
        # compute the probability of seeing treatment effects as extreme as current
        for i, unit_te in enumerate(sorted_te):
            if unit_te[0] in set(self.treatment_units):
                p_values.append(unit_te+[round((i+1)/n, 4)])
        p_values = sorted(p_values, key=lambda x: x[0])
        return p_values


            






#backward compatability 

def ols_synthetic_control(O, Z):
    solver = OLSSCPanelSolver(O, Z)
    res = solver.fit()
    return res.M, res.tau


