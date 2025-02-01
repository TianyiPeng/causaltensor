import numpy as np
from scipy.optimize import fmin_slsqp
from sklearn.metrics import mean_squared_error
from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.result import Result


class OLSSCResult(Result):
    def __init__(self, baseline = None, tau=None, beta=None, return_tau_scalar=False, individual_te=None, V=None):
        super().__init__(baseline = baseline, tau = tau, return_tau_scalar = return_tau_scalar)
        self.beta = beta # control unit weights
        self.M = baseline # the counterfactual
        self.individual_te = individual_te
        self.V = V  # predictor importance


class OLSSCPanelSolver(PanelSolver):
    def __init__(self, Y, Z, X=None, pval=False):
        """
        @param Y: T x N matrix to be regressed
        @param Z: T x N intervention matrix of 0s and 1s
        @param X: K x N covariates (optional)
        """
        self.Y = Y
        self.X = X
        self.T0, self.Y0, self.Y1, self.X0, self.X1, self.control_units, self.treatment_units = self.preprocess(Z)
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
        Y0 = self.Y[:, control_units]
        Y1 = self.Y[:, treatment_units]
        if self.X is not None:
            X0 = self.X[:, control_units]
            X1 = self.X[:, treatment_units]
        else:
            X0 = None
            X1 = None
        T0 = np.where(Z.any(axis=1))[0][0]
        return T0, Y0, Y1, X0, X1, control_units, treatment_units
    

    
    def ols_inference(self, Y1, Y0, X1=None, X0=None):
        """
        given some treatment outcome data as well as some control outcome data,
        create a synthetic control and estimate the average treatment effect of the intervention
        
        @param Y1: outcome data for treated unit (T x 1 vector)
        
        @return counterfactual: counterfactual predicted by synthetic control
        @return tau: average treatment effect on test unit redicted by synthetic control
        """
        def loss_v(W, y_c, y_t):
            return np.mean((y_t - y_c.dot(W))**2)
        
        def w_constraint(W, y_c, y_t): 
            return np.sum(W) - 1


        
        y_c = Y0[:self.T0] # control units in pre-intervention
        y_t = Y1[:self.T0] # treatment unit in pre-intervention        
        w_start = np.array([1/y_c.shape[1]]*y_c.shape[1]) # weights for each control units in synthetic control

        if X1 is not None:
            v_start = np.array([1/X0.shape[0]]*X0.shape[0]) # weights for each predictor

            def v_constraint(V, W, X0, X1, y_c, y_t): 
                return np.sum(V) - 1
            
            def w_constraint(W, V, X0, X1): 
                return np.sum(W) - 1

            def loss_w(W, V, X0, X1):
                return mean_squared_error(X1, X0.dot(W), sample_weight=V)
            
            def optimize_W(W, V, X0, X1): 
                return fmin_slsqp(loss_w, W, bounds=[(0.0, 1.0)]*len(W), f_eqcons=w_constraint, 
                                           args=(V, X0, X1), disp=False, full_output=True)[0]
            
            def optimize_V(V, W, X0, X1, y_c, y_t):
                w_at_v = optimize_W(W, V, X0, X1)
                return loss_v(w_at_v, y_c, y_t)
            

            V = fmin_slsqp(optimize_V, v_start, args=(w_start, X0, X1, y_c, y_t), bounds=[(0.0, 1.0)]*len(v_start), disp=False, f_eqcons=v_constraint, acc=1e-6)
            W = optimize_W(w_start, V, X0, X1)
            
        else:
            V = None
            W = fmin_slsqp(loss_v, w_start, args=(y_c, y_t),
                            f_eqcons=w_constraint,
                            bounds=[(0.0, 1.0)]*len(w_start),
                            disp=False)
        
        M = Y0 @ W
        tau = np.mean((Y1-M)[self.T0:])
        return M, tau, W, V
        

    def fit(self):
        T = len(self.Y1)
        V = []
        weights = []
        tau = 0
        M = np.copy(self.Y)
        self.individual_te = []
        for i, s in enumerate(self.treatment_units):
            Y1_s = self.Y1[:,i].reshape((T,))
            if self.X is not None:
                K = len(self.X1)
                X1_s = self.X1[:,i].reshape((K,))
                counterfactual_s, tau_s, W_s, V_s = self.ols_inference(Y1_s, self.Y0, X1_s, self.X0)
                V.append(V_s)
            else:
                counterfactual_s, tau_s, W_s, V_s = self.ols_inference(Y1_s, self.Y0)
            tau += tau_s
            M[:, s] = counterfactual_s
            weights.append(W_s)
            self.individual_te.append([s, tau_s])
        
        tau /= len(self.treatment_units)  
        
        if self.pval:
            self.individual_te = self.permutation_test()

        res = OLSSCResult(baseline = M, tau = tau, individual_te=self.individual_te, beta=weights, V=V)

        
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

def ols_synthetic_control(O, Z, X=None):
    solver = OLSSCPanelSolver(O, Z, X)
    res = solver.fit()
    return res.M, res.tau






