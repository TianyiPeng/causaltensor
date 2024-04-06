import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
import statsmodels.api as sm
import statsmodels.formula.api as smf
from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.result import Result


class OLSSCResult(Result):
    def __init__(self, baseline = None, M = None, tau=None, beta=None, return_tau_scalar=False):
        super().__init__(baseline = baseline, tau = tau, return_tau_scalar = return_tau_scalar)
        self.beta = beta
        self.M = M # the counterfactual


class OLSSCPanelSolver(PanelSolver):
    def __init__(self, X, Z, select_features=False):
        """
        @param X: T x N matrix to be regressed
        @param Z: T x N intervention matrix of 0s and 1s

        @return M: the baseline matrix 
        @return tau: the treatment effect
        """
        self.X = X
        self.T0, self.Y0, self.Y1, self.control_units = self.preprocess(Z)
        self.select_features = select_features


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
        Y0 = self.X[:, control_units]
        Y1 = self.X[:, ~np.isin(np.arange(N), control_units)]
        T0 = np.where(Z.any(axis=1))[0][0]
        return T0, Y0, Y1, control_units
    

    def feature_selection(self, Y1, alphas = np.logspace(-5,2,50, base=2.0)):
        """
        Use Linear and Lasso Regressions to select which features (control stores) model the outcome (treatment store)

        @param Y1: outcome data for treated unit (T x 1 vector) 
        @param alphas: values to try for Lasso regularization strength
        
        @return ans_select: list of indices of units to select for synthetic control
        @return max_score: maximum validation R^2 achieved
        """

        num_control = self.Y0.shape[1]

        # Case for no control units
        if num_control == 0:
            return np.array([]), -np.inf

        # Create Training and Validation sets
        T0_train = int(0.75*self.T0)
        X_train, y_train = self.Y0[:T0_train, :], Y1[:T0_train]
        X_val, y_val = self.Y0[T0_train:, :], Y1[T0_train:]

        max_score = -np.inf
        ans_select = np.array([])


        # Fit Linear Regression

        # Check if we have a full-row rank matrix
        if num_control > self.T0:    # if features (num_control) are more than examples (T0_train), reduce the features to T0_train
            perm = np.random.permutation(num_control)
            select = np.array([False for i in range(num_control)])
            select[perm[:self.T0]] = True
        else:   # select everything
            select = np.array([True for i in range(num_control)])

        OLS_estimator = LinearRegression(fit_intercept=True)
        OLS_estimator.fit(X_train[:, select], y_train)
        score = OLS_estimator.score(X_val[:, select], y_val)
        if score > max_score:
            max_score = score
            ans_select = select


        # Fit Lasso Regression
        for alpha in alphas:
            lasso_estimator = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
            lasso_estimator.fit(X_train, y_train)

            # Get non-zero features (control units)
            select = (lasso_estimator.coef_ != 0)

            # ignore this alpha if no control units are selected by Lasso
            if np.sum(select) == 0:
                continue

            # Fit a Linear Model with the selected control units
            OLS_estimator = LinearRegression(fit_intercept=True)
            OLS_estimator.fit(X_train[:, select], y_train)
            score = OLS_estimator.score(X_val[:, select], y_val)
    
            if score > max_score:
                max_score = score
                ans_select = select


        #  Handle Underdetermined Linear Regression
        num_selected = np.sum(ans_select)
        if num_selected > self.T0: # if regression selects too many features, unselect some of them
            num_extra = num_selected - self.T0 + 2
            true_idx = np.where(ans_select == True)[0]
            selected_idx = true_idx[np.random.permutation(len(true_idx))[:num_extra]]
            ans_select[selected_idx] = False

        return ans_select, max_score
    

    def ols_inference(self, Y1):
        """
        given some treatment outcome data as well as some control outcome data,
        create a synthetic control and estimate the average treatment effect of the intervention
        
        @param Y1: outcome data for treated unit (T x 1 vector)
        
        @return counterfactual: counterfactual predicted by synthetic control
        @return tau: average treatment effect on test unit redicted by synthetic control
        """
        if self.select_features:
            select, max_score = self.feature_selection(Y1) #get control units from lasso
        else:
            select, max_score = np.array([True]*self.Y0.shape[1]), np.inf  #select all control units

        if select.size == 0: #if no controls, skip
            return None, None
        Y0_control = self.Y0[:,select]   # keep only selected control units
        X_pre = Y0_control[:self.T0,:]    
        y_pre = Y1[:self.T0]
        X_post = Y0_control[self.T0:,:]
        y_post = Y1[self.T0:]

        if max_score > 0:
            X_pre = sm.add_constant(X_pre)
            model = sm.OLS(y_pre, X_pre)
        else:
            y_pre -= np.mean(X_pre, axis = 1)
            y = pd.DataFrame(y_pre, columns = ['y'])
            model = smf.ols(formula='y ~ 1', data = y)
    
        results = model.fit()   
        w = np.mean(X_post, axis=0)
        t_test_index = np.concatenate(([1], w))    
        b = np.mean(y_post)
        if max_score > 0:
            t_test = results.t_test((t_test_index, b))
            Y0_control = sm.add_constant(Y0_control)
            counterfactual = results.predict(Y0_control)
        else:
            t_test = results.t_test((1, b-np.mean(X_post)))
            counterfactual = np.mean(Y0_control, axis=1) + results.params[0]


        tau = b - t_test.effect[0]
        return counterfactual, tau

    

    def fit(self):
        T, S = self.Y1.shape
        tau = 0
        M = np.copy(self.O)
        for s in range(S):
            Y1_s = self.Y1[:,s].reshape((T,))
            counterfactual_s, tau_s = self.ols_inference(Y1_s)
            tau += tau_s
            M[:, self.control_units[s]] = counterfactual_s
        
        tau /= S    
        res = OLSSCResult(M = M, tau = tau)
        return res






#backward compatability 

def ols_synthetic_control(O, Z, select_features=False):
    solver = OLSSCPanelSolver(O, Z, select_features)
    res = solver.fit()
    return res.M, res.tau