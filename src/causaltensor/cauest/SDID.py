import numpy as np
import cvxpy as cp
from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.result import Result

'''
    An implementation of "Synthetic Difference-in-Differences" from [1]
    
    Created by Tianyi Peng, 2021/03/01
    Credit to Andy Zheng for the revised version, 2022/01/15
    
    [1] Arkhangelsky, Dmitry, Susan Athey, David A. Hirshberg, Guido W. Imbens, and Stefan Wager. Synthetic difference in differences. No. w25532. National Bureau of Economic Research, 2019.
'''
class SDIDResult(Result):
    def __init__(self, baseline = None, tau=None, beta=None, row_fixed_effects=None, column_fixed_effects=None, return_tau_scalar=False):
        super().__init__(baseline = baseline, tau = tau, return_tau_scalar = return_tau_scalar)
        self.beta = beta
        self.row_fixed_effects = row_fixed_effects
        self.column_fixed_effects = column_fixed_effects
        self.M = baseline 



class SDIDPanelSolver(PanelSolver):
    def __init__(self, Z=None, O=None, treat_units = [-1], starting_time = -1):
        '''
        Input: 
            O: nxT observation matrix
            Z: nxT binary treatment matrix  
            treat_units: a list containing elements in [0, 1, 2, ..., n-1]
            starting_time: for treat_units, pre-treatment time is 0, 1, .., starting_time-1
        Output:
            the average treatment effect estimated by [1] 
        '''
        super().__init__(Z)
        if self.Z.shape[2] == 1:
            self.Z = self.Z.reshape(self.Z.shape[0], self.Z.shape[1])
        self.X = O 
        self.treat_units = treat_units
        self.starting_time = starting_time
        if (starting_time == -1):
            self.SDID_preprocess()

    def SDID_preprocess(self):
        n1, n2 = self.X.shape
        self.treat_units = []
        for i in range(n1):
            if self.Z[i, -1] != 0:
                self.treat_units.append(i)
        if len(self.treat_units) == 0:
            print('no treated unit, or the treatment is not a block!!')
            return
        i = self.treat_units[0]
        for j in range(n2-1, -1, -1):
            if self.Z[i, j] == 0:
                break
        self.starting_time = j + 1

    def fit(self):
        self.donor_units = []
        for i in range(self.X.shape[0]):
            if (i not in self.treat_units):
                self.donor_units.append(i)

        Nco = len(self.donor_units)
        Ntr = len(self.treat_units)
        Tpre = self.starting_time
        Tpost = self.X.shape[1] - self.starting_time
        
        ##Step 1, Compute regularization parameter

        D = self.X[self.donor_units, 1:self.starting_time] - self.X[self.donor_units, :self.starting_time-1]
        D_bar = np.mean(D)
        z_square = np.mean((D - D_bar)**2) * (np.sqrt(Ntr * Tpost))

        ##Step 2, Compute w^{sdid}

        w = cp.Variable(Nco)
        w0 = cp.Variable(1)
        G = np.eye(Nco)
        A = np.ones(Nco)
        #G @ w >= 0
        #A.T @ w == 1

        mean_treat = np.mean(self.X[self.treat_units, :Tpre], axis = 0)

        ## solving linear regression with constraints
        prob = cp.Problem(
            cp.Minimize(
                cp.sum_squares(
                    w0+self.X[self.donor_units, :Tpre].T @ w - mean_treat)
                + z_square * Tpre * cp.sum_squares(w)),
                        [G @ w >= 0, A.T @ w == 1])
        prob.solve()
        #print("\nThe optimal value is", prob.value) 
        #print("A solution w is")
        #print(w.value)

        w_sdid = np.zeros(self.X.shape[0]) 
        w_sdid[self.donor_units] = w.value
        w_sdid[self.treat_units] = 1.0 / Ntr

        ##Step 3, Compute l^{sdid}
        l = cp.Variable(Tpre)
        l0 = cp.Variable(1)
        G = np.eye(Tpre)
        A = np.ones(Tpre)
        #G @ w >= 0
        #A.T @ w == 1

        mean_treat = np.mean(self.X[self.donor_units, Tpre:], axis = 1)
        #print(mean_treat)
        #print(mean_treat.shape)

        prob = cp.Problem(
            cp.Minimize(
                cp.sum_squares(
                    l0+self.X[self.donor_units, :Tpre] @ l - mean_treat)),
            [G @ l >= 0, A.T @ l == 1])
        prob.solve()
        #breakpoint()
        #print("\nThe optimal value is", prob.value) 
        #print("A solution w is")
        #print(l.value)

        l_sdid = np.zeros(self.X.shape[1]) 
        l_sdid[:Tpre] = l.value
        l_sdid[Tpre:] = 1.0 / Tpost

        ##Step 4, Compute SDID estimator
        #tau = w_sdid.T @ O @ l_sdid


        n1 = self.X.shape[0]
        n2 = self.X.shape[1]

        weights = w_sdid.reshape((self.X.shape[0], 1)) @ l_sdid.reshape((1, self.X.shape[1]))

        a = np.zeros((n1, 1))
        b = np.zeros((n2, 1))
        tau = 0

        one_row = np.ones((1, n2))
        one_col = np.ones((n1, 1))
        converged = False
        for T1 in range(1000):
            a_new = np.sum((self.X-tau*self.Z-one_col.dot(b.T))*weights, axis=1).reshape((n1, 1)) / np.sum(weights, axis=1).reshape((n1, 1))
            b_new = np.sum((self.X-tau*self.Z-a.dot(one_row))*weights, axis=0).reshape((n2, 1)) / np.sum(weights, axis=0).reshape((n2, 1))
            if (np.sum((b_new - b)**2) < 1e-7 * np.sum(b**2) and
                np.sum((a_new - a)**2) < 1e-7 * np.sum(a**2)):
                converged = True
                break
            a = a_new
            b = b_new
            M = a.dot(one_row)+one_col.dot(b.T)
            tau = np.sum(self.Z*(self.X-M)*weights)/np.sum(self.Z*weights)

        res = SDIDResult(baseline = M, tau = tau)
        return res
    
# backward compatibility
def SDID(O, Z, treat_units = [-1], starting_time = -1):
    solver = SDIDPanelSolver(Z, O, treat_units, starting_time)
    res = solver.fit()
    return res.tau


    