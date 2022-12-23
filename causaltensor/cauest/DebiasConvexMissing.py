import numpy as np
import copy

def convex_algorithm_with_Omega(O, Omega, Z, l, suggest = [], eps = 1e-3, debug = False):

    M = suggest[0]
    tau = suggest[1]
    num_treat = np.sum(Z * Omega)
    for T in range(2000):
        ## update M
        u,s,vh = np.linalg.svd(Omega*(O - tau*Z) + (1-Omega)*M, full_matrices=False)
        #print(s)
        #print('before thresholding', np.sum(s), tau)
        s = np.maximum(s-l, 0)
        M = (u*s).dot(vh)

        #print(np.sum(s))
        #print(s)
        

        tau_new = np.sum(Omega * Z * (O - M)) / num_treat # update tau
        #print('tau(t) is {}, tau(t+1) is {}'.format(tau, tau_new))
        if (np.abs(tau_new - tau) < eps):
            #print('iterations', T)
            return M, tau, 'successful'
        tau = tau_new


        if (debug):
            print(tau)
    return M, tau, 'fail'

def convex_algorithm_with_Omega_with_fixed_effects(O, Omega, Z, l, suggest = [], eps = 1e-3, debug = False):

    M = suggest[0]
    a = suggest[1]
    b = suggest[2]
    tau = suggest[3]

    n1 = O.shape[0]
    n2 = O.shape[1]

    one_row = np.ones((1, n2))
    one_col = np.ones((n1, 1))
    Ω_row_sum = np.sum(Omega, axis = 1).reshape((n1, 1))
    Ω_column_sum = np.sum(Omega, axis = 0).reshape((n2, 1))
    Ω_row_sum[Ω_row_sum==0] = 1
    Ω_column_sum[Ω_column_sum==0] = 1

    for T in range(2000):
        ## update M

        u,s,vh = np.linalg.svd((O - a.dot(one_row) - one_col.dot(b.T) - tau * Z)*Omega + M*(1-Omega), full_matrices = False)

        #print(s)
        #print('before thresholding', np.sum(s), tau)
        s = np.maximum(s-l, 0)
        M_new = (u*s).dot(vh)

        if (np.sum((M-M_new)**2) < 1e-5 * np.sum(M**2)):
            #print('total iterations', T)
            break

        M = M_new
        for T1 in range(2000):
            a = np.sum(Omega*(O-M-one_col.dot(b.T)-tau*Z), axis=1).reshape((n1, 1)) / Ω_row_sum

            b_new = np.sum(Omega*(O-M-a.dot(one_row)-tau*Z), axis=0).reshape((n2, 1)) / Ω_column_sum

            if (np.sum((b_new - b)**2) < 1e-5 * np.sum(b**2)):
                break
            b = b_new
            if (T1 >= 2000):
                break

        tau = np.sum(Omega * Z * (O - M - a.dot(one_row) - one_col.dot(b.T))) / np.sum(Omega * Z)

        if (debug):
            print(tau)
    return M, a, b, tau

def non_convex_algorithm_with_Omega(O, Omega, Z, r, tau = 0, debug = False):
    M = O*Omega
    for T in range(2000):
        u,s,vh = np.linalg.svd(Omega*(O - tau*Z) + (1-Omega)*M, full_matrices=False)
        s[r:] = 0
        M = (u*s).dot(vh)
        tau_new = np.sum(Omega*Z*(O-M)) / np.sum(Omega*Z)
        if (np.abs(tau_new - tau) < 1e-4):
            return M, tau, "successful"
        tau = tau_new
        if (debug):
            print(tau)
    return M, tau, 'fail'