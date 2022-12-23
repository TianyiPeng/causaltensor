import numpy as np
import copy

def convex_algorithm_row_specific_treatments(O, Omega, Z, l, suggest = [], eps = 1e-3, debug = False):
    if (len(suggest) == 0):
        M = np.zeros_like(O)
        tau = np.zeros((O.shape[0] ,1))
    else:
        M = suggest[0]
        tau = suggest[1]

    for T in range(2000):
        ## update M
        u,s,vh = np.linalg.svd(Omega*(O - tau*Z) + (1-Omega)*M, full_matrices=False)
        #print(s)
        #print('before thresholding', np.sum(s), tau)
        s = np.maximum(s-l, 0)
        M = (u*s).dot(vh)

        #print(np.sum(s))
        #print(s)
        

        tau_new = np.sum(Omega * Z * (O - M), axis=1) / (np.sum(Omega * Z, axis = 1) + 1e-10) # update tau
        tau_new = tau_new.reshape((O.shape[0], 1))
        if (np.linalg.norm(tau_new - tau) < eps):
            #print('iterations', T)
            return M, tau, 'successful'
        tau = tau_new

        if (debug):
            print(tau)
    return M, tau, 'fail'

def debias_row_specific(M, tau, Z, l):
    u, s, vh = np.linalg.svd(M, full_matrices = False)
    r = np.sum(s >= 1e-5)
    u = u[:, :r]
    vh = vh[:r, :]
    PTperpZ = (np.eye(u.shape[0]) - u.dot(u.T)).dot(Z).dot(np.eye(vh.shape[1]) - vh.T.dot(vh))

    D = np.zeros((M.shape[0], M.shape[0]))
    for i in range(M.shape[0]):
        if (np.sum(Z[i, :]) == 0):
            continue
        Z_i = np.zeros_like(M)
        Z_i[i, :] = Z[i, :]
        PTperpZ_i = (np.eye(u.shape[0]) - u.dot(u.T)).dot(Z_i).dot(np.eye(vh.shape[1]) - vh.T.dot(vh))

        #print(D.shape, np.sum(PTperpZ_i * Z, axis = 1).shape)
        D[i, :] = np.sum(PTperpZ_i * Z, axis = 1)

    delta = np.sum(l * Z*(u.dot(vh)), axis = 1).reshape((M.shape[0], 1))
    tau_d = tau - np.linalg.pinv(D) @ delta
    return tau_d