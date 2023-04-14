import numpy as np

def DID(O, Z, tau_star = 0, output_var=False):
    n1 = O.shape[0]
    n2 = O.shape[1]

    a = np.zeros((n1, 1))
    b = np.zeros((n2, 1))
    tau = tau_star

    one_row = np.ones((1, n2))
    one_col = np.ones((n1, 1))
    for T1 in range(2000):
        a_new = np.mean(O-tau*Z-one_col.dot(b.T), axis=1).reshape((n1, 1))
        b_new = np.mean(O-tau*Z-a.dot(one_row), axis=0).reshape((n2, 1))
        if (np.sum((b_new - b)**2) < 1e-7 * np.sum(b**2) and np.sum((a_new - a)**2) < 1e-7 * np.sum(a**2)):
            break
        a = a_new
        b = b_new
        M = a.dot(one_row)+one_col.dot(b.T)
        tau = np.sum(Z*(O-M))/np.sum(Z)

    return M, tau

def DID_with_missing_entries(O, Omega, Z, tau_star = 0, debug=False):
    n1 = O.shape[0]
    n2 = O.shape[1]

    a = np.zeros((n1, 1))
    b = np.zeros((n2, 1))
    tau = tau_star

    one_row = np.ones((1, n2))
    one_col = np.ones((n1, 1))
    M = a.dot(one_row)+one_col.dot(b.T)

    for T1 in range(2000):
        a_new = np.sum(Omega*(O-tau*Z-one_col.dot(b.T)), axis=1).reshape((n1, 1)) / np.sum(Omega, axis=1).reshape((n1, 1))
        b_new = np.sum(Omega*(O-tau*Z-a_new.dot(one_row)), axis=0).reshape((n2, 1)) / np.sum(Omega, axis=0).reshape((n2, 1))

        if (np.sum((b_new - b)**2) < 1e-7 * np.sum(b**2) and np.sum((a_new - a)**2) < 1e-7 * np.sum(a**2)):
            break
        a = a_new
        b = b_new
        M = a.dot(one_row)+one_col.dot(b.T)
        tau = np.sum(Omega*Z*(O-M))/np.sum(Omega*Z)

        if debug:
            print(tau)
    return M, tau
