from causaltensor.matlib import SVD
import numpy as np

def hard_impute(O, Ω, r=1, eps=1e-4):
    M = np.zeros_like(O)
    for T in range(2000):
        M_new = SVD(O * Ω + (1-Ω) * M , r)
        if (np.linalg.norm(M-M_new) < np.linalg.norm(M)*eps):
            break
        M = M_new
    return M