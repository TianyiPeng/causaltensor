import numpy as np
import time


def ALS_solve(M, Ω, r, mu, epsilon=1e-3, max_iterations=100, debug = False):
    """
    Solve probabilistic matrix factorization using alternating least squares.

    Since loss function is non-convex, each attempt at ALS starts from a
    random initialization and returns a local optimum.

    [ Salakhutdinov and Mnih 2008 ]
    [ Hu, Koren, and Volinksy 2009 ]

    Parameters:
    -----------
    M : m x n array
        matrix to complete

    Ω : m x n array
        matrix with entries zero (if missing) or one (if present)

    r : integer
        how many factors to use

    mu : float
        hyper-parameter penalizing norm of factored U, V

    epsilon : float
        convergence condition on the difference between iterative results

    max_iterations: int
        hard limit on maximum number of iterations

    Returns:
    --------
    X: m x n array
        completed matrix
    """
    n1, n2 = M.shape

    U = np.random.randn(n1, r)
    V = np.random.randn(n2, r)

    prev_X = np.dot(U, V.T)

    def solve(M, U, Ω):
        V = np.zeros((M.shape[1], r))
        mu_I = mu * np.eye(U.shape[1])
        for j in range(M.shape[1]):
            X1 = Ω[:, j:j+1].copy() * U
            X2 = X1.T @ X1 + mu_I
            #V[j] = (np.linalg.pinv(X2) @ X1.T @ (M[:, j:j+1].copy())).T
            #print(M[:, j:j+1].shape)
            V[j] = np.linalg.solve(X2, X1.T @ (M[:, j:j+1].copy())).reshape(-1)
        return V

    for _ in range(max_iterations):

        U = solve(M.T, V, Ω.T)

        V = solve(M, U, Ω)


        X = np.dot(U, V.T)

        mean_diff = np.linalg.norm(X - prev_X) / np.linalg.norm(X)
        #if _ % 1 == 0:
        #    logger.info("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
        if (debug):
            print("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
        
        if mean_diff < epsilon:
            break
        prev_X = X

    return X

def unit_test():

    r = 10
    n = 500
    M = np.random.rand(n, r) - 0.5
    M = M @ M.T
    mask = np.random.rand(n, n) < 0.9
    t1 = time.time()
    Mhat = ALS_solve(M, mask, r, 1e-3)
    print(time.time() - t1)
    print('error', np.linalg.norm(Mhat-M) / np.linalg.norm(M))