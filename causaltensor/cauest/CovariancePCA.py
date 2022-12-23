import numpy as np

def random_subset(Ω, K, m):
    O_1 = np.reshape(Ω, -1)
    pos = np.arange(len(O_1))[O_1 == 1]
    O_list = []
    for i in range(K):
        select_pos = np.random.choice(list(pos), m, replace=False)
        new_O_1 = np.zeros((len(O_1)))
        new_O_1[select_pos] = 1
        new_O = np.reshape(new_O_1, Ω.shape)   
        O_list.append(new_O)
    return O_list

def covariance_PCA(O, Ω, suggest_r=-1, return_U = False): #slight issue of cross-validation
    O_ob = O * Ω
    A = O_ob.dot(O_ob.T) / Ω.dot(Ω.T)
    u,s,vh = np.linalg.svd(A, full_matrices=False)
    #print(s[0:2*r])

    def recover(O_train, Ω_train, r):
        U = u[:, :r] * np.sqrt(O.shape[0])

        col_sum = np.sum(Ω_train, axis=0).reshape((Ω.shape[1], 1))  
        Y = O_train.T.dot(U) / col_sum     ### Eq. (4) in their paper
        #Y = O_ob.T.dot(U) / O.shape[0] * Ω.size / np.sum(Ω)

        M = U.dot(Y.T)

        MSE = np.sum(((Ω-Ω_train)*(M-O))**2)
        return MSE, M, U

    if suggest_r == -1:
        K = 2
        p = np.sum(Ω) / np.size(Ω)
        Ω_list = random_subset(Ω, K, int(np.sum(Ω)*p)+1) 

        energy = np.sum(s)
        opt_MSE = 1e9
        opt_r = 1e9

        #Cross Validation to choose the optimal r

        for r in range(1,len(s)):
            #print(np.sum(s[r-1:]) / energy)
            if (np.sum(s[r-1:]) / energy <= 1e-3):
                break
            train_MSE = []
            for i in range(K):
                MSE, Mhat = recover(O*Ω_list[i], Ω_list[i], r)
                train_MSE.append(MSE)
            MSE = np.mean(train_MSE)
            if (MSE < opt_MSE):
                opt_MSE = MSE
                opt_r = r
            #print(MSE, r, np.sum(s[r-1:]) / energy)
    else:
        opt_r = suggest_r
    #print(opt_r)

    MSE, M, U = recover(O_ob, Ω, opt_r)
    
    tau = np.sum((1-Ω)*(O-M)) / np.sum(1-Ω)

    if (return_U):
        return M, tau, U
    else:
        return M, tau