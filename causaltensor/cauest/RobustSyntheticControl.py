import numpy as np
#from sklearn.linear_model import LinearRegression


def stagger_pattern_RSC(O, Z, suggest_r = 1):
    starting_time = O.shape[1] - np.sum(Z, axis=1).astype(int)
    donor_units = np.arange(O.shape[0])[starting_time == O.shape[1]]
    #print(donor_units)

    M = O[donor_units, :]

    u, s, vh = np.linalg.svd(M, full_matrices = False)
    r = suggest_r 
    Mnew = (u[:,:r]*s[:r]).dot(vh[:r, :])
    Mhat = np.zeros_like(O)
    Mhat[donor_units, :] = Mnew

    for i in range(O.shape[0]):
        start = starting_time[i]
        if (start == O.shape[1]):
            continue
        coef = np.linalg.pinv(Mnew[:,:start].T).dot(O[i, :start].T)
        Mhat[i, :] = Mnew.T.dot(coef)

    tau = np.sum(Z*(O-Mhat)) / np.sum(Z)
    return Mhat, tau

def synthetic_control(O, suggest_r=-1, treat_units = [0], starting_time = 100):
    ##Step 1, denoise

    if (starting_time == 0):
        raise Exception('Error: treatment starting at t=0 in synthetic control!') 


    donor_units = []
    for i in range(O.shape[0]):
        if (i not in treat_units):
            donor_units.append(i) 

    M = O[donor_units, :]

    u, s, vh = np.linalg.svd(M, full_matrices = False)

    def recover(r, start, end):
        Mnew = (u[:,:r]*s[:r]).dot(vh[:r, :])
        Mhat = np.zeros_like(O)
        Mhat[donor_units, :] = Mnew

        ##Step 2, linear regression
        Mminus = Mnew[:, :start]
        for i in treat_units: 
            coef = np.linalg.pinv(Mminus.T).dot(O[i, :start].T)
            Mhat[i, :] = Mnew.T.dot(coef)
        
        MSE = np.sum((Mhat - O)[treat_units, start:end]**2)
        return MSE, Mhat

    if (suggest_r == -1):

        energy = np.sum(s)
        valid_start = int(starting_time/2+0.5)

        opt_MSE = 1e9
        opt_r = 1e9

        #Cross Validation to choose the optimal r

        for r in range(1,len(s)):
            if (np.sum(s[r-1:]) / energy <= 0.03):
                break
            MSE, Mhat = recover(r, valid_start, starting_time)
            if (MSE < opt_MSE):
                opt_MSE = MSE
                opt_r = r
            #print(MSE, r, np.sum(s[r-1:]) / energy)

    else:
        opt_r = suggest_r

    #print(opt_r)
    MSE, Mhat = recover(opt_r, starting_time, O.shape[1])

    Z = np.zeros_like(O)
    Z[treat_units, starting_time:] = 1
    tau = np.sum(Z*(O-Mhat)) / np.sum(Z)
    return Mhat, tau