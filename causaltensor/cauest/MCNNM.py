import numpy as np
import time

def missing_algorithm(O, Ω, l, fixed_effects=False, suggest = []):
    n1 = O.shape[0]
    n2 = O.shape[1]

    if (len(suggest) == 0):
        M = np.zeros_like(O)
        a = np.zeros((n1, 1))
        b = np.zeros((n2, 1))
    else:
        M = suggest[0]
        a = suggest[1]
        b = suggest[2]
    


    one_row = np.ones((1, n2))
    one_col = np.ones((n1, 1))
    Ω_row_sum = np.sum(Ω, axis = 1).reshape((n1, 1))
    Ω_column_sum = np.sum(Ω, axis = 0).reshape((n2, 1))
    Ω_row_sum[Ω_row_sum==0] = 1
    Ω_column_sum[Ω_column_sum==0] = 1
    #t1 = time.time()
    #t_time = 0
    for T in range(2000):
        # update M
        if fixed_effects:
            #print((O+a.dot(one_row) + one_col.dot(b.T))*Ω + M*(1-Ω))
            #t3 = time.time()
            u,s,vh = np.linalg.svd((O - a.dot(one_row) - one_col.dot(b.T))*Ω + M*(1-Ω), full_matrices = False)
            #t4 = time.time()
            #t_time += t4 - t3
        else:    
            u,s,vh = np.linalg.svd(O*Ω + M*(1-Ω), full_matrices = False)
        s = np.maximum(s - l, 0)
        M_new = (u*s).dot(vh)

        if (np.sum((M-M_new)**2) < 1e-5 * np.sum(M**2)):
            #print('total iterations', T)
            break

        M = M_new
        if fixed_effects:
            for T1 in range(2000):
                a = np.sum(Ω*(O-M-one_col.dot(b.T)), axis=1).reshape((n1, 1)) / Ω_row_sum

                b_new = np.sum(Ω*(O-M-a.dot(one_row)), axis=0).reshape((n2, 1)) / Ω_column_sum

                if (np.sum((b_new - b)**2) < 1e-5 * np.sum(b**2)):
                    break
                b = b_new
            if (T1 >= 2000):
                break
            #if (T1 >= 10):
            #    print(T1)
    #t2 = time.time()
    #print(t2 - t1, t_time)
        
                
    if fixed_effects:
        tau = np.sum((1-Ω)*(O-M-a.dot(one_row)-one_col.dot(b.T))) / np.sum(1-Ω)
    else:
        tau = np.sum((1-Ω)*(O-M)) / np.sum(1-Ω)
    return M, a, b, tau

def tune_missing_algorithm_with_rank(O, Ω, fixed_effects=True, suggest_r = 1, suggest_lambda = -1, real_data = False):
    if (np.sum(np.sum(Ω, axis=1)==0)>0 or np.sum(np.sum(Ω, axis=0)==0) > 0):
        return O, 0

    coef = 1.1
    l = 0
    if (real_data):
        if (suggest_lambda > 0):
            l = suggest_lambda
        else:
            u, s, vh = np.linalg.svd(O*Ω, full_matrices = False)
            l = s[1]*coef
    else:
        if (suggest_lambda > 0):
            l = suggest_lambda
            M, a, b, tau = missing_algorithm(O, Ω, l, fixed_effects)
            return M, a, b, tau
        else:
            u, s, vh = np.linalg.svd(O*Ω, full_matrices = False)
            l = s[1]*coef
        

    pre_M, pre_a, pre_b, pre_tau = missing_algorithm(O, Ω, l, fixed_effects)
    l = l / coef
    while (True):
        M, a, b, tau = missing_algorithm(O, Ω, l, fixed_effects, suggest = [pre_M, pre_a, pre_b])
        #print(l, np.linalg.matrix_rank(M))
        if (np.linalg.matrix_rank(M) > suggest_r):
            return pre_M, pre_a, pre_b, pre_tau
        pre_M = M
        pre_a = a
        pre_b = b
        pre_tau = tau
        l = l / coef

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


### not applicable currently
def tune_missing_algorithm_with_crossing_validation(O, Ω, fixed_effects=True, suggest_lambda = -1):
    if (np.sum(np.sum(Ω, axis=1)==0)>0 or np.sum(np.sum(Ω, axis=0)==0) > 0):
        return O, 0

    one_row = np.ones((1, O.shape[1]))
    one_col = np.ones((O.shape[0], 1))
    
    p = np.sum(Ω) / np.size(Ω)
    ### determine the initial l
    K = 2
    Ω_list = random_subset(Ω, K, int(np.sum(Ω)*p)+1)
    l = 1e9
    for i in range(K):
        if fixed_effects == True:
            n1 = O.shape[0]
            n2 = O.shape[1]
            a = np.zeros((n1, 1))
            b = np.zeros((n2, 1))
            Ω_row_sum = np.sum(Ω_list[i], axis = 1).reshape((n1, 1))
            Ω_column_sum = np.sum(Ω_list[i], axis = 0).reshape((n2, 1))
            Ω_row_sum[Ω_row_sum==0] = 1
            Ω_column_sum[Ω_column_sum==0] = 1
            for T1 in range(2000):
                a = np.sum(Ω_list[i]*(O-one_col.dot(b.T)), axis=1).reshape((n1, 1)) / Ω_row_sum

                b_new = np.sum(Ω_list[i]*(O-a.dot(one_row)), axis=0).reshape((n2, 1)) / Ω_column_sum

                if (np.sum((b_new - b)**2) < 1e-5 * np.sum(b**2)):
                    break
                b = b_new
            u, s, vh = np.linalg.svd((O-a.dot(one_row)-one_col.dot(b.T))*Ω_list[i], full_matrices = False)

            # M = np.zeros_like(O)
            # tau = np.sum((1-Ω)*(O-M-a.dot(one_row)-one_col.dot(b.T))) / np.sum(1-Ω)
            # return M, a, b, tau
            l = min(l, s[1]*1.1)
            #print(s[0:5])
        else:
            u, s, vh = np.linalg.svd(O*Ω_list[i], full_matrices = False)
            l = min(l, s[1]*1.1)
    if (suggest_lambda > 0):
        l = suggest_lambda

    suggest_M = []
    for i in range(K):
        suggest_M.append(np.zeros_like(O))

    opt_MSE = 1e9
    opt_l = 1e9
    counter = 0
    while (True):
        '''
            cross-validation
        '''
        valid_MSE = []
        train_MSE = []
        test_MSE = []
        for i in range(K):
            M, a, b, tau = missing_algorithm(O, Ω_list[i], l, fixed_effects, suggest_M = suggest_M[i])
            #M, a, b, tau = missing_algorithm(O, Ω_list[i], l, fixed_effects, suggest_M = suggest_M[i], suggest_a = suggest_a[i], suggest_b = suggest_b[i])
            valid_MSE.append(np.sum(((O-M-a.dot(one_row)-one_col.dot(b.T))*(Ω-Ω_list[i]))**2) / np.sum(Ω-Ω_list[i]))
            train_MSE.append(np.sum(((O-M-a.dot(one_row)-one_col.dot(b.T))*(Ω_list[i]))**2) / np.sum(Ω_list[i]))
            #test_MSE.append(np.sum(((O-100-M-a.dot(one_row)-one_col.dot(b.T))*(1-Ω))**2) / np.sum(1-Ω))
            suggest_M[i] = M
        MSE = np.mean(valid_MSE)
        if (MSE < opt_MSE):
            opt_MSE = MSE
            opt_l = l
            counter = 0
        else:
            counter += 1
        #print(l, np.linalg.matrix_rank(suggest_M[0]), np.sqrt(np.mean(train_MSE)), np.sqrt(np.mean(valid_MSE)), np.sqrt(np.mean(test_MSE)))
        
        if (counter >= 2 or MSE > 2*np.mean(train_MSE)): #the condition for stopping decreasing lambda
            break
        l = l / 1.2
    #print('opt_l is ', opt_l)
    M, a, b, tau = missing_algorithm(O, Ω, opt_l, fixed_effects, suggest_M = np.zeros_like(O))
    return M, a, b, tau