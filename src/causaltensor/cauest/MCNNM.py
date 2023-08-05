import numpy as np

def MC_NNM_with_l(O, Ω, l, suggest = [], covaraites = []):
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
    for T in range(2000):
        # update M
        u,s,vh = np.linalg.svd((O - a.dot(one_row) - one_col.dot(b.T))*Ω + M*(1-Ω), full_matrices = False)
        
        s = np.maximum(s - l, 0)
        M_new = (u*s).dot(vh)

        if (np.sum((M-M_new)**2) < 1e-7 * np.sum(M**2)):
            #print('total iterations', T)
            break

        M = M_new
        for T1 in range(2000):
            a = np.sum(Ω*(O-M-one_col.dot(b.T)), axis=1).reshape((n1, 1)) / Ω_row_sum

            b_new = np.sum(Ω*(O-M-a.dot(one_row)), axis=0).reshape((n2, 1)) / Ω_column_sum

            if (np.sum((b_new - b)**2) < 1e-7 * np.sum(b**2)):
                break
            b = b_new
        if (T1 >= 2000):
            break
            
    tau = np.sum((1-Ω)*(O-M-a.dot(one_row)-one_col.dot(b.T))) / np.sum(1-Ω)
    return M, a, b, tau

def MC_NNM_with_suggested_rank(O, Ω, suggest_r = 1):
    if (np.sum(np.sum(Ω, axis=1)==0)>0 or np.sum(np.sum(Ω, axis=0)==0) > 0):
        print("Since a whole row or a whole column is treated, the matrix completion algorithm won't work!")
        return O, 0

    suggest_r = min(suggest_r, O.shape[0])
    suggest_r = min(suggest_r, O.shape[1])
    coef = 1.1
    u, s, vh = np.linalg.svd(O*Ω, full_matrices = False)
    l = s[1]*coef    

    pre_M, pre_a, pre_b, pre_tau = MC_NNM_with_l(O, Ω, l)
    l = l / coef
    while (True):
        M, a, b, tau = MC_NNM_with_l(O, Ω, l, suggest = [pre_M, pre_a, pre_b])
        if (np.linalg.matrix_rank(M) >= suggest_r):
            return M, a, b, tau
        pre_M = M
        pre_a = a
        pre_b = b
        pre_tau = tau
        l = l / coef

def MC_NNM_with_cross_validation(O, Ω, K=5, list_l = []):
    """
    K-fold cross validation
    
    """

    one_row = np.ones((1, O.shape[1]))
    one_col = np.ones((O.shape[0], 1))
    def MSE_validate(M, a, b, valid_Ω):
        delta = (valid_Ω)*(O-M-a.dot(one_row)-one_col.dot(b.T))
        return np.sum(delta**2) / np.size(M)

    if (np.sum(np.sum(Ω, axis=1)==0)>0 or np.sum(np.sum(Ω, axis=0)==0) > 0):
        print("Since a whole row or a whole column is treated, the matrix completion algorithm won't work!")
        return O, 0  

    #K-fold cross validation
    train_list = []
    valid_list = []
    p = np.sum(Ω) / np.size(Ω)
    np.random.seed(2)
    for k in range(K):
        select = np.random.rand(O.shape[0], O.shape[1]) <= p
        train_list.append(Ω * select)
        valid_list.append(Ω * (1 - select)) 

    if (len(list_l) == 0):
        # smart tuning of lambda
        pre_M_s = []
        pre_a_s = []
        pre_b_s = []
        coef = 1.1
        _, s, _ = np.linalg.svd(O*Ω, full_matrices = False)
        l = s[1]*coef  
        l_opt = l
        pre_error = 0
        for k in range(K):
            M, a, b, tau = MC_NNM_with_l(O, train_list[k], l)
            pre_M_s.append(M)
            pre_a_s.append(a)
            pre_b_s.append(b)
            pre_error += MSE_validate(M, a, b, valid_list[k])
        
        l = l / coef
        for T in range(1000):
            error = 0
            for k in range(K):
                M, a, b, tau = MC_NNM_with_l(O, train_list[k], l, suggest = [pre_M_s[k], pre_a_s[k], pre_b_s[k]])
                pre_M_s[k] = M
                pre_a_s[k] = a
                pre_b_s[k] = b
                error += MSE_validate(M, a, b, valid_list[k])
            if (error >= pre_error and np.linalg.matrix_rank(M) > 2):
                l_opt = l * coef
                break
            pre_error = error
            l = l / coef
    else:
        pre_M_s = []
        pre_a_s = []
        pre_b_s = []
        pre_error = 0
        l_opt = list_l[0]
        for k in range(K):
            M, a, b, tau = MC_NNM_with_l(O, train_list[k], list_l[0])
            pre_M_s.append(M)
            pre_a_s.append(a)
            pre_b_s.append(b)
            pre_error += MSE_validate(M, a, b, valid_list[k])

        for l in list_l[1:]:
            error = 0
            for k in range(K):
                M, a, b, tau = MC_NNM_with_l(O, train_list[k], l, suggest = [pre_M_s[k], pre_a_s[k], pre_b_s[k]])
                pre_M_s[k] = M
                pre_a_s[k] = a
                pre_b_s[k] = b
                error += MSE_validate(M, a, b, valid_list[k])
            if (error < pre_error):
                l_opt = l
                pre_error = error

    M, a, b, tau = MC_NNM_with_l(O, Ω, l_opt)
    return M, a, b, tau