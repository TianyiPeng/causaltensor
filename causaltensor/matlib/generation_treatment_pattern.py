import numpy as np

def generate_Z(pattern_tuple = ['adaptive'], M0 = 0):
    '''
        generate the binary matrix Z for different patterns 
    '''
    while (True):
        if (pattern_tuple[0] == 'adaptive'):
            a = pattern_tuple[1][0]
            b = pattern_tuple[1][1]
            Z = adpative_treatment_pattern(a, b, M0)
    
        if (pattern_tuple[0] == 'iid'):
            p_treat = np.random.rand()*0.5
            Z = np.random.rand(n1, n2) <= p_treat

        if (pattern_tuple[0] == 'block'):
            m2 = pattern_tuple[1][1]
            Z, treat_units = simultaneous_adoption(pattern_tuple[1][0], m2, M0)
        
        if (pattern_tuple[0] == 'stagger'):
            m2 = pattern_tuple[1][1]
            Z = stagger_adoption(pattern_tuple[1][0], m2, M0)

        ## if some row or some column is all treated; or Z=0; generate Z again  
        if (np.sum(np.sum(1-Z, axis=0) == 0) > 0 or np.sum(np.sum(1-Z, axis=1) == 0) > 0 or np.sum(Z)==0): 
            if (pattern_tuple[0] == 'adaptive'):
                return Z, 'fail'
            continue
        break
    if (pattern_tuple[0] == 'block'):
        return Z, treat_units
    if (pattern_tuple[0] == 'adaptive'):
        return Z, 'success'
    return Z

def adpative_treatment_pattern(lowest_T, lasting_T, M):
    '''

        Input: lowest_T, lasting_T, M

        For each row i, if M(i,j) is the smallest among M(i, j-lowest_T:j) and no treatments on (i, j-lowest_T:j), then start the treatment on M(i,j+1) to M(i,j+lasting_T+1)
    '''

    Z = np.zeros_like(M)
    for i in range(Z.shape[0]):
        j = 0
        #print(i)
        while j < Z.shape[1]:
            flag = 0
            for k in range(1, lowest_T+1):
                if (j-k < 0 or Z[i, j-k]==1 or M[i,j] > M[i,j-k]):
                    flag = 1
                    break

            #print(i, j)
            if (flag == 0):
                for k in range(1, lasting_T+1):
                    if (j+k < Z.shape[1]):
                        Z[i, j+k] = 1
                j += lasting_T + lowest_T
            else:
                j = j + 1
    return Z

def iid_treatment(prob=1, shape=(1,1)):
    """
        Generate treatment pattern Z by i.i.d Bernulli random variabls Bern(prob)
        
        Parameters:
        
        prob: Bern(prob)
    """
    return np.random.rand(shape[0], shape[1]) <= prob

def block_treatment_testone(m1, m2, M):
    Z = np.zeros_like(M)
    Z[:m1, m2:] = 1
    return Z

def block_treatment_testtwo(M):
    Z = np.zeros_like(M)
    ratio = np.random.rand()*0.8
    m1 = int(M.shape[0]*ratio)+1
    m2 = int(M.shape[1]*(1-ratio))-1
    Z[:m1, m2:] = 1
    return Z

def simultaneous_adoption(m1, m2, M):
    '''
        randomly select m1 units, adopt the treatment in [m2:]
    '''
    Z = np.zeros_like(M)
    treat_units = np.random.choice(range(M.shape[0]), m1, replace=False)
    Z[treat_units, m2:] = 1
    return Z, treat_units

def stagger_adoption(m1, m2, M):
    '''
        randomly select m1 units, adopt the treatment in after m2: randomly
    '''
    Z = np.zeros_like(M)
    treat_units = np.random.choice(range(M.shape[0]), m1, replace=False)
    for i in treat_units:
        j = np.random.randint(m2, high=M.shape[1])
        Z[i, j:] = 1 
    #Z[treat_units, m2:] = 1
    return Z

if (__name__ == 'main'):
    M = np.zeros((5, 5))
    print(simultaneous_adoption(2, 2, M))
