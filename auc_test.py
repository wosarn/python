import numpy as np

def auc(score, target):
    """ test funkcji - liczenie AUC 
    Argumenty:
    score - score (wyzsza wartosc sprzyja targetowi "1")
    target - target binarny (0/1)
    
    Wynik
    auc_coef - wspolczynnk AUC
    """

    score        = np.array(score)
    target       = np.array(target)
    n1           = sum(target)
    n0           = sum(1-target)
    unique_score = np.unique(score)
    cdf1         = np.array([sum((s >= score)*(target == 1)) for s in unique_score])/float(n1)
    cdf0         = np.array([sum((s >= score)*(target == 0)) for s in unique_score])/float(n0)   
    
    auc_coef     = sum(0.5*(cdf1[:-1]+cdf1[1:])*(cdf0[1:]-cdf0[:-1]))
    return(auc_coef)
    
    


