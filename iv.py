# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 21:36:12 2018

@author: Wojtek
"""


def iv(data_frame, ind_target, col_list = [] ):
    """
    IV calculations
    data_frame - pandas data frame
    ind_target - target position in the data frame
    col_list   - list of columns position in the data frame for which IV will be calculated. If ignored - all columns will be taken into account
    """ 
    import pandas as pd
    import numpy as np    
    
    target     = data_frame.iloc[:,ind_target]
    if len(col_list) == 0:
        col_list = range(data_frame.shape[1])

    data_frame = data_frame.iloc[:,col_list]   
    iv_summary = pd.DataFrame({'Variable':data_frame.columns, 'IV':np.repeat(None, len(data_frame.columns))}) [['Variable', 'IV']]
    for i in range(data_frame.shape[1]):
        prob_class       = data_frame.iloc[:,i].value_counts(normalize = True).sort_index()
        prob_class0      = data_frame.iloc[:,i].loc[target == 0].value_counts(normalize = True).sort_index()
        prob_class1      = data_frame.iloc[:,i].loc[target == 1].value_counts(normalize = True).sort_index()
        if len(prob_class) == len(prob_class0) & len(prob_class) == len(prob_class1):
            iv_summary.IV[i] = sum((prob_class1 - prob_class0)*np.log(prob_class1/prob_class0))
    return iv_summary     

