def vba(df):
    min_val  = list(np.repeat(np.nan, df.shape[1])) 
    med_val  = list(np.repeat(np.nan, df.shape[1])) 
    max_val  = list(np.repeat(np.nan, df.shape[1]))
    mean_val = list(np.repeat(np.nan, df.shape[1])) 
    std_val  = list(np.repeat(np.nan, df.shape[1]))
    for i in range(df.shape[1]):
        if pd.api.types.is_numeric_dtype(df.iloc[:,i]):
            d = df.iloc[:,i].describe()
            min_val[i] = d[3]
            med_val[i]  = d[5]
            max_val[i]  = d[7]
            mean_val[i] = d[1]
            std_val[i]  = d[2]

    flag_numeric = [pd.api.types.is_numeric_dtype(df.iloc[:,i])*1 for i in range(df.shape[1])]
    mcv    = [list(df.iloc[:,i].value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False))[0] for i in range(df.shape[1])]
    lcv    = [list(df.iloc[:,i].value_counts(normalize=True, sort=True, ascending=True, bins=None, dropna=False))[0] for i in range(df.shape[1])]
    df_vba = pd.DataFrame({'Variable':df.columns, 
                           'flag_numeric': flag_numeric,
                           'NaN_count':df.isnull().sum(), 
                           'Unique_count':df.nunique(dropna=False), 
                           'NaN_frac':df.isnull().sum()/len(df), 
                           'mcv_frac':mcv, 
                           'lcv_frac':lcv,
                           'min_val':min_val,
                           'med_val':med_val,
                           'max_val':max_val,
                           'mean_val':mean_val,
                           'std_val':std_val},
                           columns = ['Variable', 'flag_numeric','NaN_count', 'Unique_count', 'NaN_frac', 'mcv_frac', 
                                      'lcv_frac', 'min_val', 'med_val', 'max_val', 'mean_val', 'std_val'])


    print(*df_vba, sep='\n')
    df_vba.to_csv('vba.csv', sep=';', index = False, decimal = ',')
    return df_vba