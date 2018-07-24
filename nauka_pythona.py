# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:54:16 2018

@author: Wojtek
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 09:42:20 2018

@author: Wojtek
"""

#import pyodbc 
#server = 'LAPTOP-FLVDOFJN\SQLEXPRESS' 
#database = 'DPR' 
#username = 'myusername' 
#password = 'mypassword' 
#cnxn = pyodbc.connect('DRIVER={SQL Server Native Client 11.0};SERVER='+server+';DATABASE='+database+')  #;UID='+username+';PWD='+ password)
#cursor = cnxn.cursor()

import pandas as pd
import numpy as np
import pyodbc 
from patsy import dmatrices
conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=LAPTOP-FLVDOFJN\SQLEXPRESS;"
                      "Database=DPR;"
                      "Trusted_Connection=yes;")

sql = """
SELECT 
                                [umo].[umowa]
                                ,[umo].[podumo]
                                ,[umo].[oferta]
                                ,[umo].[FFINRPFH_czy]
                                ,[umo].[KLi_NIP]
                                ,[umo].[KLi_NAZWASKR]
                                ,[umo].[KLi_NAZWA]
                                ,[umo].[PORECZYCIEL_CZY]
                                ,[umo].[PORECZYCIEL_NIP]
                                ,[umo].[TR_DTZAWC]
                                ,[umo].[TR_DTZAWC_MC]
                                ,[umo].[TR_DTZAWC_Q]
                                ,[umo].[TR_DTZAWC_RokOkres]
                                ,[umo].[TR_DTZAWC_H]
                                ,[umo].[TR_DTZAWC_Y]
                                ,[umo].[TR_D30M3]
                                ,[umo].[TR_D60M3]
                                ,[umo].[TR_D60M6]
                                ,[umo].[TR_D60M12]
                                ,[umo].[TR_D60M24]
                                ,[umo].[TR_D60M36]
                                ,[umo].[TR_D90M12]
                                ,[umo].[TR_D90M24]
                                ,[umo].[TR_D90M36]
                                ,[umo].[TR_STATUS_UMOWY_BIEZ]
                                ,[umo].[TR_POZIOM_wykonanie]
                                ,[umo].[TR_POZIOM_estymator]
                                ,[umo].[TR_ANEKS_dataZawarciaUmowyAneksowanej]
                                ,[umo].[TR_ANEKS_dataZawarciaAneksu]
                                ,[umo].[TR_ANEKS_dataPierwszejZmianyKlienta]
                                ,[umo].[TR_ANEKS_RODZAJ_id]
                                ,[umo].[TR_HANDLOWIEC_SIEC]
                                ,[umo].[TR_DOSTAWCA_NIP]
                                ,[umo].[TR_DOSTAWCA_KATEGORIA]
                                ,[umo].[TR_DOSTAWCA_KLASYFIKACJA]
                                ,[umo].[TR_DOSTAWCA_LSD]
                                ,[umo].[TR_DOSTAWCA_LSD_MiU]
                                ,[umo].[TR_DOSTAWCA_LSD_TRUCK]
                                ,[umo].[TR_DOSTAWCA_LSD_OSD]
                                ,[umo].[TR_TYP_LEASINGU]
                                ,[umo].[TR_TYP_UMOWY]
                                ,[umo].[TR_WARIANT]
                                ,[umo].[TR_PROCEDURA]
                                ,[umo].[TR_HARMONOGRAM]
                                ,[umo].[TR_HARMONOGRAM_czy_odrocz]
                                ,[umo].[TR_LEAS_ZWROTNY_CZY]
                                ,[umo].[TR_OKRES]
                                ,[umo].[TR_WO]
                                ,[umo].[TR_IRR]
                                ,[umo].[TR_WARTOSC_WYKUPU]
                                ,[umo].[TR_WARTOSC_WYKUPU_UDZIAL_CI]
                                ,[umo].[TR_WARTOSC_WYKUPU_UDZIAL_RAT]
                                ,[umo].[TR_WARTOSC_WYKUPU_UDZIAL_KON]
                                ,[umo].[TR_CI]
                                ,[umo].[TR_CI_PROC]
                                ,[umo].[TR_PD_2013]
                                ,[umo].[TR_EL_2013]
                                ,[umo].[KL_KAT_ZAW_2013]
                                ,[umo].[KL_SCORING_2013]
                                ,[umo].[PRZEDM_asset]
                                ,[umo].[PRZEDM_asset_DPR]
                                ,[umo].[PRZEDM_OPIS_PRZEDMIOTU]
                                ,[umo].[PRZEDM_OPIS_PRZEDMIOTU_GRUPA]
                                ,[umo].[PRZEDM_OPIS_PRZEDMIOTU_GRUPA2]
                                ,[umo].[PRZEDM_ROKPROD]
                                ,[umo].[PRZEDM_WIEK_URUCH]
                                ,[umo].[PRZEDM_WIEK_KONIEC_UM]
                                ,[umo].[PRZEDM_KLASA]
                                ,[umo].[PRZEDM_LUX]
                                ,[umo].[PRZEDM_PREMIUM]
                                ,[umo].[PRZEDM_MARKA]
                                ,[umo].[PRZEDM_MARKA_ID]
                                ,[umo].[PRZEDM_MODEL]
                                ,[umo].[PRZEDM_SEGMENT]
                                ,[umo].[PRZEDM_STAWKA_AMORTYZACJI]
                                ,[umo].[KL_KAT_ZAW_GRUPA]
                                ,[umo].[KL_ZRODLO_DANYCH]
                                ,[umo].[KL_PKD2007_MAP]
                                ,[umo].[KLi_PKD2007_MAP]
                                ,[umo].[KL_DZIALALNOSC_OPIS]
                                ,[umo].[KL_DZIALALNOSC]
                                ,[umo].[KL_nTSL]
                                ,[umo].[KL_nBUD]
                                ,[umo].[KL_nBUD_nTSL]
                                ,[umo].[KL_nWZP]
                                ,[umo].[KL_BRANZA_OPIS]
                                ,[umo].[KL_BRANZA_OPIS_GRUPA]
                                ,[umo].[KL_BRANZA_PREF_ZAW]
                                ,[umo].[KL_BRANZA_PREF_BIEZ]
                                ,[umo].[KL_TYP_DZIALALNOSCI]
                                ,[umo].[KL_KLASA_PRZEWOZNIKA]
                                ,[umo].[KL_DATAROZPDZIAL]
                                ,[umo].[KL_CZASPROWDZIALGOSP_L]
                                ,[umo].[KL_FORMA_DZIAL]
                                ,[umo].[KL_STAN_CYWILNY]
                                ,[umo].[KL_LICZBA_PRAC]
                                ,[umo].[KL_WIEK_PESELDEC]
                                ,[umo].[KL_DOCHOD_RB]
                                ,[umo].[KL_DOCHOD_RU]
                                ,[umo].[KL_DOCHOD_MC]
                                ,[umo].[KL_DOCHOD_12M]
                                ,[umo].[KL_OBROT_RB]
                                ,[umo].[KL_OBROT_RU]
                                ,[umo].[KL_OBROT_12M]
                                ,[umo].[KL_OBROT_MAX]
                                ,[umo].[KL_OBROT_MC]
                                ,[umo].[KL_ZYSK_RU]
                                ,[umo].[KL_LICZBA_MIES_RU]
                                ,[umo].[KL_LICZBA_MIES_RB]
                                ,[umo].[KL_FIN_RODZAJ_KSIEGOWOSCI_RU_id]
                                ,[umo].[KL_FIN_RODZAJ_KSIEGOWOSCI_RB_id]
                                ,[umo].[KL_FIN_TYP_DOK_RU_id]
                                ,[umo].[KL_FIN_TYP_DOK_RB_id]
                                ,[umo].[KL_FIN_RYCZALT_czy]
                                ,[umo].[KL_RENT_RU]
                                ,[umo].[KL_RENT_RB]
                                ,[umo].[KL_RENT_MC]
                                ,[umo].[KL_DYNAM_DOCHOD]
                                ,[umo].[KL_DYNAM_OBROT]
                                ,[wyk].[TR_Default_FirstDate30]
                                ,[wyk].[TR_Default_FirstDate60]
                                ,[wyk].[TR_Default_FirstDate90]
                                ,[wyk].[TR_FRAUD_DataStatusu]
                              FROM 
                                [DPR].[dbo].[mk_dane_umow] [umo]
                                inner join
                                [DPR].[dbo].[mk_SCORE_umowy_wykonanie] [wyk] on [umo].[umowa] = [wyk].[umowa]
                              WHERE 
                                [TR_DTZAWC_RokOkres] >= 201101
                                and [umo].[FFINRPFH_czy] = 1
"""
#df = pandas.io.sql.read_sql(sql, conn)
df = pd.read_sql(sql, conn)
conn.close()

#------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------- Ładowanie i zapis danych -------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------#
import pyodbc
conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=LAPTOP-FLVDOFJN\SQLEXPRESS;"
                      "Database=DPR;"
                      "Trusted_Connection=yes;")

sql = """
            SELECT 
                *
            FROM 
                [DPR].[dbo].[mk_dane_umow] [umo]
"""
df = pd.read_sql(sql, conn)
conn.close()

nba = pd.read_csv("nba_2013.csv")  #ładowanie z csv
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';') #ładowanie csv z http


# zapis danych do plików
df_cut = df.iloc[np.arange(0,101),np.arange(0,11)]
df_cut.to_csv('E:\\Wojtek\\_DSCN_\\Narzedzia\\Python\\testy\\leasing_subset.csv', sep=';', index = False)
df_cut.to_csv('E:\\Wojtek\\_DSCN_\\Narzedzia\\Python\\testy\\leasing_subset.txt', sep=';', index = False)
df_cut.to_excel(excel_writer = 'E:\\Wojtek\\_DSCN_\\Narzedzia\\Python\\testy\\leasing_subset.xlsx', index = False)




#------------------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------- Tworzenie ramki danych -------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------#

#--- Tworzenie ramki

dates = pd.date_range('20130101', periods=6)
dataf = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))  #-- utworzenie ramki z macierzy numpy 6x4 z kolumnami A, B, C, D
dataf2 = pd.DataFrame(np.transpose([1,2,3,4,5]), np.transpose([6,7,8,9,0]), columns = ['Var1', 'Var2'])   # to nie działa!!!!!
dataf2 = pd.DataFrame({'Var1':[1,2,3,4,5], 'Var2':[6,7,8,9,0]}) #-- utworzenie ramki z osobnych kolumn zczepionych jako słownik

dataf['Var'] = [1,2,3,4,5,6]  #dodanie nowej kolumny do istniejącej ramki

dff = pd.DataFrame({'Variable': dataf.columns, 'Type':dataf.dtypes}).iloc[:,[1,0]]
x = [1,2,3,4,5]
df_test = pd.DataFrame(x, columns=['x']) #tu są problemy!!!

d1 = pd.DataFrame({'Var1':[1,2,3,4,5]})
d2 = pd.DataFrame({'Var2':[6,7,8,9,0]})
 
d3 = pd.concat([d1, d2], axis=1) #odpowiednik cbind
d4 = pd.concat([d1, d2], axis=0, ignore_index=True) #odpowiednik rbind - nie działa!
d4 = d1.append(d2) #odpowiednik rbind - nie działa!
#------------------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------- Print ---------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------#

print(*df.columns, sep='\n') # w ten sposób wywietla się wszystko (w podstawowej opcji "srodek" jest pomijany)


#------------------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ Stringi ---------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------#

["s" + str(i) for i in np.arange(1,11)]  #odpowiednik paste("s", 1:10, sep = "")
path_var = 'E:\\Wojtek\\_DSCN_\\Analiza_danych\\Leas\\data_set\\'
path_var+'variables.csv'







#------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------- Podstawowy ogląd danych -------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------#

type(df)   #sprawdzenie czy df jest data frame
df.shape  #odpowiednik dim
len(df)  #odpowiednik nrow(df)
len(df.columns)  #liczba kolumn
df.shape[1] #liczba kolumn
df.head(5) #top 5
df.columns  #odpowiednik names(df)
df.index  # odpowiednik rownames(df)
df.describe()  #odpowiednik summary(df)
df.info(verbose = True, max_cols = True, null_counts = True)  #odpowiednik str(df)
df.dtypes  #typy kolumn w ramce danych
df.select_dtypes( exclude = ['object']) #zwraca ramkę tylko z kolumnami nie-stringami (wyłączenie stringów)
df.select_dtypes( include = [np.number]) # zawarcie tylko numerycznych

pd.DataFrame([1,2,3.2])
pd.api.types.is_numeric_dtype(pd.DataFrame([1,2,3.2]))

is_type(pd.DataFrame([1,2,3.2]), np.number)

pd.api.types.is_numeric_dtype([1,2,3.2])

a=df_filtered.select_dtypes( include = [np.number])
a.shape


#------------------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- Obróbka ramek danych -------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------#

#---- table, is.na
pd.isnull(df.TR_TYP_LEASINGU)  #sprawdzenie missing values czyli R-owych NA dla kolumny 'TR_TYP_LEASINGU'
df.TR_TYP_LEASINGU.isnull().value_counts()
df.TR_TYP_LEASINGU.isnull().sum()
df.TR_TYP_LEASINGU.describe()
df.iloc[:,0].describe()   #wygląda na to, że jesli chcemy odwołać się do kolumny poprzez jej numer, to trzeba użyć metody iloc: df.iloc[:,3] - czwarta kolumna ramki df!
df.iloc[:,0]
df.loc[:, ['KL_DYNAM_DOCHOD', 'KL_DYNAM_OBROT']].describe()  # a to przykład wyciągania kolumn po nazwach!

df_test = df.drop(['umowa'], axis = 1)  #usuwanie zmiennej
df_test = df.iloc[:,[10,14,27]]
df_test.columns
df.TR_TYP_LEASINGU.value_counts()  #odpowiednik table dla pandas!
df.TR_TYP_LEASINGU.value_counts(normalize=True, sort=True, ascending=False, bins=None, dropna=False) #argument normalize = True oznacza, że mamy rozkład - odpowiednik prop.table
pd.isnull(df.TR_TYP_LEASINGU).value_counts() 
df.TR_TYP_LEASINGU.nunique()  #odpowiednik length(unique(x))
np.sort(df.TR_TYP_LEASINGU.unique()) #odpowiednik unique(x)

#--- ifelse
x = np.where(pd.isnull(df.loc[:,'KL_DYNAM_DOCHOD'] ), -999, df.loc[:,'KL_DYNAM_DOCHOD'] )  #odpowiednik ifelse
x = np.where(pd.isnull(df.loc[:,'KL_DYNAM_DOCHOD'] ), -999, np.where(df.loc[:,'KL_DYNAM_DOCHOD'] < 0, -1, 1)) #odpowiednik ifelse
df['KL_DYNAM_DOCHOD_bin'] = np.where(pd.isnull(df.loc[:,'KL_DYNAM_DOCHOD'] ), -999, np.where(df.loc[:,'KL_DYNAM_DOCHOD'] < 0, -1, 1))  #odpowiednik ifelse + dodanie nowej kolumny1)


df.TR_TYP_LEASINGU.hist()
df.boxplot(column='KL_DYNAM_DOCHOD')
df.groupby(df.TR_TYP_LEASINGU).mean()


df.T  #transpozycja ramki danych



#---- filtrowanie wierszy
target_name = 'TR_D90M12'
df.loc[:,target_name].describe()
df1 = df[ (df.FFINRPFH_czy == 1) 
            & (df.TR_ANEKS_RODZAJ_id  == 0) 
            & (df.TR_POZIOM_wykonanie == 1)  
            & (df.PORECZYCIEL_CZY == 0)
            & (pd.isnull(df.TR_FRAUD_DataStatusu)) 
            & (~ pd.isnull(df.loc[:,target_name]))]
df1.shape

#-- podzbiór wierszy i kolumn
col = df.columns.isin(predictors)+df.columns.isin([target_name])  # to raczej nie jest eleganckie rozwiązanie
target_name = 'TR_D90M12'
df_filtered = df.loc[ (df.FFINRPFH_czy == 1) 
            & (df.TR_ANEKS_RODZAJ_id  == 0) 
            & (df.TR_POZIOM_wykonanie == 1)  
            & (df.PORECZYCIEL_CZY == 0)
            & (pd.isnull(df.TR_FRAUD_DataStatusu)) 
            & (~ pd.isnull(df.loc[:,target_name])), col]
df_filtered.shape


#poniżej alternatywa dla filtrowania, ale nie idzie, bo są jakies problemy z 'pd'
df1 = df.query( '(FFINRPFH_czy == 1) & (TR_ANEKS_RODZAJ_id  == 0) & (TR_POZIOM_wykonanie == 1) & (PORECZYCIEL_CZY == 0) & (pd.isnull(TR_FRAUD_DataStatusu)) & (~pd.isnull(df.loc[:,target_name]))')
df1.shape
#sprawdzanie braków danych w ramce df1:





#----- podzbiory wierszy i kolumn w ramce
col_list = data_frame.columns.get_loc(['FFINRPFH_czy', 'TR_ANEKS_RODZAJ_id']) # to nie działa, chyba tylko jedną kolumnę można odpytać
col_list = [data_frame.columns.get_loc(col) for col in ['FFINRPFH_czy', 'TR_ANEKS_RODZAJ_id']] #ale można tak
data_frame.iloc[:,col_list] #zwróci podzbiór kolumn
data_frame.iloc[:, data_frame.columns.isin(['FFINRPFH_czy', 'TR_ANEKS_RODZAJ_id'])] #to też zwróci ramkę z dwiema kolumnami
data_frame.loc[:, data_frame.columns.isin(['FFINRPFH_czy', 'TR_ANEKS_RODZAJ_id'])] #to też zwróci ramkę z dwiema kolumnami
ind_target = data_frame.columns.get_loc(target_name)


#'KL_STAN_CYWILNY','PRZEDM_asset_DPR'


#----- prop.table
pd.crosstab(df_filtered.KL_STAN_CYWILNY, df_filtered.PRZEDM_asset_DPR) #licznosci
pd.crosstab(df_filtered.KL_STAN_CYWILNY, df_filtered.PRZEDM_asset_DPR, normalize = True, dropna = False) #rozkład dwuwymiarowy
pd.crosstab(df_filtered.KL_STAN_CYWILNY, df_filtered.PRZEDM_asset_DPR, normalize = 'columns', dropna = False) #rozkład każdej kolumny względem wierszy
pd.crosstab(df_filtered.KL_STAN_CYWILNY, df_filtered.PRZEDM_asset_DPR, normalize = 'index', dropna = False) #rozkład każdego wiersza względem kolumn



#------- Binowanie zmiennych ciągłych + kwantyle
n=5
#q = df.KL_RENT_RU.quantile(q = [0, 0.25, 0.5, 0.75, 1])#.tolist()
q = df.KL_RENT_RU.quantile(q = np.linspace(0, 10, n+1)/10)#.tolist()
b = pd.cut(df.KL_RENT_RU, bins = q, include_lowest = True)
#b = pd.cut(df.KL_RENT_RU, bins = q, include_lowest = True, labels = ['A', 'B', 'C', 'D']) #zamiast przedziałów etykietki w postaci wielkich liter A-D
b.value_counts(dropna=False).sort_index()
b = b.cat.add_categories(['blank'])
b.cat.categories
b = b.fillna('blank')
b.cat.categories
b.value_counts(dropna=False).sort_index()

#https://stackoverflow.com/questions/47053770/pandas-cut-how-to-convert-nans-or-to-convert-the-output-to-non-categorical
#http://benalexkeen.com/bucketing-continuous-variables-in-pandas/



#---- Korelacja
# Pearson correlation
iris.corr()

# Kendall Tau correlation
iris.corr('kendall')

# Spearman Rank correlation
iris.corr('spearman')
###Note that the two last correlation measures require you to rank the data before calculating the coefficients. You can easily do this with rank(): for the exercise above, the iris data was ranked by executing iris.rank() for you.


#----- standaryacja
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)

rescaledX = scaler.transform(X)


#------------------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ Missing values --------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------#

target_name = 'TR_D90M12'
df_filtered = df[ (df.FFINRPFH_czy == 1) 
            & (df.TR_ANEKS_RODZAJ_id  == 0) 
            & (df.TR_POZIOM_wykonanie == 1)  
            & (df.PORECZYCIEL_CZY == 0)
            & (pd.isnull(df.TR_FRAUD_DataStatusu)) 
            & (~ pd.isnull(df.loc[:,target_name]))]
df_filtered.shape


df_filtered.isnull().sum() #liczba NA dla każdej zmiennej w ramce danych

df_filtered._get_numeric_data()

nan_summary = pd.DataFrame({'Variable':df_filtered.columns, 'NaN_count':df_filtered.isnull().sum()})  #.iloc[:,[1,0]]  #zwróci ramkę w której znajdą się dwie kolumny: nazwa zmiennej i liczba NA

nan_summary = pd.DataFrame({'Variable':df_filtered.columns, 'NaN_count':df_filtered.isnull().sum(), 'Unique_count':df_filtered.nunique(dropna=False), 'NaN_percent':df_filtered.isnull().sum()/len(df_filtered) })



df_filtered.dropna(axis = 0, how = 'any')  #usuń z ramki danych wiersze w których choć jedna kolumna zawiera NA
df_filtered.dropna(axis = 1, how = 'all')  #usuń z ramki danych kolumny w których wszystkie wiersze zawierają NA
df_filtered.dropna(axis = 0, threshold = 10)  #zachowaj ramce danych tylko te wiersze w których choć 10 kolumn zawiera wartosć nie NA
df_filtered.shape
df_filtered.describe()


#------------------- Missing values: imputacja 
df_filtered.loc[:, ['TR_PD_2013', 'TR_EL_2013']].fillna(-1) #uzupełnij wartoscią -1, uwaga - nie nadpisuje kolumn w ramce, trzeba zrobic podstawienie
for i in data_frame.columns:
    data_frame.loc[:,i] = data_frame.loc[:,i].fillna(mean(data_frame.loc[:,i])) #jesli kolumna ma typ numeryczny
    data_frame.loc[:,i] = data_frame.loc[:,i].fillna('blank')  #jesli kolumna ma typ tekstowy
    #co jesli ma typ categorical


df2.describe()
predictors_categ = var_df.variable[ (var_df.type_var == 'pred') &  (var_df.type_pred == 'd')]
col_categ = df2.columns.isin(predictors_categ)
df2.iloc[:, col_categ].fillna('blank')   #dla zmiennych kategorycznych
df2.KL_FIN_TYP_DOK_RB_id.value_counts()
df2.KL_FIN_TYP_DOK_RB_id.isnull().sum()

df['var1'] = df['var1'].fillna(df['var1'].mean())  #dla zmiennych ciągłych, ale pojedynczych


df_filtered2 = df_filtered.loc[:,['KL_DOCHOD_RB','KL_DOCHOD_RU','KL_DOCHOD_MC','KL_DOCHOD_12M','KL_OBROT_RB','KL_OBROT_RU','KL_OBROT_12M','KL_OBROT_MAX','KL_OBROT_MC']]

from sklearn.preprocessing import Imputer 
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0) # parametryzacja imputera
mean_imputer = mean_imputer.fit(df_filtered2)
# Apply imputation
df_new = mean_imputer.transform(df_filtered2.values)
df_new = pd.DataFrame(df_new, columns = df_filtered2.columns)
df_new.isnull().sum()


#------------------- Budowa modelu regresji logistycznej


path_var = 'E:\\Wojtek\\_DSCN_\\Analiza_danych\\Leas\\data_set\\'
var_df = pd.read_csv(path_var+'variables.csv', sep=';')
var_df.columns()
var_df.type_var.value_counts()
var_df.head(10)
var_df.type_pred.value_counts(dropna=False)


#predictors = var_df.variable[ (var_df.type_var == 'pred') | (var_df.variable == target_name) ]
predictors = var_df.variable[ var_df.type_var == 'pred' ]
ind_target = data_frame.columns.get_loc(target_name)
type(predictors)
#type(list)(predictors)
pd.match(df.columns, predictors) #zwraca pozycję wystąpienia: R-owy odpowiednik which( vect1 %in% vec2 )


col = df.columns.isin(predictors)+df.columns.isin([target_name])  # to raczej nie jest eleganckie rozwiązanie

target_name = 'TR_D90M12'
df_filtered = df.loc[ (df.FFINRPFH_czy == 1) 
            & (df.TR_ANEKS_RODZAJ_id  == 0) 
            & (df.TR_POZIOM_wykonanie == 1)  
            & (df.PORECZYCIEL_CZY == 0)
            & (pd.isnull(df.TR_FRAUD_DataStatusu)) 
            & (~ pd.isnull(df.loc[:,target_name])), col]
df_filtered.shape

df_filtered2 = df_filtered.dropna(axis = 0, how = 'any')
df_filtered2.shape

#df_filtered._get_numeric_data()

var_in_model = ['KL_CZASPROWDZIALGOSP_L','TR_CI_PROC', 'PRZEDM_WIEK_KONIEC_UM','KL_DYNAM_OBROT','TR_OKRES','KL_WIEK_PESELDEC','KL_RENT_RB']

df_filtered2_dumm = pd.get_dummies(df_filtered2.loc[:,['KL_BRANZA_OPIS', 'KL_FORMA_DZIAL']], prefix='Categ', columns=['KL_BRANZA_OPIS', 'KL_FORMA_DZIAL'], drop_first = True, dummy_na = False) # drop_first - wyrzucamy pierwszy poziom; w macierzy jest n-1 wartoci
df_filtered2_dumm = pd.get_dummies(df_filtered2.loc[:,['PRZEDM_asset_DPR']], prefix='Categ', columns=['PRZEDM_asset_DPR'], drop_first = True, dummy_na = False) # drop_first - wyrzucamy pierwszy poziom; w macierzy jest n-1 wartoci

df_filtered2_dumm.shape
df_filtered3 = pd.concat([df_filtered2.loc[:,df_filtered2.columns.isin(var_in_model)], df_filtered2_dumm], axis=1)


y = df_filtered2.loc[:, target_name]

from sklearn.model_selection import train_test_split
df_filtered3_train, df_filtered3_test, y_train, y_test = train_test_split(df_filtered3, y, test_size = 0.4, random_state=111)


train = df_filtered3.sample(frac=0.8, random_state=1) #  inny sposób na podział próby na TRN i TST
test = df_filtered3.loc[~df_filtered3.index.isin(train.index)]


#Fit Logit model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

logReg = LogisticRegression().fit(df_filtered3_train, y_train)

y_pred = logReg.predict(df_filtered3_test)
logReg.fit().summary()
import statsmodels.api as sm
logit_model=sm.Logit(y_train,df_filtered3_train)
result=logit_model.fit()
print(result.summary())

#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
auc(false_positive_rate, true_positive_rate)

output = pd.DataFrame(y_pred = y_pred, y_test = y_test)
output = pd.concat([df1.iloc[:,0], df1.iloc[:,1]]), axis=1, keys=['df1', 'df2'])

#---- Zapis modelu
from sklearn.externals import joblib
filename = 'E:\\Wojtek\\_DSCN_\\Analiza_danych\\Leas\\data_set\\logRegTest.sav'
joblib.dump(logReg, filename)

import pickle
filename2 = 'E:\\Wojtek\\_DSCN_\\Analiza_danych\\Leas\\data_set\\logRegTest2.sav'
pickle.dump(logReg, open(filename2, 'wb'))

#---- Ładowanie modelu
filename = 'E:\\Wojtek\\_DSCN_\\Analiza_danych\\Leas\\data_set\\logRegTest.sav'
loaded_model = joblib.load(filename)
result = loaded_model.score(df3_test, y_test)
print(result)


#----- Zapis danych do csv
df.to_csv(file_name, sep='\t')


 logit = sm.Logit(y_train, X_train)
result = logit.fit()

#Summary of Logistic regression model
result.summary()
result.params


df3_train.isnull().sum()

logit = sm.Logit(df3_train.iloc[:,0], df3_train.iloc[:,])
result = logit.fit()

#Summary of Logistic regression model
result.summary()
result.params


#---------------

def get_var_category(series):
    unique_count = series.nunique(dropna=False)
    total_count = len(series)
    if pd.api.types.is_numeric_dtype(series):
        return 'Numerical'
    elif pd.api.types.is_datetime64_dtype(series):
        return 'Date'
    elif unique_count==total_count:
        return 'Text (Unique)'
    else:
        return 'Categorical'

def print_categories(df):
    for column_name in df.columns:
        print(column_name, ": ", get_var_category(df[column_name]))



#------------------
        
import pandas_profiling 

pandas_profiling.ProfileReport(beers_and_breweries)

#----------------
import pandas as pd
import datetime
import numpy as np

df = pd.DataFrame({'date': [datetime.datetime(2010,1,1)+datetime.timedelta(days=i*15) 
for i in range(0,100)]})

This works ..

df['year'] = [d.year for d in df['date']]

This also works ..

df['year'] = df['date'].apply(lambda x: x.year)

But this does not ..

df['year'] = df['date'].apply(year)

Nor does this ..

df['year'] = df['date'].year



#-------------------- WYKRESY 

# wzięte z https://www.datacamp.com/community/tutorials/deep-learning-python

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)

ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
#ax[0].legend(loc='best')
#ax[1].legend(loc='best')
fig.suptitle("Distribution of Alcohol in % Vol")
plt.show()


import numpy as np
print(np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
print(np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15]))



import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(red['quality'], red["sulphates"], color="red")
ax[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Quality")
ax[0].set_ylabel("Sulphates")
ax[1].set_ylabel("Sulphates")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([0,2.5])
ax[1].set_ylim([0,2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Wine Quality by Amount of Sulphates")

plt.show()  



import seaborn as sns
corr = wines.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.show()



#gęstosć
import seaborn as sns
sns.set(color_codes=True)
sns.set_palette(sns.color_palette("muted"))

sns.distplot(beers["ibu"].dropna());