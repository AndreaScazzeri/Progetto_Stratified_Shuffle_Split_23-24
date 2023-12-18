import pandas as pd

def gestisci_elementi_vuoti(df):
    pass

def feature_scaling(df):
    pass

def divisione_features(df:pd.DataFrame):
    dimensioni = df.shape
    var_ind = df.iloc[:, :dimensioni[1] - 1]
    prev = df.iloc[:, dimensioni[1] - 1:]
    return var_ind, prev