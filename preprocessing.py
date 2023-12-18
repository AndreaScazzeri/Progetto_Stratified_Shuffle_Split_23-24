import pandas as pd
import numpy as np

df = pd.read_csv('breast_cancer.csv')

def gestisci_elementi_vuoti(df):
    pass

def feature_scaling(df: pd.DataFrame, tipo = 'stand'):

    if tipo == 'stand':
      #Si effettua il feature scaling utilizzando la tecnica di standardizzazione
      media = np.mean(df, axis=0)
      deviazione_standard = np.std(df, axis=0)
      standardizzazione = ((df - media)/deviazione_standard)
      return media, deviazione_standard, standardizzazione
    else:
        df_min = df.min()
        df_max = df.max()
        normalizzazione = (df - df_min)/(df_max - df_min)
        return normalizzazione

if __name__ == '__main__':

    tipo_di_feature_scaling = input('Inserisci stand o norm in base al tipo di feature scaling che si vuole eseguire: ')
    if tipo_di_feature_scaling == 'stand' or tipo_di_feature_scaling == 'norm':
        risultato_feature_scaling = feature_scaling(df, tipo = tipo_di_feature_scaling)
        print(f'Il feature scaling utilizzando il metodo di {tipo_di_feature_scaling} è: ')
        print(' ')
        print(risultato_feature_scaling)


def divisione_features(df):
    pass