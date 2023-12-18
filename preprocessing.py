import pandas as pd

class PreProcessing():
    def gestisci_elementi_vuoti(df):
        pass

    def feature_scaling(df: pd.DataFrame, tipo='stand'):

        if tipo == 'stand':
            # Si effettua il feature scaling utilizzando la tecnica di standardizzazione
            media = np.mean(df, axis=0)
            deviazione_standard = np.std(df, axis=0)
            standardizzazione = ((df - media) / deviazione_standard)
            return media, deviazione_standard, standardizzazione
        else:
            df_min = df.min()
            df_max = df.max()
            normalizzazione = (df - df_min) / (df_max - df_min)
            return normalizzazione
    def divisione_features(df:pd.DataFrame):
        dimensioni = df.shape
        var_ind = df.iloc[:, :dimensioni[1] - 1]
        prev = df.iloc[:, dimensioni[1] - 1:]
        return var_ind, prev