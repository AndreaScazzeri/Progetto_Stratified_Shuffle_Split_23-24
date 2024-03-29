import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class PreProcessing:
    def gestisci_elementi_vuoti(self, df: pd.DataFrame):
        """
        Elimina le righe che contengono valori nulli
        """
        df = df.dropna(axis=0)
        # Qualora si decidesse di salvare il dataframe in un file csv
        # new_file = 'dataset_senza_elementi_vuoti.csv'
        # df.to_csv(new_file, index=False)
        return df

    def divisione_features(self, df: pd.DataFrame):
        """
        Dato un dataframe restituisce una tupla dove il primo elemento è un dataFrame che contiene tutte le variabili
        indipendenti, mentre il secondo elemento sono le predizioni
        """
        dimensioni = df.shape
        var_ind = df.iloc[:, :dimensioni[1] - 1]
        prev = df.iloc[:, dimensioni[1] - 1:]
        return var_ind, prev



# la classe feature scaling è una classe astratta (strategy pattern)
class FeatureScaling(ABC):
    """
    Classe per l'implementazione del pattern strategy per rendere modulare la tipologia di feature
    scale che l'utente vuole scegliere
    """
    @abstractmethod
    def scale(self,df:pd.DataFrame):
        pass
    # metodo statico della classe astratta per la creazione di oggetti che effettuano lo scale
    @staticmethod
    def create(tipo: str = 'stand'):
        if tipo == 'stand':
            return FeatureScalingStand()
        elif tipo == 'norm':
            return FeatureScalingNorm()

# classe per l'implementazione dello strategy pattern usato per il feature scaling
class FeatureScalingStand(FeatureScaling):
    def scale(self,df:pd.DataFrame):
        """
        Si effettua il feature scaling utilizzando la tecnica di standardizzazione
        :param df: dataFrame del quale si vuole effettuare il la standardizzazione
        :return: media = è la media di ogni series presente nel dataFrame
        deviazione_standard = è la deviazione standard di ogni series presente nel DataFrame
        standardizzazione = è lo stesso dataFrame di partenza sul quale è stata effettuato il feature scaling
        """
        media = np.mean(df, axis=0)
        deviazione_standard = np.std(df, axis=0)
        standardizzazione = ((df - media) / deviazione_standard)
        return standardizzazione

# classe per l'implementazione dello strategy pattern usato per il feature scaling
class FeatureScalingNorm(FeatureScaling):
    def scale(self, df: pd.DataFrame):
        """
        Si effettua il feature scaling utilizzando la tecnica di normalizzazione
        :param df: dataFrame del quale si vuole effettuare il la normalizzazione
        :return: normalizzazione = è lo stesso dataFrame di partenza sul quale è stata effettuato il feature scaling
        """
        df_min = df.min()
        df_max = df.max()
        normalizzazione = (df - df_min) / (df_max - df_min)
        return normalizzazione