import numpy as np
from splitting import Splitting
import pandas as pd
import random


class StratifiedShuffleSubsamplingSplitter(Splitting):
    def split(self, df: pd.DataFrame, parametro_splitting, n_divisioni, seed=1):
        """
        Metodo ereditato dalla classe splitting. Serve per splittare con lo stratified shuffle subsampling il dataframe
        che gli viene passato
        :param df: dataframe da splittare
        :param parametro_splitting: parametro che indica la percentuale di elementi che devono essere nel test set
        :param n_divisioni: numero di volte che deve essere ripetuto lo splitting
        :param seed: parametro che serve a generare sempre gli stessi numeri casuali, utile per la riproducibilitá
        :return: deve restituire una lista lunga n_divisioni dove ogni elemento è una tupla (train set, test set)
        """
        if seed is not None:
            random.seed(seed)
        risultato = []
        for i in range(0, n_divisioni):
            # divido il dataframe in funzione della classe
            df_classe2 = df[df['Class'] == 2]
            df_classe4 = df[df['Class'] == 4]
            # calcolo quale è la numerosita' del test set attesa di ogni classe
            numerosita_attesa_classe2 = int(round(parametro_splitting*len(df_classe2)))
            numerosita_attesa_classe4 = int(round(parametro_splitting*len(df_classe4)))
            # estraggo gli indici delle righe che devono essere nel test set di ogni classe
            indici_classe_2 = random.sample(range(0, len(df_classe2)), numerosita_attesa_classe2)
            indici_classe_4 = random.sample(range(0, len(df_classe4)), numerosita_attesa_classe4)
            # creo i test set con le due classi
            df_classe2_modificato = df_classe2.iloc[indici_classe_2]
            df_classe4_modificato = df_classe4.iloc[indici_classe_4]
            testset = pd.concat([df_classe2_modificato, df_classe4_modificato])
            # creo il train set
            n2 = np.array(range(0, len(df_classe2)))
            n4 = np.array(range(0, len(df_classe4)))
            # seleziono gli indici che non sono stati usati nel test set
            indici_train2 = list(np.setdiff1d(n2, indici_classe_2))
            indici_train4 = list(np.setdiff1d(n4, indici_classe_4))
            # identifico gli elementi da includere nel train set
            df_train_classe2_modificato = df_classe2.iloc[indici_train2]
            df_train_classe4_modificato = df_classe4.iloc[indici_train4]
            trainset = pd.concat([df_train_classe2_modificato, df_train_classe4_modificato])
            risultato.append((trainset, testset))
        return risultato
