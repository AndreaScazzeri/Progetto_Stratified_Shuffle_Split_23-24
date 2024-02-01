from splitting import Splitting
import pandas as pd
import random
class HoldoutSplitter(Splitting):
    def split(self, df: pd.DataFrame, ps, numero_divisioni, seed):
        '''
        Metodo ereditato dalla classe splitting. Serve per splittare con l'holdout il dataframe che gli viene passato
        :return: restituisce una tupla di dataframe. Il primo è il dataframe del trainset il secondo è il dataframe del testset

        Tale Funzione si preoccupa di Dividere o 'Splittare' il dataset in TESTSET e TRAININGSET utilizzando la tecnica dell'holdout

        Parametri:
        1) df: il dataframe da splittare deve essere una lista di liste, dove ogni lista rappresenta una riga del dataset
        2) grandezza_test_set: Si é deciso che la proporzione del dataset da utilizzare per il testset sia settata di default a 0.2 (il 20%),
                               tale valore é specificato nel file kNNPipeline.py
        3) seed: parametro che serve a generare dei numeri casuali, utile per la riproducibilitá

        Restituisce:
        1) train_set: ovvero il set di addestramento
        2) test_set: ovvero il set di test.
        '''
        #Inizio ad implementare la funzione split della classe HoldoutSplitter
        if seed is not None:
            random.seed(seed)

        #Calcolo la grandezza del test set
        grandezza_test_set = int(len(df) * ps)

        # Calcola l'indice di divisione tra test set e train set
        indice_di_divisione_random = random.randint(0, len(df))

        # Dividiamo ora il dataset in test set e train set

        #Creo il test set estraendo il 20% dei dati in modo ordinato e il restante 80% lo metto nel train set
        if indice_di_divisione_random > len(df)-grandezza_test_set:
            test_set = pd.concat(df.iloc[indice_di_divisione_random:] , df.iloc[:grandezza_test_set - (len(df)-indice_di_divisione_random)])
            train_set = df.iloc[grandezza_test_set - (len(df)-indice_di_divisione_random) +1: indice_di_divisione_random]

        else:
            test_set = df.iloc[indice_di_divisione_random:indice_di_divisione_random+grandezza_test_set]
            train_set = pd.concat([df.iloc[:indice_di_divisione_random], df.iloc[indice_di_divisione_random + grandezza_test_set:]])


        return [(train_set, test_set)]





