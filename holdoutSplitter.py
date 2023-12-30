from splitting import Splitting
import pandas as pd
import random
import pandas as pd
class HoldoutSplitter(Splitting):
    def split(self, df: pd.DataFrame, grandezza_test=0.2, numero_casuale=None):
        '''
        Metodo ereditato dalla classe splitting. Serve per splittare con l'holdout il dataframe che gli viene passato
        :return: restituisce una tupla di dataframe. Il primo è il dataframe del trainset il secondo è il dataframe del testset

        Tale Funzione si preoccupa di Dividere o 'Splittare' il dataset in TESTSET e TRAININGSET utilizzando la tecnica dell'holdout

        Parametri:
        1) df: il dataframe da splittare deve essere una lista di liste, dove ogni lista rappresenta una riga del dataset
        2) grandezza_test: Si é deciso che la proporzione del dataset da utilizzare per il testset sia settata di default a 0.2 (il 20%)
        3) numeri_casuali: parametro che serve a generare dei numeri casuali, utile per la riproducibilitá

        Restituisce:
        1) train_set: ovvero il set di addestramento
        2) test_set: ovvero il set di test.
        '''
        #Inizio ad implementare la funzione split della classe HoldoutSplitter

        #Questo costrutto mi é utile per la riproducibilitá, ovvero, se il 'seme' é lo stesso in diverse esecuzioni del programma, le sequenze di numeri casuali generate
        #saranno uguali, questo mi permetterá di ottenere gli stessi risultati.
        if numero_casuale is not None:
            random.seed(numero_casuale)


        #Calcolo la dimensione del testset
        grandezza_test = int(len(df) * grandezza_test)

        #Estraggo in modo casuale le righe per il testset. Se il dataframe ha un indice personalizzato utilizzo iloc per estrarre le righe
        indici_test_set = random.sample(range(len(df)), grandezza_test)
        test_set = df.iloc[indici_test_set]

        #creo il trainingset escludendo le righe utilizzate per il testset
        train_set = df.drop(index=indici_test_set)

        return  train_set, test_set





