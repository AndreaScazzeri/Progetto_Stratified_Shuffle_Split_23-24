import pandas as pd
import numpy as np
from tqdm import tqdm
class KNN:
    '''
    Classe KNN che deve implementare l'addestramento del k-Nearest Neighbor con tutte le funzioni necessarie
    ATTENZIONE: non ci sono particolari pattern da utilizzare per questa classe, quindi chi la implementerà sceglie le funzioni
    da definire e come farlo
    '''

    #la funzione doTrain deve solamente calcolare la distanza euclidea di tutti i punti nel train da uno dei punti che stanno nel test
    def __init__(self,esperimento: list, k: int, esperimento_index: int):
        """
        Costruttore della classe kNN:
        :param esperimento: è una lista che contiene due elementi, il primo è una tupla che contiene le informazioni del
                            train set, il secondo, anch'esso una tupla che contiene le informazioni del test set.
        :param k: è il parametro che mi indica quanti valori più vicini al punto devo prendere
        :param esperimento_index: è l'indice dell'esperimento che sto eseguendo, in altre parole il numero di esperimento
        """
        self.k = k
        self.x_train = esperimento[0][0].iloc[:, 1:]
        self.y_train = esperimento[0][1]
        self.x_test = esperimento[1][0].iloc[:, 1:]
        self.y_test = esperimento[1][1]
        self.esperimento_index = esperimento_index

    def calculate_distances(self, row_test):
        """
        Questa funzione calcola la distanza tra un elemento del test set e tutti gli elementi del train set restituendo
        tali distanze rispettivamente agli indici degli elementi del train test
        """
        distances = []
        for index_train, row_train in self.x_train.iterrows():
            dist = np.linalg.norm(row_train - row_test)
            distances.append([index_train,dist])
        return np.array(distances)

    def doPrediction(self):
        """
        Questa funzione esegue le predizioni su ogni elemento del test e restituisce la predizione rispetto alla classe
        più numerosa nei k elementi più vicini
        """
        #copio le predizioni solo per avere la struttura del dataframe che ci servirà per calcolare le metriche
        predictions = self.y_test.copy()
        # passo alla funzione tqdm l'iteratore che è il dataframe del test set, questa mi permette di visualizzare la barra di progresso durante l'esecuzione delle predizioni
        for index_test, row_test in tqdm(self.x_test.iterrows(), total=len(self.x_test), desc='Progresso predizioni esperimento '+str(self.esperimento_index+1), ncols=100):
            distances = self.calculate_distances(row_test)
            dist_ordinate = distances[distances[:,1].argsort()]
            nearest_neighbors_indices = dist_ordinate[:self.k,0].astype(int)
            nearest_labels = []
            for indice in nearest_neighbors_indices:
                nearest_labels.append(self.y_train.loc[indice])
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            # aggiungo la predizione al dataframe delle predizioni che ha la stessa struttura del dataframe delle verità
            predictions.loc[index_test] = predicted_label
        return predictions