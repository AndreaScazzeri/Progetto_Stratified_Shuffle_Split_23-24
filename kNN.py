
import numpy as np

class KNN:
    '''
    Classe KNN che deve implementare l'addestramento del k-Nearest Neighbor con tutte le funzioni necessarie
    ATTENZIONE: non ci sono particolari pattern da utilizzare per questa classe, quindi chi la implementerà sceglie le funzioni
    da definire e come farlo
    '''

    #la funzione doTrain deve solamente calcolare la distanza euclidea di tutti i punti nel train da uno dei punti che stanno nel test
    def __init__(self,esperimento: list, k=3):
        self.k = k
        self.x_train = esperimento[0][0]
        self.y_train = esperimento[0][1]
        self.x_test = esperimento[1][0]
        self.y_test = esperimento[1][1]

    def doTrain(self, data_train):
        self.data_train = data_train

    def calculate_distances(self, x):
        distances = np.linalg.norm(self.data_train - x, axis=1)
        return distances

    def doPrediction(self):
        predictions = []
        for Data_test in data_test:
            distances = self.calculate_distances(Data_test)
            nearest_neighbors_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.data_train[nearest_neighbors_indices]
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)
        return np.array(predictions)