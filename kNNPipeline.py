import pandas as pd
from preprocessing import *
from splittingFactory import SplittingFactory
class KNNPipeline:

    def __init__(self, path: str, fs: str = 'stand', splitting_type: str = 'holdout', parametro_splitting: float = 0.2,
                 n_divisioni: int = 5,ar: bool = False, er: bool = False, sens: bool = False, spec: bool = False, gm: bool = False,
                 all_metrics: bool = True):
        """
        COSTRUTTORE DELLA CLASSE KNNPIPELINE
        :param path: percorso del file che contiene il dataset
        :param fs: stringa che specifica con che modo effettuare il feature scaling: 'stand' per standardizzazione,
                    'norm' per normalizzazione (di default 'stand')
        :param splitting_type: stringa che specifica con quale splittig del dataset effettuare l'addestramento del
                    modello. Può essere di 3 tipi: 'holdout' per usare l'holdout; 'sss' per usare lo stratified shuffle
                    subsampling; 'both' per usarle entrambe
        :param parametro_splitting: è un parametro che indica il valore percentuale delle dimensioni di test e train
                    (es. 0.7 se voglio che il rapporto sia 70-30) se si sceglie holdout o il numero di esperimenti da
                    eseguire se si sceglie lo stratified shuffle subsampling
        :param n_divisioni: numero di volte che deve essere ripetuto lo splitting e quindi gli esperimenti con il metodo
                    stratified shuffle split, con l'holdout viene ignorato (di default vale 5)
        :param ar: booleano che specifica se si vuole utilizzare la metrica Accuracy Rate
        :param er: booleano che specifica se si vuole utilizzare la metrica Error Rate
        :param sens: booleano che specifica se si vuole utilizzare la metrica Sensitivity
        :param spec: booleano che specifica se si vuole utilizzare la metrica Specificity
        :param gm: booleano che specifica se si vuole utilizzare la metrica Geometry Mean
        :param all_metrics: booleano che specifica se si vogliono utilizzare tutte le metriche precedenti
        :return:
        """
        self.path = path
        self.fs = fs
        self.splitting_type = splitting_type
        self.parametro_splitting = parametro_splitting
        self.n_divisioni = n_divisioni
        self.ar = ar
        self.er = er
        self.sens = sens
        self.spec = spec
        self.gm = gm
        self.all_metrics = all_metrics
        # esegue la pipeline
        self.doPipeline()


    def doPipeline(self):
        """
        Metodo che esegue tutta la pipeline del kNN
        :return: decidere che cose restituisce
        """
        dataset = pd.read_csv(self.path) # implementare il factory pattern per rendere modulare la lettura dei dati
        preProcess = PreProcessing()

        # elimino le righe che contengono valori nulli
        dataset_corretto = preProcess.gestisci_elementi_vuoti(dataset)
        splitter = SplittingFactory().create(self.splitting_type)
        data_splitted = splitter.split(dataset_corretto, self.parametro_splitting, self.n_divisioni)

        # data_splitted è una lista di coppie di dataframe train e test
        # data_splitted = [(train, test), (train, test), ... , (train, test)]
        # ognuna di queste ora deve essere inserita in una lista [(data train, truth train), (data test, truth test)]
        # per farlo si utilizza la funzione divisione_features di preProcessing
        # esperimenti = [[(data train, truth train), (data test, truth test)], ... , [(data train, truth train), (data test, truth test)]]
        # su ogni elemento data_train e data_test di queste coppie di dataframe deve anche avvenire il feature scaling,
        # quindi creiamo l'oggetto feature_scaler
        feature_scaler = FeatureScaling.create(self.fs)
        esperimenti = []
        for tupla in data_splitted:
            data_train = feature_scaler.scale(preProcess.divisione_features(tupla[0])[0])
            data_test = feature_scaler.scale(preProcess.divisione_features(tupla[1])[0])
            truth_train = preProcess.divisione_features(tupla[0])[1]
            truth_test = preProcess.divisione_features(tupla[1])[1]
            esperimenti.append([(data_train, truth_train),(data_test, truth_test)])
