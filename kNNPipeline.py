import pandas as pd
from preprocessing import *
class KNNPipeline:

    def __init__(self, path:str, fs:str='stand', splitting_type: str='holdout', ar:bool=False, er:bool=False,
                 sens:bool=False, spec:bool=False, gm:bool=False, all_metrics:bool=True):
        '''COSTRUTTORE DELLA CLASSE KNNPIPELINE

        :param path: percorso del file che contiene il dataset
        :param fs: stringa che specifica con che modo effettuare il feature scaling: 'stand' per standardizzazione, 'norm' per normalizzazione (di default 'stand')
        :param splitting_type: stringa che specifica con quale splittig del dataset effettuare l'addestramento del modello. Può essere di 3 tipi: 'holdout' per usare l'holdout; 'sss' per usare lo stratified shuffle subsampling; 'both' per usarle entrambe
        :param ar: booleano che specifica se si vuole utilizzare la metrica Accuracy Rate
        :param er: booleano che specifica se si vuole utilizzare la metrica Error Rate
        :param sens: booleano che specifica se si vuole utilizzare la metrica Sensitivity
        :param spec: booleano che specifica se si vuole utilizzare la metrica Specificity
        :param gm: booleano che specifica se si vuole utilizzare la metrica Geometry Mean
        :param all_metrics: booleano che specifica se si vogliono utilizzare tutte le metriche precedenti
        :return:
        '''
        self.path = path
        self.fs = fs
        self.splitting_type = splitting_type
        self.ar = ar
        self.er = er
        self.sens = sens
        self.spec = spec
        self.gm = gm
        self.all_metrics = all_metrics
        #esegue la pipeline
        self.doPipeline()

    def doPipeline(self):
        '''
        Metodo che esegue tutta la pipeline del kNN
        :return: decidere che cose restituisce
        '''
        dataset = pd.read_csv(self.path)
        preProcess = PreProcessing()
        #aggiungere la gestione degli elementi vuoti dopo la sua implementazione
        #dataset_corretto = preProcess.gestisci_elementi_vuoti(dataset)
        ds = preProcess.divisione_features(dataset)
        data = ds[0]
        truth = ds[1]
        if self.fs == 'stand':
            feature_scaler = FeatureScaling.create('stand')
            data = feature_scaler.scale(data)
        elif self.fs == 'norm':
            feature_scaler = FeatureScaling.create('norm')
            data = feature_scaler.scale(data)
        #continuare la pipeline