import pandas as pd
from preprocessing import *
from splittingFactory import SplittingFactory
class KNNPipeline:

    def __init__(self, path:str, fs:str='stand', splitting_type: str='holdout', rapporto:int=70, n_esperimenti:int=5,ar:bool=False, er:bool=False,
                 sens:bool=False, spec:bool=False, gm:bool=False, all_metrics:bool=True):
        '''COSTRUTTORE DELLA CLASSE KNNPIPELINE

        :param path: percorso del file che contiene il dataset
        :param fs: stringa che specifica con che modo effettuare il feature scaling: 'stand' per standardizzazione, 'norm' per normalizzazione (di default 'stand')
        :param splitting_type: stringa che specifica con quale splittig del dataset effettuare l'addestramento del modello. Può essere di 3 tipi: 'holdout' per usare l'holdout; 'sss' per usare lo stratified shuffle subsampling; 'both' per usarle entrambe
        :param rapporto: specifica il rapporto che deve esserci tra le dimensioni di test e train nel caso di scelta dell'holdout
        :param n_esperimenti: specifica il numero di volte con cui si deve ripetere lo splitting sss
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
        dataset = pd.read_csv(self.path) #implementare il factory pattern per rendere modulare la lettura dei dati
        preProcess = PreProcessing()
        #aggiungere la gestione degli elementi vuoti dopo la sua implementazione
        dataset_corretto = preProcess.gestisci_elementi_vuoti(dataset)
        ds = preProcess.divisione_features(dataset_corretto)
        data = ds[0]
        truth = ds[1]
        if self.fs == 'stand':
            feature_scaler = FeatureScaling.create('stand')
            data = feature_scaler.scale(data)
        elif self.fs == 'norm':
            feature_scaler = FeatureScaling.create('norm')
            data = feature_scaler.scale(data)
        splitter = SplittingFactory().create(self.splitting_type)
        if type(splitter)=='tuple':
            data_splitted_1 = splitter[0].split(data)
            data_splitted_2 = splitter[0].split(data)
        else:
            data_splitted = splitter.split(data)
        #continuare la pipeline considerando l'eventualità di aver utilizzato entrambi i metodi di splitting