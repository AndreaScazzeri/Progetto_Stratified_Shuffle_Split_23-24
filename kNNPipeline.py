import pandas as pd
from preprocessing import *
from splittingFactory import SplittingFactory
class KNNPipeline:

    def __init__(self, path:str, fs:str='stand', splitting_type: str='holdout', parametro_splitting:int=5,ar:bool=False, er:bool=False,
                 sens:bool=False, spec:bool=False, gm:bool=False, all_metrics:bool=True):
        '''COSTRUTTORE DELLA CLASSE KNNPIPELINE

        :param path: percorso del file che contiene il dataset
        :param fs: stringa che specifica con che modo effettuare il feature scaling: 'stand' per standardizzazione, 'norm' per normalizzazione (di default 'stand')
        :param splitting_type: stringa che specifica con quale splittig del dataset effettuare l'addestramento del modello. Può essere di 3 tipi: 'holdout' per usare l'holdout; 'sss' per usare lo stratified shuffle subsampling; 'both' per usarle entrambe
        :param parametro_splitting: è un parametro che indica o il valore percentuale delle dimensioni di test e train (es. 70 se voglio che il rapporto sia 70-30) se si sceglie holdout o il numero di esperimenti da eseguire se si sceglie lo stratified shuffle subsampling
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
        self.parametro_splitting = parametro_splitting
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
        # ds = preProcess.divisione_features(dataset_corretto)
        # data = ds[0]
        # truth = ds[1]
        feature_scaler = FeatureScaling.create(self.fs)
        data = feature_scaler.scale(dataset_corretto)
        splitter = SplittingFactory().create(self.splitting_type)
        # data_splitted = splitter.split(data, n_divisioni=self.parametro_splitting)
        # print(type(data_splitted))
        #NB data splitted può essere una o più coppie di dataframe train e test