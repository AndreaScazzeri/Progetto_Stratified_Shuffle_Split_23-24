import pandas as pd
from preprocessing import *
from splittingFactory import SplittingFactory
from kNN import KNN
from metrics import *
from plotPerformance import *
class KNNPipeline:

    def __init__(self, path: str, fs: str = 'stand', splitting_type: str = 'holdout', parametro_splitting: float = 0.2,
                 n_divisioni: int = 5, k: int = 7, ar: bool = False, er: bool = False, sens: bool = False, spec: bool = False, gm: bool = False,
                 all_metrics: bool = True, seed: int = None, show_boxplot: bool = False, show_lineplot: bool = False, show_table: bool = True):
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
        :param k: numero di vicini da considerare per la classificazione (di default vale 7)
        :param ar: booleano che specifica se si vuole utilizzare la metrica Accuracy Rate
        :param er: booleano che specifica se si vuole utilizzare la metrica Error Rate
        :param sens: booleano che specifica se si vuole utilizzare la metrica Sensitivity
        :param spec: booleano che specifica se si vuole utilizzare la metrica Specificity
        :param gm: booleano che specifica se si vuole utilizzare la metrica Geometry Mean
        :param all_metrics: booleano che specifica se si vogliono utilizzare tutte le metriche precedenti
        :param seed: seme per il pescaggio random, di train_set e test_set, utile per la riproducibilitá
        :param show_boxplot: booleano che specifica se si vuole visualizzare il boxplot delle performance
        :param show_lineplot: booleano che specifica se si vuole visualizzare il lineplot delle performance
        :param show_table: booleano che specifica se si vuole visualizzare la tabella copn i valori delle performance per ogni esperimento
        :return:
        """
        self.path = path
        self.fs = fs
        self.splitting_type = splitting_type
        self.parametro_splitting = parametro_splitting
        self.n_divisioni = n_divisioni
        self.k = k
        self.ar = ar
        self.er = er
        self.sens = sens
        self.spec = spec
        self.gm = gm
        self.all_metrics = all_metrics
        self.seed = seed
        self.show_boxplot = show_boxplot
        self.show_lineplot = show_lineplot
        self.show_table = show_table
        # esegue la pipeline
        self.dataset, self.performance = self.doPipeline()


    def doPipeline(self):
        """
        Metodo che esegue tutta la pipeline del kNN
        :return: decidere che cose restituisce
        """
        dataset = pd.read_csv(self.path) # implementare il factory pattern per rendere modulare la lettura dei dati
        preProcess = PreProcessing()

        # elimino le righe che contengono valori nulli
        dataset_corretto = preProcess.gestisci_elementi_vuoti(dataset)
        feature_scaler = FeatureScaling.create(self.fs)
        # eseguo il feature scaling solo sulle variabili indipendenti
        dataset_scalato = feature_scaler.scale(dataset_corretto.iloc[:,:-1])
        # creo un nuovo dataset che contiene le variabili indipendenti scalate e la variabile dipendente
        dataset_finale = pd.concat([dataset_scalato, dataset_corretto.iloc[:,-1]], axis=1)
        splitter = SplittingFactory().create(self.splitting_type)
        # a seconda del tipo di splitting scelto eseguo lo splitting del dataset. Se sono stati specificati i parametri
        # 'holdout' o 'sss' questo for loop viene eseguito una sola volta. Altrimenti se sono stati scelti entrambi viene eseguito due volte:
        # una per l'holdout e una per lo stratified shuffle subsampling (che a sua volta ripete lo splitting n_divisioni volte)
        for splitter in splitter:
            data_splitted = splitter.split(dataset_finale, self.parametro_splitting, self.n_divisioni, self.seed)

            # data_splitted è una lista di coppie di dataframe train e test
            # data_splitted = [(train, test), (train, test), ... , (train, test)]
            # ognuna di queste ora deve essere inserita in una lista [(data train, truth train), (data test, truth test)]
            # per farlo si utilizza la funzione divisione_features di preProcessing
            # esperimenti = [[(data train, truth train), (data test, truth test)], ... , [(data train, truth train), (data test, truth test)]]
            # su ogni elemento data_train e data_test di queste coppie di dataframe deve anche avvenire il feature scaling,
            # quindi creiamo l'oggetto feature_scaler

            esperimenti = []
            for tupla in data_splitted:
                data_train = preProcess.divisione_features(tupla[0])[0]
                data_test = preProcess.divisione_features(tupla[1])[0]
                truth_train = preProcess.divisione_features(tupla[0])[1]
                truth_test = preProcess.divisione_features(tupla[1])[1]
                esperimenti.append([(data_train, truth_train),(data_test, truth_test)])

            # creo un dataframe che conterrà le performance di ogni esperimento
            performance = pd.DataFrame({'Esperimento': [], 'Accuracy Rate': [], 'Error Rate': [], 'Sensitivity': [], 'Specificity': [], 'Geometry Mean': []})
            i = 0
            for esperimento in esperimenti:
                # Creo un oggetto kNN che prende in input un esperimento
                kNN = KNN(esperimento, self.k, i)
                # Eseguo le predizioni per il kNN e le salvo in un dataframe con la stessa struttura delle verità
                predizioni = kNN.doPrediction()
                # Calcolo le metriche
                perf = Metrics.get_metrics(predizioni, esperimento[1][1], self.ar, self.er, self.sens, self.spec, self.gm, self.all_metrics)
                # aggiungo le informazioni dell'esperimento dal dataframe che contiene le performance
                perf['Esperimento'] = len(performance)+1
                perf = pd.DataFrame(perf, index=[0])
                # concateno le performance con gli esperimenti precedenti
                performance = pd.concat([performance, perf], ignore_index=True)
                performance['Esperimento'] = performance['Esperimento'].astype(int)
                i += 1

            #creo l'oggetto plotter per visualizzare i risultati
            plotter = PlotPerformance(performance)
            if self.show_table:
                plotter.plotTable()
            if performance.shape[0] > 1:
                if self.show_lineplot:
                    plotter.plotLineplot()
                if self.show_boxplot:
                    plotter.plotBoxplot()

        return dataset_corretto, performance

    def predict(self, object_to_pred: pd.DataFrame):
        """
        Metodo che predice la classe di un oggetto
        :param object_to_pred: oggetto da predire deve essere un dataframe con la stessa struttura del dataset su cui è
                stato fatto il training. Può anche essere un insieme di oggetti
        :return: 'predizione' è la classe predetta ed è un dataframe che contiene la predizione
        """
        # controllo subito se il dataframe da predire ha la stessa struttura del dataframe su cui è stato fatto il training
        if (object_to_pred.columns != self.dataset.columns).any():
            raise ValueError('Il dataframe da predire deve avere la stessa struttura del dataframe su cui è stato fatto il training')

        # ho bisogno di rifare il feature scaling con l'oggetto da predire. Quindi devo creare un oggetto feature_scaler
        feature_scaler = FeatureScaling.create(self.fs)
        # il feature scaling lo devo fare su tutto il dataset di train non scalato al quale vengono aggiunti i nuovi oggetti da predire
        dataset_scalato = feature_scaler.scale(pd.concat([self.dataset.iloc[:,:-1], object_to_pred.iloc[:,:-1]], axis=0, ignore_index=True))

        #l'oggetto KNN deve prendere in ingresso un esperimento, che è stato pensato con la struttura di una lista contente
        #due tuple, la prima con il train set (dati e verità) e la seconda con il test set (dati e verità).
        # In questo caso il test set è composto dal solo elemento da predire e la verità è ovviamente assente, ma mi aspetto che
        # il dataframe da predire ha già la colonna con la verità, magari con valori NaN o negativi (a disrezione dell'utente)

        # Il primo elemento della tupla corrispondente al train set è la porzione di dataset di train scalato corrispondente alle righe
        # di train, il secondo elemento è la verità corrispondente a queste righe che non viene scalata quindi la prendo dal dataset originale

        # Il secondo elemento della tupla corrispondente al test set è la porzione di dataset di train scalato corrispondente alle righe
        # da predire, il secondo elemento è la predizione che dovrò fare che ancora una volta prendo dal dataset originale
        esperimento=[(dataset_scalato.iloc[:self.dataset.shape[0],:], self.dataset.iloc[:,-1]),
                     (dataset_scalato.iloc[self.dataset.shape[0]:,:], object_to_pred.iloc[:,[-1]])]
        kNN=KNN(esperimento, self.k)
        # eseguo la predizione
        predizione = kNN.doPrediction()
        return predizione