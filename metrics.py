from abc import ABC, abstractmethod
import pandas as pd
import math as m

class Metrics(ABC):
    '''
    Classe astratta per l'implentazione del pattern strategy per usare più metriche
    '''
    @abstractmethod
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        pass
    @staticmethod
    def get_metrics(predictions:pd.DataFrame, truth:pd.DataFrame, ar:bool='False', er:bool='False', sens:bool='False',
                    spec:bool='False', gm:bool='False', all_metrics:bool='True') -> dict:
        results = {}
        if all_metrics:
            calculatorAR = AccuracyRate()
            calculatorER = ErrorRate()
            calculatorSens = Sensitivity()
            calculatorSpec = Specificity()
            calculatorGM = GeometryMean()
            results['Accuracy Rate'] = calculatorAR.calculate_metrics(predictions, truth)
            results['Error Rate'] = calculatorER.calculate_metrics(predictions, truth)
            results['Sensitivity'] = calculatorSens.calculate_metrics(predictions, truth)
            results['Specificity'] = calculatorSpec.calculate_metrics(predictions, truth)
            results['Geometry Mean'] = calculatorGM.calculate_metrics(predictions, truth)
        else:
            if ar:
                calculator=AccuracyRate()
                results['Accuracy Rate']=calculator.calculate_metrics(predictions, truth)
            if er:
                calculator = ErrorRate()
                results['Error Rate'] = calculator.calculate_metrics(predictions, truth)
            if sens:
                calculator = Sensitivity()
                results['Sensitivity'] = calculator.calculate_metrics(predictions, truth)
            if spec:
                calculator = Specificity()
                results['Specificity'] = calculator.calculate_metrics(predictions, truth)
            if gm:
                calculator = GeometryMean()
                results['Geometry Mean'] = calculator.calculate_metrics(predictions, truth)
        return results

class AccuracyRate(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola l'accuracy rate: metrica che indica quanto il modello predice bene
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore dell'accuracy rate
        '''
        #Inizio ad implementare la classe AccuracyRate

        #Confronto gli elementi, colonna per colonna, dei 2 dataframe. Come risultante avrò una serie di valori
        #booleani che mi dicono se i valori coincidono
        valori_coincidenti = (predictions == truth).all(axis=1)

        #Calcolo l'accuracy rate utilizzando una media, essendo tali valori 0/1 , quello che restituirà sarà
        #la percentuale di 'uni' nella serie di valori
        accuracy_rate = valori_coincidenti.mean()
        return accuracy_rate

class ErrorRate(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola l'error rate: metrica che indica quanto il modello predice male (é l'inverso dell'accuracy)
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore dell'error rate
        '''

        # Inizio ad implementare la funzione calculate_metrics per la classe ErrorRate, che è simile alla
        # funzione per il calcolo dell'accuratezza.

        # Confronto gli elementi, colonna per colonna, dei 2 dataframe. Come risultante avrò una serie di valori
        # booleani che mi dicono se i valori non coincidono
        valori_non_coincidenti = (predictions != truth).all(axis=1)

        # Calcolo l'error rate utilizzando una media, essendo tali valori 0/1 , quello che restituirà sarà
        # la percentuale di 'zeri' nella serie di valori
        error_rate = valori_non_coincidenti.mean()

        return error_rate

class Sensitivity(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola la sensitivity: metrica che indica quanto il modello ha predetto bene per la classe 4
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore della sensitivity
        '''

        # Inizio ad implementare la funzione calculate_metrics per la classe Sensitivity

        # Estrae un dataframe con soltanto veritá di classe 4 (tutti i positivi)
        df_truth4 = truth[truth['Class'] == 4]

        #Ottengo gli indici di dove ho le veritá (4 nel nostro caso)
        indici_truth = df_truth4.index

        # Ottengo gli elementi di predizione i cui indici coincidono con le veritá di classe 4
        predizioni_classe_4 = predictions.loc[indici_truth]


        # Confronto gli elementi, riga per riga, dei 2 dataframe. Come risultante avrò una serie di valori
        # booleani che indica se, per ogni campione, la predizione e la verità sono entrambe uguali a 1 (4 nel nostro caso)
        true_positive = (predizioni_classe_4 == df_truth4).all(axis=1)

        #Calcolo la media
        sensitivity_rate = true_positive.mean()

        return sensitivity_rate

class Specificity(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola la specificity: metrica che indica quanto il modello ha predetto bene per la classe 2
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore della specificity
        '''

        # Inizio ad implementare la funzione calculate_metrics per la classe Specificity

        # Estrae un dataframe con soltanto veritá di classe 2 (tutti i negativi)
        df_truth2 = truth[truth['Class'] == 2]

        # Ottengo gli indici del vettore delle veritá le cui veritá sono di classe 2
        indici_truth = df_truth2.index

        # Ottengo gli elementi di predizione i cui indici coincidono con le veritá di classe 2
        predizioni_classe_2 = predictions.loc[indici_truth]

        # Confronto gli elementi, riga per riga, dei 2 dataframe. Come risultante avrò una serie di valori
        # booleani che indica se, per ogni campione, la predizione e la verità sono entrambe uguali a 0 (2 nel nostro caso)
        true_positive = (predizioni_classe_2 != df_truth2).all(axis=1)

        #Calcolo la media
        specificity_rate = true_positive.mean()

        return specificity_rate

class GeometryMean(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola la media geometrica: metrica che permette di capire come il modello sta performando sulle due classi in
                                     contemporanea (4 e 2).
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore della media geometriva
        '''

        # Inizio ad implementare la funzione calculate_metrics per la classe GeometryMean
        tpr_istanza = Sensitivity()
        tnr_istanza = Specificity()
        tpr = tpr_istanza.calculate_metrics(predictions, truth)
        tnr = tnr_istanza.calculate_metrics(predictions, truth)

        g_mean = m.sqrt(tpr * tnr)

        return g_mean
