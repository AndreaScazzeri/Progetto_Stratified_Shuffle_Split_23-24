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
        Calcola l'accuracy rate
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore dell'accuracy rate
        '''
        #Inizio ad implementare la classe AccuracyRate

        #Vedo quali sono gli insiemi degli elementi che sono comuni agli oggetti prediction e truth
        indici_truth = truth.index
        indici_coincidenti = predictions.index.intersection(indici_truth)

        #Estraggo le righe dal dataframe 'predictions' che contengono gli indici specificati,
        #ovvero 'indici_coincidenti' tramite la funzione loc
        predizioni_coincidenti = predictions.loc[indici_coincidenti]

        #Estraggo le righe dal dataframe 'truth' che contengono gli indici specificati,
        #ovvero 'indici_coincidenti' tramite la funzione loc
        verita_coincidenti = truth.loc[indici_coincidenti]

        #Confronto gli elementi, colonna per colonna, dei 2 dataframe. Come risultante avrò una serie di valori
        #booleani che mi dicono se i valori coincidono
        valori_coincidenti = (predizioni_coincidenti == verita_coincidenti).all(axis=1)

        #Calcolo l'accuracy rate utilizzando una media, essendo tali valori 0/1 , quello che restituirà sarà
        #la percentuale di 'uni' nella serie di valori
        accuracy_rate = valori_coincidenti.mean()
        return accuracy_rate

class ErrorRate(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola l'error rate
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore dell'error rate
        '''

        # Inizio ad implementare la funzione calculate_metrics per la classe ErrorRate, che è simile alla
        # funzione per il calcolo dell'accuratezza.

        indici_truth = truth.index
        indici_coincidenti = predictions.index.intersection(indici_truth)
        predizioni_coincidenti = predictions.loc[indici_coincidenti]
        verita_coincidenti = truth.loc[indici_coincidenti]

        # Confronto gli elementi, colonna per colonna, dei 2 dataframe. Come risultante avrò una serie di valori
        # booleani che mi dicono se i valori non coincidono
        valori_non_coincidenti = (predizioni_coincidenti != verita_coincidenti).all(axis=1)

        # Calcolo l'error rate
        error_rate = valori_non_coincidenti.mean()
        return error_rate

class Sensitivity(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola la sensitivity
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore della sensitivity
        '''

        # Inizio ad implementare la funzione calculate_metrics per la classe Sensitivity

        indici_truth = truth.index
        indici_coincidenti = predictions.index.intersection(indici_truth)
        predizioni_coincidenti = predictions.loc[indici_coincidenti]
        verita_coincidenti = truth.loc[indici_coincidenti]

        # Confronto gli elementi, colonna per colonna, dei 2 dataframe. Come risultante avrò una serie di valori
        # booleani che indica se, per ogni campione, la predizione e la verità sono entrambe uguali a 1
        true_positive = ((predizioni_coincidenti == 1) & (verita_coincidenti == 1)).all(axis=1)

        sensitivity_rate = true_positive.mean()
        return sensitivity_rate

class Specificity(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola la specificity
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore della specificity
        '''

        # Inizio ad implementare la funzione calculate_metrics per la classe Specificity

        indici_truth = truth.index
        indici_coincidenti = predictions.index.intersection(indici_truth)
        predizioni_coincidenti = predictions.loc[indici_coincidenti]
        verita_coincidenti = truth.loc[indici_coincidenti]

        # Confronto gli elementi, colonna per colonna, dei 2 dataframe. Come risultante avrò una serie di valori
        # booleani che indica se, per ogni campione, la predizione e la verità sono entrambe uguali a 0 (False)
        true_positive = ((predizioni_coincidenti == 0) & (verita_coincidenti == 0)).all(axis=1)

        specificity_rate = true_positive.mean()
        return specificity_rate
class GeometryMean(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola la media geometrica
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore della media geometriva
        '''

        # Inizio ad implementare la funzione calculate_metrics per la classe GeometryMean
        tpr_istanza = Sensitivity()
        tnr_istanza = Specificity()
        tpr = tpr_istanza.calculate_metrics(predictions, truth)
        tnr = tnr_istanza.calculate_metrics(predictions, truth)

        g_mean = m.sqrt(tpr + tnr)

        return g_mean
