from abc import ABC, abstractmethod
import pandas as pd

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
        indici_coincidenti = predictions.index.intersection(truth.index)

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
        pass
class Sensitivity(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola la sensitivity
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore della sensitivity
        '''
        pass
class Specificity(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola la specificity
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore della specificity
        '''
        pass
class GeometryMean(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        '''
        Calcola la media geometrica
        :param predictions: dataframe che contiene le predizioni effettuate
        :param truth: dataframe che contiene i risultati veri
        :return: resituisce il valore della media geometriva
        '''
        pass