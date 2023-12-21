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
    def get_metrics(predictions:pd.DataFrame, truth:pd.DataFrame, ar:bool='False', er:bool='False', sens:bool='False', spec:bool='False', gm:bool='False', all_metrics:bool='True'):
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


class AccuracyRate(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        pass
class ErrorRate(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        pass
class Sensitivity(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        pass
class Specificity(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        pass
class GeometryMean(Metrics):
    def calculate_metrics(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        pass