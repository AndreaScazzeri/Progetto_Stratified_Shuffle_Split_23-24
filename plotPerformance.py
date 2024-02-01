import pandas as pd
import matplotlib.pyplot as plt
class PlotPerformance:

    """
    La classe PlotPerformance implementa la possibilitá di visualizzare graficamente i risultati ottenuti dallo script
    metrics.py.
    Il parametro usato é il seguente:
    :performance: é un DataFrame che contiene le performance di Accuratezza, Errore, Sensitivity, Specificity e Media
    Geometrica per ogni esperimento.
    """

    def __init__(self, performance: pd.DataFrame):
        self.performance = performance



    def plotBoxplot(self):
        # Il seguente metodo permette la definizione di un boxplot
        plt.figure(figsize=(12, 10))
        plt.boxplot([self.performance['Accuracy Rate'], self.performance['Error Rate'],
                               self.performance['Sensitivity'], self.performance['Specificity'], self.performance['Geometry Mean']],
                              labels=['Accuracy Rate', 'Error Rate', 'Sensitivity', 'Specificity', 'Geometry Mean'])
        plt.title('Performance medie del modello kNN')
        plt.savefig("Plots/kNN_box_plot.png", dpi=500)
        plt.show()

    def plotLineplot(self):
        #Il seguente metodo permette di eseguire un grafico di linea per visualizzare gli andamenti degli esperimenti
        plt.figure(figsize=(12, 10))
        plt.plot(self.performance['Esperimento'], self.performance['Accuracy Rate'], label='Accuracy Rate', color='red')
        plt.plot(self.performance['Esperimento'], self.performance['Error Rate'], label='Error Rate', color='blue')
        plt.plot(self.performance['Esperimento'], self.performance['Sensitivity'], label='Sensitivity', color='green')
        plt.plot(self.performance['Esperimento'], self.performance['Specificity'], label='Specificity', color='yellow')
        plt.plot(self.performance['Esperimento'], self.performance['Geometry Mean'], label='Geometry Mean', color='black')
        plt.title('Performance del modello kNN nei diversi esperimenti')
        plt.xlabel('Esperimento')
        plt.ylabel('Performance')
        plt.legend()
        plt.savefig("Plots/kNN_line_plot.png", dpi=500)
        plt.show()