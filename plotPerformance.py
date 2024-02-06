import pandas as pd
import matplotlib.pyplot as plt
import os
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
        plt.boxplot([self.performance['Accuracy Rate'], self.performance['Sensitivity'], self.performance['Specificity'],
                     self.performance['Geometry Mean']], labels=['Accuracy Rate', 'Sensitivity', 'Specificity', 'Geometry Mean'])
        plt.title('Performance medie del modello kNN')
        if os.path.exists('Plots'):
            plt.savefig("Plots/kNN_box_plot.png", dpi=500)
        else:
            os.mkdir('Plots')
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
        if os.path.exists('Plots'):
            plt.savefig("Plots/kNN_line_plot.png", dpi=500)
        else:
            os.mkdir('Plots')
            plt.savefig("Plots/kNN_line_plot.png", dpi=500)
        plt.show()

    def plotTable(self):
        # Il seguente metodo permette di visualizzare una tabella con le performance di ogni esperimento del modello kNN
        fig, ax = plt.subplots()
        ax.axis('off')
        # pd.plotting.table richiede in ingresso un oggetto di tipo Axes e un DataFrame che contiene le informazioni da visualizzare
        # quindi bisogna creare prima un oggetto di tipo Axes e lo si fa con plt.subplots()
        pd.plotting.table(ax, self.performance, loc='center', cellLoc='center')
        if os.path.exists('Plots'):
            plt.savefig("Plots/kNN_table_plot.png", dpi=500)
        else:
            os.mkdir('Plots')
            plt.savefig("Plots/kNN_table_plot.png", dpi=500)
        plt.show()
