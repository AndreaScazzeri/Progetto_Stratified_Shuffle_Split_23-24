import pandas as pd
import matplotlib.pyplot as plt
class PlotPerformance:

    def __init__(self, performance: pd.DataFrame):
        self.performance= performance

    def plotBoxplot(self):
        boxplot = plt.boxplot([self.performance['Accuracy Rate'], self.performance['Error Rate'],
                               self.performance['Sensitivity'], self.performance['Specificity'], self.performance['Geometry Mean']],
                              labels=['Accuracy Rate', 'Error Rate', 'Sensitivity', 'Specificity', 'Geometry Mean'])
        plt.title('Performance medie del modello kNN')
        plt.show()

    def plotLineplot(self):
        plt.plot(self.performance['Esperimento'], self.performance['Accuracy Rate'], label='Accuracy Rate', color='red')
        plt.plot(self.performance['Esperimento'], self.performance['Error Rate'], label='Error Rate', color='blue')
        plt.plot(self.performance['Esperimento'], self.performance['Sensitivity'], label='Sensitivity', color='green')
        plt.plot(self.performance['Esperimento'], self.performance['Specificity'], label='Specificity', color='yellow')
        plt.plot(self.performance['Esperimento'], self.performance['Geometry Mean'], label='Geometry Mean', color='black')
        plt.title('Performance del modello kNN nei diversi esperimenti')
        plt.xlabel('Esperimento')
        plt.ylabel('Performance')
        plt.legend()
        plt.show()