from splitting import Splitting
import pandas as pd
class StratifiedShuffleSubsamplingSplitter(Splitting):
    def split(self, df: pd.DataFrame, n_divisioni:int):
        '''
        Metodo ereditato dalla classe splitting. Serve per splittare con lo stratified shuffle subsampling il dataframe che gli viene passato
        :param df: dataframe da splittare
        :param n_divisioni: numero di volte che deve essere ripetuto lo splitting
        :return: deve restituire n_divisioni di coppie train set - test set
        '''
        pass