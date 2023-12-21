from splitting import Splitting
import pandas as pd
class HoldoutSplitter(Splitting):
    def split(self, df: pd.DataFrame, rapporto:int):
        '''
        Metodo ereditato dalla classe splitting. Serve per splittare con l'holdout il dataframe che gli viene passato
        :return: restituisce una tupla di dataframe. Il primo è il dataframe del trainset il secondo è il dataframe del testset
        '''
        pass