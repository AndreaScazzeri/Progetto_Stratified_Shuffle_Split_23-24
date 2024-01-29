from splitting import Splitting
import pandas as pd
import random
class StratifiedShuffleSubsamplingSplitter(Splitting):
    def split(self, df: pd.DataFrame, grandezza_test=0.2, numero_casuale=None, n_divisioni:int=10):
        '''
        Metodo ereditato dalla classe splitting. Serve per splittare con lo stratified shuffle subsampling il dataframe che gli viene passato
        :param df: dataframe da splittare
        :param n_divisioni: numero di volte che deve essere ripetuto lo splitting
        :return: deve restituire n_divisioni di coppie train set - test set
        '''
        if numero_casuale is not None:
            random.seed(numero_casuale)
        risultato=[]
        for i in range(1,n_divisioni):
            print(df)
            df_classe2=df[df['Class']==2]
            df_classe4=df[df['Class']==4]
            numerosita_attesa_classe2=grandezza_test*len(df_classe2)
            numerosita_attesa_classe4=grandezza_test*len(df_classe4)
            indici_classe_2=random.sample(range(1,len(df_classe2)),numerosita_attesa_classe2)
            indici_classe_4 = random.sample(range(1, len(df_classe4)), numerosita_attesa_classe4)
            df_classe2_modificato=df_classe2.iloc[indici_classe_2]
            df_classe4_modificato=df_classe4.iloc[indici_classe_4]
            risultato.append((df_classe2_modificato,df_classe4_modificato))
        return risultato




