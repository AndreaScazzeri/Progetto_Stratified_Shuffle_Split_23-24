from abc import ABC,abstractmethod
#ABC serve per far discendere Splitting da una classe astratta
import pandas as pd
class Splitting(ABC):
    @abstractmethod
    def split(self,df:pd.DataFrame,ps:float):
        pass