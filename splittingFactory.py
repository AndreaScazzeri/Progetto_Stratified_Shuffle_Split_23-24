from holdoutSplitter import HoldoutSplitter
from stratifiedShuffleSubsamplingSplitter import StratifiedShuffleSubsamplingSplitter

class SplittingFactory:
    def create(self, splitting_type:str):
        if splitting_type=='holdout':
            splitter=HoldoutSplitter()
        elif splitting_type=='sss':
            splitter=StratifiedShuffleSubsamplingSplitter()
        elif splitting_type=='both':
            splitter1=HoldoutSplitter()
            splitter2 = StratifiedShuffleSubsamplingSplitter()
            splitter=(splitter1,splitter2)
        else:
            raise RuntimeError('unknown file extension')

        return splitter