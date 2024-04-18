
import yfinance as yf
from datetime import datetime

import numpy as np


# todo: inherit from factory
class APIHandler:
    pass

class YahooFinance(APIHandler):

    __allowed_types = ['Adj Close'] 
    
    @staticmethod
    def load(
        symbols: list,
        starttime: datetime,
        endtime: datetime,
        # todo: find all possible keywords
        value_type: str = 'Adj Close'
    ):
        
        yf.pdr_override()

        # dictionary of charts and their names
        charts = {s: yf.download(s, starttime, endtime)[value_type].to_numpy()
                for s in symbols}
        
        # clean and scale the data
        for key in charts.keys():

            # there are NaN entries in the array, we have to remove these
            charts[key] = charts[key][np.logical_not(np.isnan(charts[key]))]

        # todo: return type dataframe instead of list of numpy series
        return charts