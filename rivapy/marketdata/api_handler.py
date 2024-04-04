
import yfinance as yf
from datetime import datetime

import numpy as np


# todo: inherit from factory
class APIHandler:
    pass

class YahooFinance(APIHandler):

    __allowed_types = ['Adj_Close'] 
    
    def load(
        ticker: dict,
        starttime: datetime,
        endtime: datetime,
        # todo: find all possible keywords
        value_type: str = 'Adj_Close'
    ):
        
        yf.pdr_override()

        # dictionary of charts and their names
        charts = {k: yf.download(v, starttime, endtime)[value_type].to_numpy()
                for k, v in ticker.items()}

        # clean and scale the data
        for key in charts.keys():

            # there are NaN entries in the array, we have to remove these
            charts[key] = charts[key][np.logical_not(np.isnan(charts[key]))]

        return charts