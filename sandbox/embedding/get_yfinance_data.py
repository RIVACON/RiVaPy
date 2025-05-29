import json
import numpy as np
import pandas as pd
import yfinance as yf

def get_daily_return_adj_close(data: pd.DataFrame, window_size: int=30):
    print(data['Adj Close'].values.shape[0] )
    return np.array_split(data['Adj Close'].values,window_size)
    
with open('/home/doeltz/doeltz/development/RiVaPy/sandbox/embedding/underlyings.json') as f:
    underlyings = json.load(f)

time_intervalls =[ ("2024-05-01", "2024-08-01"), 
                  ("2024-06-01", "2024-08-01"), 
                  ("2024-07-01", "2024-08-01")] 
result =[]  
for udl in underlyings.values():
    for intervall in time_intervalls:
        print(udl["TICKER"]["YAHOO"])
        try:
            data = yf.download(udl["TICKER"]["YAHOO"] , start=intervall[0], end=intervall[1])
            data = list(data['Adj Close'].values)
            if len(data)>0:
                result.append([f"{udl['TICKER']['YAHOO']}:{intervall[0]}-{intervall[1]}", data] )
        except:
            pass
with open('/home/doeltz/doeltz/development/RiVaPy/sandbox/embedding/yfinance_data/data.json', "w") as f:
    json.dump(result, f)

