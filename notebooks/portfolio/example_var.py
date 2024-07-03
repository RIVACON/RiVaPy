### CHECK THIS CODE IN DETAIL

'''
This here is just a simple example to acquire concrete familiarity
with the notion of Value At Risk
'''
import numpy as np
import pandas as pd
import yfinance as yf
from math import floor
import matplotlib.pyplot as plt

def divide_in_batches(samples, n_batches):
    '''
    Take a list of samples and divide them in bacthes,
    where each batch contains averages of elements.
    '''
    tot = len(samples)
    batch_size = floor(tot / n_batches)    
    dropped = tot - (batch_size * n_batches)
    print(f"Dividing {tot} samples in {n_batches} batches ", end='')
    print(f"of size {batch_size}. Dropping last {dropped} elements.") 
    batched = np.zeros(n_batches)
    for nth in range(n_batches):
        # Here to improve the bacthes construction
        offset = nth * batch_size
        batched[nth] = np.mean(samples[offset : offset+batch_size])
#        breakpoint()
    return batched
#---        


def take_data(ticker, start, end):
    # Get historical stock price data
    stock_data = yf.download(ticker, start, end)

    # Create a column in the dataset storing the daily returns
    # Here we mean simply the FRACTIONAL_CHANGE, FC i, e.:
    # FC for the day n is equal to the unique number satisfying:
    # x_n = x_(n-1) + FC x_(n-1) 
    # Multiplying by 100 we get the percentage change.
    stock_data["Daily_Return"] = stock_data["Adj Close"].pct_change()
    samples = stock_data["Daily_Return"].dropna()
    return samples
#---

def compute_var(time_series):
    # The lower, the safer we act
    conf_level_in_percent = 5
    var = - np.percentile(time_series, conf_level_in_percent)
    print(f"V@R: {var:.2f}")
    print("(the lower, the better)")
    return var
#---


if __name__ == "__main__":
    ticker = "MSFT"
    start = "2020-01-01"
    end = "2024-06-01"
    data = take_data(ticker, start, end)
    compute_var(data)
    
    fig, ax = plt.subplots()
    ax.plot(data)
    fig.autofmt_xdate()
    plt.show()

    
