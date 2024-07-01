### CHECK THIS CODE IN DETAIL

'''
This here is just a simple example to acquire concrete familiarity
with the notion of Value At Risk
'''
import numpy as np
import pandas as pd
import yfinance as yf

# Get historical stock price data
stock_data = yf.download("AAPL", start="2021-01-01", end="2021-12-31")

# Create a column in the dataset storing the daily returns
# Here we mean simply the FRACTIONAL_CHANGE, FC i, e.:
# FC for the day n is equal to the unique number satisfying:
# x_n = x_(n-1) + FC x_(n-1) 
# Multiplying by 100 we get the percentage change.
stock_data["Daily_Return"] = stock_data["Adj Close"].pct_change()

# Now we *assume that all the returns are samples coming from the same
# probability distribution X*. This assumption is huge, but common.
# Therefore the Value AT Risk of them will essentially be just the percentile:
samples = stock_data["Daily_Return"].dropna()

# The lower, the safer we act
conf_level = 0.05
var = np.percentile(-samples, conf_level)
print(f"V@R: {var:.2f}")
print("(the lower, the better)")



