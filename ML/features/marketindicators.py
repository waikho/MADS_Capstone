

import pandas as pd
import numpy as np
#import pandas_datareader as pdr
import getdata as gd

def get_relative_strength(stock_close_price, index_close_price, span=20):
    """
    Calculate the relative strength of a stock to the
    a selected index using the exponentially weighted moving average

    :param stock_close_price: series of stock close price with datetime as index
    :param index_close_price: series of index close price with datetime as index
    :param span: window to calculate the exponential rolling average
    """

    # Calculate the returns for both the stock and the index
    stock_returns = stock_close_price.pct_change()
    index_returns = index_close_price.pct_change()

    # Calculate the exponentially weighted moving average of the returns for both the stock and the index
    stock_ewma = stock_returns.ewm(span=span, adjust=False).mean()
    index_ewma = index_returns.ewm(span=span, adjust=False).mean()

    # Calculate the relative strength of the stock to the index
    relative_strength = stock_ewma / index_ewma

    return relative_strength.dropna()


def returns(s):
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index=s.index[1:]))

def df_rolling_autocorr(df, window, lag=1):
    """
    Compute rolling column-wise autocorrelation for a DataFrame.
    """

    return (df.rolling(window=window).corr(df.shift(lag)))


