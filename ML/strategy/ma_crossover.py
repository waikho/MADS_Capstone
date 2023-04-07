import numpy as np
import pandas as pd

def ma_crossover_trend(close, fast=20, slow=50, exp=True):
    """
    get trend labeling based on moving average crossover strategy

    :param close: (pd series) of closing prices
    :param fast: (int) window for fast moving average, default 20
    :param slow: (int) window for slow moving average, default 50
    :param exp: (boolean) True if use ema, False use rolling average

    return (pd.Dataframe) with fast moving average, slow moving average, and label
    
    """
    d = pd.DataFrame(close)
    if exp == True:
        d['fast_ma'] = d['close'].rolling(window=fast, min_periods=fast).mean()
        d['slow_ma'] = d['close'].rolling(window=slow, min_periods=slow).mean()        
    else:
        d['fast_ma'] = d['close'].ewm(span=fast).mean()
        d['slow_ma'] = d['close'].ewm(span=slow).mean()

    d['label'] = np.nan
    long = d['fast_ma'] >= d['slow_ma'] 
    short = d['fast_ma'] < d['slow_ma'] 
    d.loc[long, 'label'] = 1
    d.loc[short, 'label'] = -1
    d= d.drop(['close'], axis=1)
    return d