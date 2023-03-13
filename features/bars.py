
import numpy as np
import pandas as pd
from datetime import datetime


def generate_dollarbars(trades, frequency=1000):

    """
    # expects a numpy array with trades
    # each trade is composed of: [time, price, quantity]
    https://towardsdatascience.com/advanced-candlesticks-for-machine-learning-ii-volume-and-dollar-bars-6cda27e3201d
    """
    times = trades[:,0]
    prices = trades[:,1]
    volumes = trades[:,2]
    ans = np.zeros(shape=(len(prices), 6))
    candle_counter = 0
    dollars = 0
    lasti = 0
    for i in range(len(prices)):
        dollars += volumes[i]*prices[i]
        if dollars >= frequency:
            ans[candle_counter][0] = np.datetime64(times[i])       # time
            ans[candle_counter][1] = prices[lasti]                     # open
            ans[candle_counter][2] = np.max(prices[lasti:i+1])         # high
            ans[candle_counter][3] = np.min(prices[lasti:i+1])         # low
            ans[candle_counter][4] = prices[i]                         # close
            ans[candle_counter][5] = np.sum(volumes[lasti:i+1])        # volume
            candle_counter += 1
            lasti = i+1
            dollars = 0
    
    ans = ans[:candle_counter]
    columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    ans = pd.DataFrame(ans, columns=columns)
    ans['time'] = ans['time'].apply(lambda x: datetime.fromtimestamp(x/1000000))
    ans.index = pd.DatetimeIndex(ans.time)
    ans = ans.drop(columns='time')

    return ans

