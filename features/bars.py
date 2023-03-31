
import numpy as np
import pandas as pd
import datetime


def generate_dollarbars(ticker_df, dv_thres=1000):

    """
    # expects a numpy array with trades
    # each trade is composed of: [time, price, quantity]
    https://towardsdatascience.com/advanced-candlesticks-for-machine-learning-ii-volume-and-dollar-bars-6cda27e3201d
    """

    times = ticker_df['tranx_date']
    prices = ticker_df['close']
    volumes = ticker_df['vol']
    ans = pd.DataFrame(columns = ['time', 'open', 'high', 'low', 'close', 'volume'])
    candle_counter = 0
    dollars = 0
    lasti = 0
    temp_dict = {}
    for i in range(len(prices)):
        dollars += volumes[i]*prices[i]
        if dollars >= dv_thres:
            temp_dict['time'] = times[i]       # time
            temp_dict['open'] = prices[lasti]                     # open
            temp_dict['high'] = np.max(prices[lasti:i+1])         # high
            temp_dict['low'] = np.min(prices[lasti:i+1])         # low
            temp_dict['close'] = prices[i]                         # close
            temp_dict['volume'] = np.sum(volumes[lasti:i+1])        # volume
            
            ans = pd.concat([ans, pd.DataFrame([temp_dict])], ignore_index=True)

            candle_counter += 1
            lasti = i+1
            dollars = 0

    ans.index = pd.DatetimeIndex(ans.time)
    ans = ans.drop(columns='time')

    return ans


def transform_index_based_on_dollarbar(dollar_bars, index_df):
    """
    given a df of index, transform and align it based on the index datatime of the dollar_bar df

    """
    db_dates = dollar_bars.index
    ans = pd.DataFrame(columns = ['time', 'open', 'high', 'low', 'close', 'volume'], index=db_dates)

    for i, ed_date in enumerate(db_dates):
        if i == 0:
            st_date=index_df.index.min()
        else:
            st_date=db_dates[i]
        
        sub_df = index_df[(index_df['tranx_date']>st_date) & (index_df['tranx_date']<=ed_date)]

        ans.loc[ed_date]['time'] = ed_date                      # time
        ans.loc[ed_date]['open'] = index_df.loc[st_date].open   # open
        ans.loc[ed_date]['high'] = np.max(sub_df.close)         # high
        ans.loc[ed_date]['low'] = np.min(sub_df.close)          # low
        ans.loc[ed_date]['close'] = index_df.loc[ed_date].close # close
        ans.loc[ed_date]['volume'] = np.sum(sub_df.vol)         # volume

    ans.index = pd.DatetimeIndex(ans.time)
    ans = ans.drop(columns='time')


    return ans