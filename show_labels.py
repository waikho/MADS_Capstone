import pandas as pd
import matplotlib.pyplot as plt
import trendlabeling as tlb
import numpy as np
import datetime as dt

import getdata as gd
import filters as flt 
import triplebarrier as tb


df = gd.get_yf_data(tickers= "SPY AAPL ALGM DNOW", period='1y', interval='1d')
df = df[df['Ticker'] == 'ALGM']

# convert Adj Close to numpy
time_series = df['Adj Close'].to_numpy()

# define window size
window_size_max = 7

#define threshold 
threshold = 0.0

# get trend scanning labels
label_output = tlb.get_trend_scanning_labels(time_series=time_series, window_size_max=window_size_max, threshold=threshold)

# drop last rolling window size -1 rows
n = window_size_max-1
df.drop(df.tail(n).index, inplace = True)

# append the slope and labels to the df
df['slope'] = label_output['slope']
df['label'] = label_output['label']


# get the event points with cumsum filter
raw_time_series = df['Adj Close']
all_events, pos_events, neg_events = flt.cusum_filter(raw_time_series, threshold=0.08, time_stamps=True)


# create scatter
plt.figure(figsize=(20,5))
plt.scatter(df.index, df['Adj Close'], s=20, c=df.label, cmap='RdYlGn')
for d in pos_events:
   plt.axvline(d, color='green') 
for d in neg_events:
   plt.axvline(d, color='red') 
plt.show()

# get daily volatility

df['DailyVol'] = tb.getDailyVolatility(raw_time_series, span=100)
df['DailyVol_upper'] = df['Adj Close'] + df.DailyVol/2
df['DailyVol_lower'] = df['Adj Close'] - df.DailyVol/2

# create scatter with daily vol
fig, ax = plt.subplots()
#plt.figure(figsize=(20,5))
ax.plot(df.index, df['Adj Close'], '-')
ax.fill_between(df.index, df['DailyVol_lower'], df['DailyVol_upper'], alpha = 0.1, color='b')

df