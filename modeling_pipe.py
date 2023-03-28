import pandas as pd
import numpy as np

import config
import psycopg
from psycopg.rows import dict_row
import getdata as gd

import strategy.trendlabeling as tlb

import afml.filters.filters as flt 
import afml.labeling.triplebarrier as tbar
import afml.util.volatility as vol
from afml.sample_weights.attribution import get_weights_by_return
import features.bars as bars  
import features.marketindicators as mkt
from afml.cross_validation.cross_validation import PurgedKFold
import crossvalidation as cv

from sklearn.model_selection import train_test_split, KFold


def get_alpaca_daily_data(ticker, index=['SPY'], config=config.pgSecrets):
    """
    :param ticker: list of tickers as string
    :param index: list of index as string
    :param secrets: secrets and key
    
    :return tuple of dataframes - df for selected tickers, df for selected index tickers 
    """
    ticker_df = pd.DataFrame()
    index_df = pd.DataFrame()

    for t in ticker:
        _df = gd.getTicker(t, config=config)
        ticker_df = pd.concat(ticker_df, _df) 

    if index:
        for i in index:
            _df = gd.getTicker(i, config=config)
            index_df = pd.concat(index_df, _df) 

    return ticker_df, index_df if index_df else None


def preprocessing():

    return None



# get trades to form dollar bars
trades = df[['datetime', 'close', 'vol']].to_numpy()

# define the dollar value to sample the data
frequency = (df['vol']*df['close']).resample('D').sum()[:-10].mean()/50.0

# generate the dollar bars
dollar_bars = bars.generate_dollarbars(trades, frequency=frequency) 

#_index_SPY = index_SPY.reset_index()
trades_SPY = index_SPY[['datetime', 'close', 'vol']].to_numpy()

# define the dollar value to sample the data
frequency_SPY = (index_SPY['vol']*index_SPY['close']).resample('D').sum()[:10].mean()/50.0

# generate the dollar bars
dollar_bars_SPY = bars.generate_dollarbars(trades_SPY, frequency=frequency_SPY) 

# model 1 - trend scanning labels
time_series = dollar_bars.close.to_numpy()
window_size_max= 7

# get trend scanning labels
label_output = pd.DataFrame(tlb.get_trend_scanning_labels(time_series=time_series, 
                                             window_size_max=window_size_max, 
                                             threshold=0.0,
                                             opp_sign_ct=3,
                                             side='both'), 
                            index= dollar_bars.index[window_size_max-1:])

dollar_bars = dollar_bars.join(label_output, how='outer')

dollar_bars['label'] = dollar_bars['label'].shift(1) 
dollar_bars['slope'] = dollar_bars['slope'].shift(1) 

# Event Filtering
close = dollar_bars.close.copy()

# get Daily Volatility
dailyVolatility = vol.getDailyVol(close, span=50)

# apply cusum filter to identify events as cumulative log return passed threshold
#tEvents = flt.getTEvents(close, h=dailyVolatility.mean()*0.5)
tEvents = flt.cusum_filter(close, threshold=dailyVolatility.mean()*0.5, signal=None)

# Define vertical barrier - subjective judgment
num_days = 5

t1 = tbar.add_vertical_barrier(tEvents, close, num_days=num_days)

# get side labels from trend following method
side_labels = []

for dt in dollar_bars.index:
    side_labels.append(dollar_bars.loc[dt]['label'])

side_labels = pd.Series(side_labels, index=dollar_bars.index)

# get horizontal bars
ptsl = [1,1]

# select minRet
minRet = 0.015 # requires at least 1.5% percent return

# Run in single-threaded mode on Windows
import platform, os
if platform.system() == "Windows":
    cpus = 1
else:
    cpus = os.cpu_count() - 1
    
events = tbar.get_events(dollar_bars.close, 
                         t_events=tEvents, 
                         pt_sl=ptsl, 
                         target=dailyVolatility, 
                         min_ret=minRet, 
                         num_threads=cpus, 
                         vertical_barrier_times=t1,
                         side_prediction=side_labels).dropna()

labels = tbar.get_bins(triple_barrier_events = events, close=close)

# Drop underpopulated labels
clean_labels  = tbar.drop_labels(labels)


# model 2 - features
# # Log Returns
dollar_bars['log_ret'] = np.log(dollar_bars['close']).diff().shift(1)

# Momentum
dollar_bars['mom1'] = dollar_bars['close'].pct_change(periods=1).shift(1)
dollar_bars['mom2'] = dollar_bars['close'].pct_change(periods=2).shift(1)
dollar_bars['mom3'] = dollar_bars['close'].pct_change(periods=3).shift(1)
dollar_bars['mom4'] = dollar_bars['close'].pct_change(periods=4).shift(1)
dollar_bars['mom5'] = dollar_bars['close'].pct_change(periods=5).shift(1)

# Volatility
dollar_bars['volatility_50'] = dollar_bars['log_ret'].rolling(window=50, min_periods=50, center=False).std().shift(1)
dollar_bars['volatility_31'] = dollar_bars['log_ret'].rolling(window=31, min_periods=31, center=False).std().shift(1)
dollar_bars['volatility_15'] = dollar_bars['log_ret'].rolling(window=15, min_periods=15, center=False).std().shift(1)

# Serial Correlation (Takes about 4 minutes)
window_autocorr = 50

dollar_bars['autocorr_1'] = dollar_bars['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=1), raw=False).shift(1)
dollar_bars['autocorr_2'] = dollar_bars['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=2), raw=False).shift(1)
dollar_bars['autocorr_3'] = dollar_bars['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=3), raw=False).shift(1)
dollar_bars['autocorr_4'] = dollar_bars['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=4), raw=False).shift(1)
dollar_bars['autocorr_5'] = dollar_bars['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=5), raw=False).shift(1)

# Get the various log -t returns
dollar_bars['log_t1'] = dollar_bars['log_ret'].shift(1).shift(1)
dollar_bars['log_t2'] = dollar_bars['log_ret'].shift(2).shift(1)
dollar_bars['log_t3'] = dollar_bars['log_ret'].shift(3).shift(1)
dollar_bars['log_t4'] = dollar_bars['log_ret'].shift(4).shift(1)
dollar_bars['log_t5'] = dollar_bars['log_ret'].shift(5).shift(1)

# relative strength SPY at various -t
dollar_bars['rs_SPY_t1'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_SPY.close).shift(1)
dollar_bars['rs_SPY_t2'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_SPY.close).shift(2)
dollar_bars['rs_SPY_t3'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_SPY.close).shift(3)
dollar_bars['rs_SPY_t4'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_SPY.close).shift(4)
dollar_bars['rs_SPY_t5'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_SPY.close).shift(5)


dollar_bars = dollar_bars.join(clean_labels['bin']).dropna()

X = dollar_bars.iloc[:, :-1]
y = dollar_bars.iloc[:, -1]

col = ['slope', 'label', 'log_ret',
       'mom1', 'mom2', 'mom3', 'mom4', 'mom5', 'volatility_50',
       'volatility_31', 'volatility_15', 'autocorr_1', 'autocorr_2',
       'autocorr_3', 'autocorr_4', 'autocorr_5', 'log_t1', 'log_t2', 'log_t3',
       'log_t4', 'log_t5', 'rs_SPY_t1', 'rs_SPY_t2', 'rs_SPY_t3', 'rs_SPY_t4',
       'rs_SPY_t5']

X = X[col]

RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=RANDOM_STATE)

return_based_sample_weights = get_weights_by_return(events.loc[X_train.index], dollar_bars.loc[X_train.index, 'close'])
return_based_sample_weights_test = get_weights_by_return(events.loc[X_test.index], dollar_bars.loc[X_test.index, 'close'])

parameters = {'C':[100,
                   1000],
              'gamma':[0.001,
                       0.0001], 
              }

n_splits=4

cv_gen_standard = KFold(n_splits)
cv_gen_purged = PurgedKFold(n_splits=n_splits, samples_info_sets=events.loc[X_train.index].t1)


best_params = cv.perform_grid_search(X_train, y_train, cv_gen_purged, 'f1', parameters, events, dollar_bars, type='seq_boot_SVC', sample_weight=return_based_sample_weights.values)

model_metrics = pd.DataFrame(columns = ['type', 'best_model', 'best_cross_val_score', 'recall', 'precision', 'accuracy','run_time'])
model_metrics = model_metrics.append(best_params, ignore_index = True)  
model_metrics['train_test'] = 'Train'

clf = best_params['top_model'].squeeze()

from sklearn.metrics import f1, recall_score, precision_score, accuracy_score

y_pred = clf.predict(X_test)

test_results = {
    'type':'seq_boot_SVC',
    'best_model':clf,
    'best_cross_val_score':f1(np.array(y.iloc[y_test]), y_pred, sample_weight=return_based_sample_weights_test),
    'recall': recall_score(np.array(y.iloc[y_test]), y_pred, sample_weight=return_based_sample_weights_test),
    'precision':precision_score(np.array(y.iloc[y_test]), y_pred, sample_weight=return_based_sample_weights_test), 
    'accuracy':accuracy_score(np.array(y.iloc[y_test]), y_pred, sample_weight=return_based_sample_weights_test),
    'run_time':0,
    'train_test':'Test'
}

model_metrics.iloc[1] = test_results




clf