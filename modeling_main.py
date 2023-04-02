import pandas as pd
import numpy as np

import config
import psycopg
from psycopg.rows import dict_row
import getdata as gd
import platform, os
import datetime

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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def get_data(ticker):

    df = gd.getTicker_Daily(ticker)
    df = df.sort_values(by='tranx_date')
    df.index = df.tranx_date

    return df

def get_index(ticker_df, index_ticker='SPY'):
    """
    return df for a selected index that matches the ticker
    
    """
    start_date = ticker_df.tranx_date.min()
    end_date = ticker_df.tranx_date.max()
    df = gd.getTicker_Daily(index_ticker)
    df = df.sort_values(by='tranx_date')
    index_df = df[(df['tranx_date']>=start_date) & (df['tranx_date']<=end_date)]
    index_df.index = index_df.tranx_date

    return index_df


def normalizing(ticker_df, dv_multiple=2):
    """
    Perform preprocessing for one ticker 

    :param ticker_df: (dataframe) single ticker dataframe from get_alpaca_daily_data
    :param dv_multiple: (int) dollar value multiple, this determine how frequent the resample show be, default as 2

    :return dollar_bars_df: dataframe, normalized dollar_bars
    """
        
    # generate dollar bars
    dv_thres = (ticker_df['vol']*ticker_df['close']).resample('D').sum()[:-10].mean()*dv_multiple # average of the dollar value from the last 10 days * 2
    dollar_bars = bars.generate_dollarbars(ticker_df, dv_thres=dv_thres) 

    return dollar_bars


def trend_labeling(dollar_bars, window_size_max=7):
    """
    Perform trend_labeling for one ticker
    :param dollar_bars: (dataframe) processed dollar bars
    :param window_size_max: (int) trend scanning label windowsize

    :return dollar_bars_df: dataframe, normalized dollar_bars with trend labels and trend slope for all symbols in ticker_df
    
    """
    time_series = dollar_bars.close.to_numpy()
    label_output = pd.DataFrame(tlb.get_trend_scanning_labels(
                                        time_series=time_series, 
                                        window_size_max=window_size_max), 
                                        index= dollar_bars.index[window_size_max-1:])
    dollar_bars = dollar_bars.join(label_output)
    dollar_bars['label'] = dollar_bars['label'].shift(1) # remove lookahead bias
    dollar_bars['slope'] = dollar_bars['slope'].shift(1) # remove lookahead bias

    return dollar_bars

def meta_labeling(dollar_bars, span=50, filter_multiple=1.0, num_days=5, ptsl=[1.5,1], minRet=0.0):
    """
    Perform meta-labeling for one ticker

    :param dollar_bars: (dataframe) processed dollar bars
    :param span: (int) the of rolling days to determine rolling volatility, default as 50 days
    :param filter_multiple: (float) multiple applied to the average volatility to determine threshold for cusum event filter, default as 0.5
    :param num_days: (int) vertical barrier of triple barrier labeling i.e. number of days before position is closed, default is 5 days
    :param ptsl: (list(int, int)) defines profit to loss ratio for the horizontal barriers in triple barrier labeling method, default is [1.5, 1]
    :param minRet: (float) defines the minimum % return to capture the event as a tradable event, default is 0.015 where 0.01 is \
        considered as transaction fees and 0.005 as slippage

    :return dollar_bars: (dataframe) DateTimeIndex of tradable events
    """

    # get 50-day exponential rolling volatility
    close = dollar_bars.close.copy()
    dailyVolatility = vol.getDailyVol(close, span=span)

    # Identify events as cumulative log return that passes dailyVolitility threshold
    tEvents = flt.cusum_filter(close, threshold=dailyVolatility.mean()*filter_multiple)

    # define vertical barrier - subjective judgment
    t1 = tbar.add_vertical_barrier(tEvents, close, num_days=num_days)

    # check platform and run single threaded on Mac or Windows
    if platform.system() == "Windows" or platform.system() == "Mac OS X":
        cpus = 1
    else:
        cpus = os.cpu_count() - 1
    
    # get tradable eveents
    events = tbar.get_events(dollar_bars.close, 
                                t_events=tEvents, 
                                pt_sl=ptsl, 
                                target=dailyVolatility, 
                                min_ret=minRet, 
                                num_threads=cpus, 
                                vertical_barrier_times=t1,
                                side_prediction=dollar_bars.label).dropna()
    events['isEvent'] = 1

    # get the triple barrier labels for all tradable events 
    labels = tbar.get_bins(triple_barrier_events = events, close=close)

    # drop underpopulated labels
    clean_labels  = tbar.drop_labels(labels)

    # join labels back to dataframe
    dollar_bars = dollar_bars.join(clean_labels[['bin','ret']]).join(events[['t1','trgt','isEvent']])

    return dollar_bars

def features_momentum(dollar_bars):
    """
    add momentum features to dollar bar df of one ticker
    
    """
    dollar_bars['mom1'] = dollar_bars['close'].pct_change(periods=1).shift(1)
    dollar_bars['mom2'] = dollar_bars['close'].pct_change(periods=2).shift(1)
    dollar_bars['mom3'] = dollar_bars['close'].pct_change(periods=3).shift(1)
    dollar_bars['mom4'] = dollar_bars['close'].pct_change(periods=4).shift(1)
    dollar_bars['mom5'] = dollar_bars['close'].pct_change(periods=5).shift(1)

    return dollar_bars

def features_volatility(dollar_bars):
    """
    add volatility features to dollar bar df of one ticker
    
    """
    log_ret = np.log(dollar_bars['close']).diff()

    dollar_bars['volatility50'] = log_ret.rolling(window=50, min_periods=50, center=False).std().shift(1)
    dollar_bars['volatility31'] = log_ret.rolling(window=31, min_periods=31, center=False).std().shift(1)
    dollar_bars['volatility15'] = log_ret.rolling(window=15, min_periods=15, center=False).std().shift(1)

    return dollar_bars

def features_log_returns(dollar_bars):
    """
    add log returns at vairous -t to dollar bar df of one ticker
    
    """
    log_ret = np.log(dollar_bars['close']).diff()

    dollar_bars['log1'] = log_ret.shift(1)
    dollar_bars['log2'] = log_ret.shift(2)
    dollar_bars['log3'] = log_ret.shift(3)
    dollar_bars['log4'] = log_ret.shift(4)
    dollar_bars['log5'] = log_ret.shift(5)

    return dollar_bars

def features_serial_correlation(dollar_bars, corr_period=50):
    """
    add serial correlation at vairous -t to dollar bar df of one ticker
    
    """
    log_ret = np.log(dollar_bars['close']).diff()

    dollar_bars['autocorr1'] = log_ret.rolling(window=corr_period, min_periods=corr_period, center=False).apply(lambda x: x.autocorr(lag=1), raw=False).shift(1)
    dollar_bars['autocorr2'] = log_ret.rolling(window=corr_period, min_periods=corr_period, center=False).apply(lambda x: x.autocorr(lag=2), raw=False).shift(1)
    dollar_bars['autocorr3'] = log_ret.rolling(window=corr_period, min_periods=corr_period, center=False).apply(lambda x: x.autocorr(lag=3), raw=False).shift(1)
    dollar_bars['autocorr4'] = log_ret.rolling(window=corr_period, min_periods=corr_period, center=False).apply(lambda x: x.autocorr(lag=4), raw=False).shift(1)
    dollar_bars['autocorr5'] = log_ret.rolling(window=corr_period, min_periods=corr_period, center=False).apply(lambda x: x.autocorr(lag=5), raw=False).shift(1)

    return dollar_bars

def features_SPY_RS(dollar_bars, dollar_bars_SPY):
    """
    add relative strength SPY at various -t to dollar bar df of one ticker
    
    """
    dollar_bars['rs_SPY_1'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_SPY.close).shift(1)
    dollar_bars['rs_SPY_2'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_SPY.close).shift(2)
    dollar_bars['rs_SPY_3'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_SPY.close).shift(3)
    dollar_bars['rs_SPY_4'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_SPY.close).shift(4)
    dollar_bars['rs_SPY_5'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_SPY.close).shift(5)

    return dollar_bars


def features_COMP_RS(dollar_bars, dollar_bars_COMP):
    """
    add relative strength SPY at various -t to dollar bar df of one ticker
    
    """
    dollar_bars['rs_COMP_1'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_COMP.close).shift(1)
    dollar_bars['rs_COMP_2'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_COMP.close).shift(2)
    dollar_bars['rs_COMP_3'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_COMP.close).shift(3)
    dollar_bars['rs_COMP_4'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_COMP.close).shift(4)
    dollar_bars['rs_COMP_5'] = mkt.get_relative_strength(dollar_bars.close, dollar_bars_COMP.close).shift(5)

    return dollar_bars

def modeling(dollar_bars, type='seq_boot_SVC', RANDOM_STATE = 42):
    """
    modeling and perform hyperparameter tuning for a single stock dataframe
    dataframe comes with trend labels, meta labels, and additional features
    :param dollar_bars: (dataframe) with features and events
    :param type: (sklearn model) default as sequential bootstrapping SVC
    :param RANDOM_STATE: (int) random state seed, default as 42
    :return clf, model_metrics: (classifier, dataframe) classification model for the ticker, model metrics for train and test dataset on a dataframe
    """
    # this symbol
    symbol = dollar_bars.symbol.unique()[0]

    # train test split
    _X = dollar_bars.iloc[:, :-1]
    y = dollar_bars.iloc[:, -1]
    col = ['index', 'tranx_date', 'symbol', 'open', 'close', 'high', 'low', 'vol', 'vwap', 'trade_count',
           't1', 'target', 'ret'
           ]
    X = _X.drop(col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=RANDOM_STATE)

    # sample weights
    events = dollar_bars[dollar_bars['isEvent']==1]
    return_based_sample_weights = get_weights_by_return(events.loc[X_train.index], dollar_bars.loc[X_train.index, 'close'])
    return_based_sample_weights_test = get_weights_by_return(events.loc[X_test.index], dollar_bars.loc[X_test.index, 'close'])

    # set hyperparameter params
    parameters = {'C':[100, 1000],
                'gamma':[0.001, 0.0001], 
                }

    # set KFold splits
    n_splits=4
    cv_gen_purged = PurgedKFold(n_splits=n_splits, samples_info_sets=events.loc[X_train.index].t1)

    # get the best parameters for the selected model
    best_params = cv.perform_grid_search(X_train, y_train, cv_gen_purged, 'f1', parameters, events, dollar_bars, type=type, sample_weight=return_based_sample_weights.values)

    # table to record down training results
    model_metrics = pd.DataFrame(columns = ['type', 'best_model', 'best_cross_val_score', 'recall', 'precision', 'accuracy','run_time'])
    model_metrics = model_metrics.append(best_params, ignore_index = True)  
    model_metrics['train_test'] = 'Train'
    model_metrics['symbol'] = symbol
    model_metrics['performed_on'] = datetime.datetime.now()

    # perform testing and record score
    y_pred = clf.predict(X_test)
    test_results = {
        'type':'seq_boot_SVC',
        'best_model':clf,
        'best_cross_val_score':f1_score(np.array(y.iloc[y_test]), y_pred, sample_weight=return_based_sample_weights_test),
        'recall': recall_score(np.array(y.iloc[y_test]), y_pred, sample_weight=return_based_sample_weights_test),
        'precision':precision_score(np.array(y.iloc[y_test]), y_pred, sample_weight=return_based_sample_weights_test), 
        'accuracy':accuracy_score(np.array(y.iloc[y_test]), y_pred, sample_weight=return_based_sample_weights_test),
        'run_time':0,
        'train_test':'Test',
        'symbol':symbol,
        'performed_on':datetime.datetime.now()
    }
    model_metrics.iloc[1] = test_results
    
    # final model with best params
    clf = best_params['top_model'].squeeze()

    return clf, model_metrics

def get_one_model(ticker, config=config.pgSecrets):

    """
    get the meta-labelled model for one ticker
    :return clf, model_metrics: (sklearn classification model object, dataframe): tuple of model object and a dataframe of modeling metrics
    
    """

    ticker_df = gd.getTicker_Daily(ticker, config=config)
    ticker_df.index = pd.to_datetime(ticker_df.tranx_date)

    # preprocessing
    dollar_bars = normalizing(ticker_df)

    # add features
    dollar_bars = trend_labeling(dollar_bars)
    dollar_bars = meta_labeling(dollar_bars)
    dollar_bars = features_momentum(dollar_bars)
    dollar_bars = features_volatility(dollar_bars)
    dollar_bars = features_log_returns(dollar_bars)
    dollar_bars = features_serial_correlation(dollar_bars)

    # add features - relative strength to SPY
    index_SPY = get_index(ticker_df, 'SPY')
    dollar_bars_SPY = bars.transform_index_based_on_dollarbar(dollar_bars, index_SPY)
    dollar_bars = features_SPY_RS(dollar_bars, dollar_bars_SPY)

    # # add features - relative strength to COMP
    # index_COMP = get_index(ticker_df, 'COMP')
    # dollar_bars_COMP = bars.transform_index_based_on_dollarbar(dollar_bars, index_COMP)
    # dollar_bars = features_COMP_RS(dollar_bars, dollar_bars_COMP)
   

    # get the output model and train test metrics
    clf, model_metrics = modeling(dollar_bars)

    return clf, model_metrics

def get_multiple_models(ticker_lst, config=config.pgSecrets):
    """
    :param ticker_lst: (list) of ticker symbols

    :return clfs, model_metrics_df: (dict, dataframe): tuple of dictionary of symbol:classifiers k,v pairs \
        and a dataframe of modeling metrics
    """
    clfs = {}
    model_metrics_df = pd.DataFrame()

    for ticker in ticker_lst:
        clf, model_metrics = get_one_model(ticker, config=config)
        clfs[ticker] = clf
        model_metrics_df = pd.concat([model_metrics_df, model_metrics], ignore_index=True)
    
    return clfs, model_metrics_df