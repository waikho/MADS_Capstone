import pandas as pd
import numpy as np

import config
import psycopg
from psycopg.rows import dict_row
import getdata as gd
import platform, os
import datetime
import time
import math

import strategy.trendlabeling as tlb
import strategy.ma_crossover as ma

import afml.filters.filters as flt 
import afml.labeling.triplebarrier as tbar
import afml.util.volatility as vol
from afml.sample_weights.attribution import get_weights_by_return
import features.bars as bars  
import features.marketindicators as mkt
from afml.cross_validation.cross_validation import PurgedKFold
import crossvalidation as cv

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from scipy.stats import ks_2samp

import warnings
warnings.filterwarnings('ignore')


def get_data(ticker):

    df = gd.getTicker_Daily(ticker)
    df = df.sort_values(by='tranx_date')
    df.index = df.tranx_date

    return df

def get_index(ticker_df, index_ticker='SPY'):
    """
    return df for a selected index that matches the ticker
    
    """
    df = gd.getTicker_Daily(index_ticker)
    df = df.sort_values(by='tranx_date')

    tic_start_date = ticker_df.tranx_date.min()
    tic_end_date = ticker_df.tranx_date.max()

    idx_start_date = df.tranx_date.min()
    idx_end_date = df.tranx_date.max()

    start_date = max([tic_start_date, idx_start_date])
    end_date = min([tic_end_date, idx_end_date])

    index_df = df[(df['tranx_date']>=start_date) & (df['tranx_date']<=end_date)]
    index_df.index = index_df.tranx_date

    return index_df


def normalizing(ticker_df, dv_multiple=5):
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

def ma_crossover_labeling(dollar_bars, exp=True):
    """
    Perform trend_labeling for one ticker
    :param dollar_bars: (dataframe) processed dollar bars
    :return dollar_bars_df: dataframe, normalized dollar_bars with trend labels and trend fast/slow moving avg for all symbols in ticker_df
    """

    label_output = ma.ma_crossover_trend(dollar_bars.close, exp=exp)
    dollar_bars = dollar_bars.join(label_output)
    dollar_bars['label'] = dollar_bars['label'].shift(1) # remove lookahead bias
    dollar_bars['fast_ma'] = dollar_bars['fast_ma'].shift(1) # remove lookahead bias    
    dollar_bars['slow_ma'] = dollar_bars['slow_ma'].shift(1) # remove lookahead bias    

    return dollar_bars

def meta_labeling(dollar_bars, span=50, filter_multiple=1.0, num_days=20, ptsl=[1.5,1], minRet=0.0):
    """
    Perform meta-labeling for one ticker

    :param dollar_bars: (dataframe) processed dollar bars
    :param span: (int) the of rolling days to determine rolling volatility, default as 50 days
    :param filter_multiple: (float) multiple applied to the average volatility to determine threshold for cusum event filter, default as 0.5
    :param num_days: (int) vertical barrier of triple barrier labeling i.e. number of days before position is closed, default is 5 days
    :param ptsl: (list(int, int)) defines profit to loss ratio for the horizontal barriers in triple barrier labeling method, default is [1.5, 1]
    :param minRet: (float) defines the minimum % return to capture the event as a tradable event, default is 0.015 where 0.01 is \
        considered as transaction fees and 0.005 as slippage

    :return events, dollar_bars: (tuple of dataframe, dataframe) DateTimeIndex of tradable events from triple barrier method, dollar bars with events data
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

    # get the triple barrier labels for all tradable events 
    labels = tbar.get_bins(triple_barrier_events = events, close=close)

    # drop underpopulated labels
    clean_labels  = tbar.drop_labels(labels)

    # join labels back to dataframe
    dollar_bars = dollar_bars.join(clean_labels['bin'])

    # join the ret from dollar_bars to event df
    events = events.join(clean_labels[['bin', 'ret']]).join(dollar_bars[['close', 'label']])

    return events, dollar_bars

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

def error_df(symbol, model_metrics, message):
    """
    helper method for modeling
    output df to record down un-trainable cases
    :param symbol: (str) ticker symbol
    :param model_metrics: (dataframe) of model results
    :param message: (str) error message from modeling 

    """
    model_metrics.loc[0] = np.nan
    model_metrics['train_test'] = message
    model_metrics['symbol'] = symbol
    model_metrics['performed_on'] = datetime.datetime.now()
    model_metrics['event_st_date'] = np.nan
    model_metrics['event_en_date'] = np.nan
    return model_metrics

def backtest_metrics(events, X_test, y_test, y_pred, transaction_cost=0.0, slippage=0.0, risk_free_rate=0.036):
    """
    helper method for modeling method
    Calculate the cumulative returns of the ticker in the test set
    
    return (dict) of df and metrics
    """
    bt_metrics = {}

    # prep the dataframe
    results_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}, index=X_test.index)
    X_test_with_results = pd.concat([X_test, results_df], axis=1)
    merged_df = pd.merge(X_test_with_results, events, left_index=True, right_index=True)

    merged_df['y_pred_trade_ret'] = 0
    merged_df.loc[merged_df['y_pred'] == 1, 'y_pred_trade_ret'] = merged_df.loc[merged_df['y_pred'] == 1, 'ret'] - (transaction_cost + slippage)
    merged_df_sorted = merged_df.sort_values('t1')

    ### cumulative returns by day plot
    merged_df_sorted['cumulative_returns'] = (merged_df_sorted['y_pred_trade_ret'] + 1).cumprod() - 1
    cum_rtn_plot = merged_df_sorted.groupby('t1')['cumulative_returns', 'y_pred_trade_ret'].last()
    bt_metrics['cum_rtn_plot'] = pd.DataFrame(cum_rtn_plot)
    bt_metrics['cum_rtn'] = cum_rtn_plot.iloc[-1].cumulative_returns

    # annualized return
    days_diff = (X_test.index.max() - X_test.index.min()).days
    bt_metrics['annualized_rtn'] = math.pow(1+bt_metrics['cum_rtn'], 365/days_diff) -1

    ### sharpe ratio by day plot
    daily_risk_free_rate = risk_free_rate/ 252

    # calculate daily returns for each trading day
    trading_days = pd.DataFrame(pd.bdate_range(start=merged_df.index.min(), end=cum_rtn_plot.index.max(), freq='B'), columns=['trading_days'])
    trading_days.index =trading_days['trading_days']
    sharpe_df = pd.merge(trading_days, cum_rtn_plot['cumulative_returns'], left_index=True, right_index=True, how='left').fillna(method='ffill')
    sharpe_df['cumulative_returns'] = sharpe_df['cumulative_returns'].fillna(0)
    sharpe_df['daily_returns'] = sharpe_df['cumulative_returns'].diff().fillna(0)
    sharpe_df['day'] = 1
    sharpe_df['trading_days'] = sharpe_df['day'].cumsum()
    sharpe_df['rolling_std_dev'] =sharpe_df['daily_returns'].expanding().std(ddof=0)
    sharpe_df['cumulative_sharpe'] = ((sharpe_df['cumulative_returns']/sharpe_df['trading_days'])-daily_risk_free_rate) / sharpe_df['rolling_std_dev'] * np.sqrt(252)
    bt_metrics['sharpe_plot'] = sharpe_df.fillna(0)

    ### sharpe ratio of investment

    # Calculate the average daily return and standard deviation of daily returns
    avg_daily_return = np.mean(bt_metrics['sharpe_plot'].daily_returns)
    std_daily_return = np.std(bt_metrics['sharpe_plot'].daily_returns)

    # Calculate the Sharpe ratio
    bt_metrics['sharpe_ratio'] = round((avg_daily_return - daily_risk_free_rate) / std_daily_return * np.sqrt(252), 2)

    return bt_metrics

def modeling(symbol, events, dollar_bars, type='sequential_bootstrapping_SVC', transaction_cost=0.0, slippage=0.0, risk_free_rate=0.036, RANDOM_STATE = 42):
    """
    modeling and perform hyperparameter tuning for a single stock dataframe
    dataframe comes with trend labels, meta labels, and additional features
    :param symbol: (str) symbol of the dollar bar
    :param events: (dataframe) triple barrier events
    :param dollar_bars: (dataframe) with features and events
    :param type: (sklearn model) default as sequential bootstrapping SVC
    :param transaction_cost: (float) assumption to deduct from returns for calculating cum returns, default 0.0
    :param slippage: (float) assumption to deduct from returns for calculating cum returns, default 0.0
    :param RANDOM_STATE: (int) random state seed, default as 42
    :return clf, model_metrics: (classifier, dataframe) classification model for the ticker, model metrics for train and test dataset on a dataframe
    """
    # df to store results
    model_metrics = pd.DataFrame(columns = ['type', 'best_model', 'best_cross_val_score', 'recall', 'precision', 'accuracy','run_time'])

    # train test split
    col = ['open', 'high', 'low', 'close', 'volume', 'bin']
    dollar_bars = dollar_bars.dropna()
    X = dollar_bars.drop(col, axis=1)
    y = dollar_bars['bin']

    if len(y.unique()) < 2:
        print('Only one class found. Unable to train model for {}.'.format(symbol))
        error_df(symbol, model_metrics, 'No Train - single class')
        return None, model_metrics
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=RANDOM_STATE)
    except ValueError:
        print('ValueError - Unable to train model for {}'.format(symbol))
        error_df(symbol, model_metrics, 'No Train - check train test split')
        return None, model_metrics
        
    # sample weights
    return_based_sample_weights = get_weights_by_return(events.loc[X_train.index], dollar_bars.loc[X_train.index, 'close'])
    return_based_sample_weights_test = get_weights_by_return(events.loc[X_test.index], dollar_bars.loc[X_test.index, 'close'])

    # set hyperparameter params
    parameters = {'max_depth':[3, 5, 7, 9],
              'n_estimators':[25, 50, 100, 250],
              'C':[4000, 6000, 8000],
                'gamma':[0.0001, 0.00001], 
                }

    # set KFold splits
    n_splits=4
    cv_gen_purged = PurgedKFold(n_splits=n_splits, samples_info_sets=events.loc[X_train.index].t1)

    # get the best parameters for the selected model
    try:
        best_params = cv.perform_grid_search(X_train, y_train, cv_gen_purged, 'f1', parameters, events, dollar_bars, type=type, sample_weight=return_based_sample_weights.values)
    except ValueError:
        print('Small sample size - Unable to train model for {}'.format(symbol))
        error_df(symbol, model_metrics, 'No Train - small sample size')
        return None, model_metrics
    
    # prep x_test data
    col = X_test.columns.to_list()
    idx = X_test.index
    X_test_scaled = StandardScaler().fit_transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=col, index=idx)

    #prep X_train data
    col = X_test.columns.to_list()
    idx = X_test.index
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_test_scaled, columns=col, index=idx)    
    
    # perform KS test on model drift
    isDrifted = 0
    ks_stat, p_value = ks_2samp(X_train_scaled.values.flatten(), X_test_scaled.values.flatten())
    alpha = 0.05 #threshold for p-value
    if p_value < alpha:
        isDrifted = 1

    # table to record down training results

    model_metrics = model_metrics.append(best_params, ignore_index = True)  
    model_metrics['train_test'] = 'Train'
    model_metrics['symbol'] = symbol
    model_metrics['isDrifted'] = isDrifted
    model_metrics['ks_stat'] = ks_stat
    model_metrics['ks_p_value'] = p_value
    model_metrics['cum_rtn'] = 0.0
    model_metrics['annualized_rtn'] = 0.0
    model_metrics['sharpe_ratio'] = 0.0
    model_metrics['performed_on'] = datetime.datetime.now()
    model_metrics['event_st_date'] = X_train.index.min()
    model_metrics['event_en_date'] = X_train.index.max()

    # perform testing and record score
    clf = best_params['best_model']
    t0 = time.time()

    y_pred = clf.predict(X_test_scaled)
    t1 = time.time()

    bt_metrics = backtest_metrics(events, X_test, y_test, y_pred, transaction_cost=transaction_cost, slippage=slippage, risk_free_rate=risk_free_rate)

    test_results = {
        'type':type,
        'best_model':clf,
        'best_cross_val_score':f1_score(y_test, y_pred, sample_weight=return_based_sample_weights_test),
        'recall': recall_score(y_test, y_pred, sample_weight=return_based_sample_weights_test),
        'precision':precision_score(y_test, y_pred, sample_weight=return_based_sample_weights_test), 
        'accuracy':accuracy_score(y_test, y_pred, sample_weight=return_based_sample_weights_test),
        'run_time':t1-t0,
        'train_test':'Test',
        'symbol':symbol,
        'isDrifted':isDrifted,
        'ks_stat':ks_stat,
        'ks_p_value':p_value,
        'cum_rtn': bt_metrics['cum_rtn'],
        'annualized_rtn': bt_metrics['annualized_rtn'], 
        'sharpe_ratio':bt_metrics['sharpe_ratio'],
        'performed_on':datetime.datetime.now(),
        'event_st_date': y_test.index.min(),
        'event_en_date': y_test.index.max(),
    }
    model_metrics = model_metrics.append(test_results, ignore_index = True)

    return clf, model_metrics

def get_one_model(ticker, method='trend_labeling', config=config.pgSecrets):

    """
    get the meta-labelled model for one ticker
    :param method: (str) defines labeling method for trends, default trend_labeling
        'trend_labeling' - trend labeling method
        'sim_ma' - simple moving average crossover
        'exp_ma' - exponential moving average crossover
    :return clf, model_metrics: (sklearn classification model object, dataframe): tuple of model object and a dataframe of modeling metrics
    
    """

    ticker_df = gd.getTicker_Daily(ticker, config=config)
    ticker_df.index = pd.to_datetime(ticker_df.tranx_date)
    index_SPY = get_index(ticker_df, 'SPY')
    index_SPY.index = pd.to_datetime(index_SPY.tranx_date)
    ticker_df = ticker_df[ticker_df.index <= index_SPY.index.max()] #make sure ticker and index has same length

    # preprocessing
    dollar_bars = normalizing(ticker_df)

    # add features
    if method == 'sim_ma':
        dollar_bars = ma_crossover_labeling(dollar_bars, exp=False)
    elif method == 'exp_ma':
        dollar_bars = ma_crossover_labeling(dollar_bars, exp=True)
    else:
        dollar_bars = trend_labeling(dollar_bars)
    events, dollar_bars = meta_labeling(dollar_bars)
    dollar_bars = features_momentum(dollar_bars)
    dollar_bars = features_volatility(dollar_bars)
    dollar_bars = features_log_returns(dollar_bars)
    dollar_bars = features_serial_correlation(dollar_bars)

    # add features - relative strength to SPY
    dollar_bars_SPY = bars.transform_index_based_on_dollarbar(dollar_bars, index_SPY)
    dollar_bars = features_SPY_RS(dollar_bars, dollar_bars_SPY)

    # # add features - relative strength to COMP
    # index_COMP = get_index(ticker_df, 'COMP')
    # dollar_bars_COMP = bars.transform_index_based_on_dollarbar(dollar_bars, index_COMP)
    # dollar_bars = features_COMP_RS(dollar_bars, dollar_bars_COMP)
   
    # get the output model and train test metrics
    clf, model_metrics = modeling(ticker, events, dollar_bars)
    model_metrics['rawdata_st_date'] = ticker_df.index.min()
    model_metrics['rawdata_en_date'] = ticker_df.index.max()
    model_metrics['SPY_st_date'] = index_SPY.index.min()
    model_metrics['SPY_en_date'] = index_SPY.index.max()
    model_metrics['method'] = method

    return clf, model_metrics

def get_multiple_models(ticker_lst, method='trend_labeling', config=config.pgSecrets):
    """
    :param ticker_lst: (list) of ticker symbols

    :return clfs, model_metrics_df: (dict, dataframe): tuple of dictionary of symbol:classifiers k,v pairs \
        and a dataframe of modeling metrics
    """
    clfs = {}
    model_metrics_df = pd.DataFrame()
    ln = str(len(ticker_lst))

    for i, ticker in enumerate(ticker_lst):
        print('Processing {}/{} {}...'.format(str(i+1), ln, ticker))
        clf, model_metrics = get_one_model(ticker, method=method, config=config)
        clfs[ticker] = clf
        model_metrics_df = pd.concat([model_metrics_df, model_metrics], ignore_index=True)
        print('Modeling completed {}/{} {}'.format(str(i+1), ln, ticker))

    return clfs, model_metrics_df