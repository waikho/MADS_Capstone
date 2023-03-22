import numpy as np
import pandas as pd

import tqdm

# Snippet 2.4, page 39, The Symmetric CUSUM Filter.
def cusum_filter(raw_time_series, threshold, signal=None, time_stamps=True):
    """
    Snippet 2.4, page 39, The Symmetric Dynamic/Fixed CUSUM Filter.
    The CUSUM filter is a quality-control method, designed to detect a shift in the
    mean value of a measured quantity away from a target value. The filter is set up to
    identify a sequence of upside or downside divergences from any reset level zero.
    We sample a bar t if and only if S_t >= threshold, at which point S_t is reset to 0.
    One practical aspect that makes CUSUM filters appealing is that multiple events are not
    triggered by raw_time_series hovering around a threshold level, which is a flaw suffered by popular
    market signals such as Bollinger Bands. It will require a full run of length threshold for
    raw_time_series to trigger an event.
    Once we have obtained this subset of event-driven bars, we will let the ML algorithm determine
    whether the occurrence of such events constitutes actionable intelligence.
    Below is an implementation of the Symmetric CUSUM filter.
    Note: As per the book this filter is applied to closing prices but we extended it to also work on other
    time series such as volatility.

    :param raw_time_series: (series) of close prices (or other time series, e.g. volatility).
    :param threshold: (float or pd.Series) when the abs(change) is larger than the threshold, the function captures
    it as an event, can be dynamic if threshold is pd.Series
    :param signal: (str) 'buy' for only buy signals, 'sell' for only sell signals, None for both signals
    :param time_stamps: (bool) default is to return a DateTimeIndex, change to false to have it return a list.
    :return: (datetime index vector) vector of datetimes when the events occurred. This is used later to sample.
    """

    t_events = []
    t_events_pos = []
    t_events_neg = []
    s_pos = 0
    s_neg = 0

    # log returns
    raw_time_series = pd.DataFrame(raw_time_series)  # Convert to DataFrame
    raw_time_series.columns = ['price']
    raw_time_series['log_ret'] = raw_time_series.price.apply(np.log).diff() #difference in log of price 

    # check if threshold is a single value or a pd series
    # make it as a new column
    if isinstance(threshold, (float, int)):
        raw_time_series['threshold'] = threshold
    elif isinstance(threshold, pd.Series):
        raw_time_series.loc[threshold.index, 'threshold'] = threshold
    else:
        raise ValueError('threshold is neither float nor pd.Series!')
    
    raw_time_series = raw_time_series.iloc[1:]  # Drop first na values

    # Get event time stamps for the entire series
    for tup in raw_time_series.itertuples():
        thresh = tup.threshold
        pos = float(s_pos + tup.log_ret) #cumulative returns
        neg = float(s_neg + tup.log_ret)
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -thresh:
            s_neg = 0
            t_events.append(tup.Index)
            t_events_pos.append(tup.Index)

        elif s_pos > thresh:
            s_pos = 0
            t_events.append(tup.Index)
            t_events_neg.append(tup.Index)       

    # Return DatetimeIndex or list
    if time_stamps:
        t_events = pd.DatetimeIndex(t_events)
        t_events_pos = pd.DatetimeIndex(t_events_pos)
        t_events_neg = pd.DatetimeIndex(t_events_neg)
    
    if signal == 'buy':
        return t_events_pos
    elif signal == 'sell':
        return t_events_neg
    else:
        return t_events

def getTEvents(gRaw, h):
    """
    Original function by Lopez 
    Snippet 2.4
    if the abs price change is greater than the threshold, it captures as an event 
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()

    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos+diff.loc[i]), min(0, sNeg+diff.loc[i])
        if sNeg < -h:
            sNeg=0
            tEvents.append(i)
        elif sPos > h:
            sPos=0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

