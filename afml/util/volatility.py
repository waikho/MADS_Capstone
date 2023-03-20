import pandas as pd



def getDailyVol(close, span=100):
    # daily vol, reindexed to close 
    """
    snippet 3.1 Daily Volatility Estimates
    Apply a span of span days to an exponentially weighted moving std dev 
    In other words, it represents the deviation of the points (used to calculate the avg) around the exponentially weighted daily returns average
    the higher means the last 'span' periods returns are more spread out from the average
    """
    # take a series of prior days, and then find the inserting index loc on the original series
    df = close.index.searchsorted(close.index-pd.Timedelta(days=1)) 

    # remove the index  = 0
    df = df[df>0] 

    # get the begin and end date for price comparison
    df = (pd.Series(close.index[df-1], index=close.index[close.shape[0]-df.shape[0]:]))

    df = close.loc[df.index]/close.loc[df.array].array-1 #daily returns
    df = df.ewm(span=span).std() #exponentially weighted moving std dev 

    return df