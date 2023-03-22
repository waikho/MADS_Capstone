import yfinance as yf
import pandas as pd

#download data from yfinance

def get_yf_data(tickers= "SPY AAPL ALGM DNOW", period='1y', interval='1d', ignore_tz = True,prepost = False):
    data = yf.download(
                tickers = tickers,  # list of tickers
                period = period,         # time period
                interval = interval, #"60m",       # trading interval
                ignore_tz = True,      # ignore timezone when aligning data from different exchanges?
                prepost = False,       # download pre/post market hours data?
                group_by='ticker',
                )

    df = pd.DataFrame(data).stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)

    return df
