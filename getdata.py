import yfinance as yf
import pandas as pd
import config
import psycopg
import pytz
import numpy as np
from psycopg.rows import dict_row


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


def pgDictToConn(secretDict):
    pgStrs = []
    for key in secretDict:
        pgStrs.append('{}={}'.format(key, secretDict[key]))
    return ' '.join(pgStrs)


def getTicker(symbol):
    #Init
    pgConnStr = pgDictToConn(config.pgSecrets)
    with psycopg.connect(pgConnStr) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            stmt = '''SELECT * FROM alpaca_minute WHERE symbol = %s'''
            data = (symbol, )
            result = cur.execute(stmt, data).fetchall()
            desc = cur.description
            cols = [col[0] for col in desc]
            conn.commit()
    df = pd.DataFrame(data=result, columns=cols)
#     df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
    df['datetime'] = df['datetime'].dt.tz_convert('America/New_York')
    return df #pd.DataFrame(data=result, columns=cols)