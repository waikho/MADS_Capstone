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

#download data from alpaca db

def pgDictToConn(secretDict):
    
    pgStrs = []
    for key in secretDict:
        pgStrs.append('{}={}'.format(key, secretDict[key]))
    return ' '.join(pgStrs)


def getTicker_minute(symbol, config=config.pgSecrets):
    #Init
    pgConnStr = pgDictToConn(config)
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


def getTickerList_minute(config=config.pgSecrets):
    pgConnStr = pgDictToConn(config)
    with psycopg.connect(pgConnStr) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            stmt = '''SELECT DISTINCT symbol FROM alpaca_minute ORDER BY symbol'''
            result = cur.execute(stmt).fetchall()
            conn.commit()
            return [row['symbol'] for row in result]
        

def getTicker_Daily(symbol, config=config.pgSecrets):
    #Init
    pgConnStr = pgDictToConn(config)
    with psycopg.connect(pgConnStr) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            stmt = '''SELECT * FROM alpaca_daily_selected WHERE symbol = %s'''
            data = (symbol, )
            result = cur.execute(stmt, data).fetchall()
            desc = cur.description
            cols = [col[0] for col in desc]
            conn.commit()
    df = pd.DataFrame(data=result, columns=cols)
    df['tranx_date'] = df['tranx_date'].dt.tz_convert('America/New_York')
    return df 

df = getTicker_Daily('AAC')


def getTickerList_Daily(config=config.pgSecrets):
    pgConnStr = pgDictToConn(config)
    with psycopg.connect(pgConnStr) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            stmt = '''SELECT DISTINCT symbol FROM alpaca_daily_selected ORDER BY symbol'''
            result = cur.execute(stmt).fetchall()
            conn.commit()
            return [row['symbol'] for row in result]
        
        
def getFilteredTickerList_minute(lowest_price=0.0, highest_price=999999999,config=config.pgSecrets):
    pgConnStr = pgDictToConn(config)
    with psycopg.connect(pgConnStr) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            stmt = '''CREATE TEMP TABLE last_tranx 
                ON COMMIT DROP
                AS
                SELECT symbol, max(datetime) AS latest FROM alpaca_minute
                GROUP BY symbol 
                '''
            cur.execute(stmt)
            stmt = '''SELECT AM.symbol FROM alpaca_minute AS AM, last_tranx AS LT
                WHERE AM.symbol = LT.symbol AND AM.datetime = LT.latest
                AND AM.close >= %s AND AM.close <= %s'''
            data = (lowest_price, highest_price)
            result = cur.execute(stmt, data).fetchall()
            conn.commit()
            return [row['symbol'] for row in result]
        
        
def getFilteredTickerList_Daily(lowest_price=0.0, highest_price=999999999,config=config.pgSecrets):
    pgConnStr = pgDictToConn(config)
    with psycopg.connect(pgConnStr) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            stmt = '''CREATE TEMP TABLE last_tranx 
                ON COMMIT DROP
                AS
                SELECT symbol, max(tranx_date) AS latest FROM alpaca_daily_selected
                GROUP BY symbol
                '''
            cur.execute(stmt)
            stmt = '''SELECT AM.symbol FROM alpaca_daily_selected AS AM, last_tranx AS LT
                WHERE AM.symbol = LT.symbol AND AM.tranx_date = LT.latest
                AND AM.close >= %s AND AM.close <= %s'''
            data = (lowest_price, highest_price)
            result = cur.execute(stmt, data).fetchall()
            conn.commit()
            return [row['symbol'] for row in result]