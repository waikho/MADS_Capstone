import db_config
import pandas as pd
import psycopg
from psycopg.rows import dict_row

def getTicker_Daily(symbol, pgConnStr=db_config.pgConnStr):
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


def getTickerList_Daily(pgConnStr=db_config.pgConnStr):
    with psycopg.connect(pgConnStr) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            stmt = '''SELECT DISTINCT symbol FROM alpaca_daily_selected ORDER BY symbol'''
            result = cur.execute(stmt).fetchall()
            conn.commit()
            return [row['symbol'] for row in result]
        
def getFilteredTickerList_Daily(lowest_price=0.0, highest_price=999999999, pgConnStr=db_config.pgConnStr):
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