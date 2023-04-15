#All imports
import alpaca
import alpaca.trading
import config
import json
import numpy as np
import requests
import psycopg
import pytz
import random
import threading
import time as systime
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from bs4 import BeautifulSoup
from datetime import datetime, time, timedelta
from html.parser import HTMLParser
from psycopg.rows import dict_row
from psycopg.types.json import Json
from pubproxpy import Level, Protocol, ProxyFetcher
from termcolor import colored


#All global variables
newYorkTz = pytz.timezone("America/New_York") 


def getAllActiveSymbols(selected_exchanges):
    trade_client = alpaca.trading.TradingClient(config.api['alpaca']['key'], config.api['alpaca']['secret'])
    assets = trade_client.get_all_assets()
    symbols = [asset.symbol for asset in assets 
           if str(asset.exchange) in selected_exchanges and str(asset.status) != 'AssetStatus.INACTIVE']
    return symbols

def getMinuteDataForStock(symbol, date_from, date_to):
    data = []
    stock_client = StockHistoricalDataClient(config.api['alpaca']['key'], config.api['alpaca']['secret'])
    current_date = date_from
    while current_date <= date_to:
        # NY Exchange Opening Hours - currently 9:30 to 16:00 - may subject to change
        open_time = datetime.combine(current_date, time(hour=config.EXCHANGE_OPEN_HOUR, 
                                                        minute=config.EXCHANGE_OPEN_MIN, tzinfo=newYorkTz))
        close_time = datetime.combine(current_date, time(hour=config.EXCHANGE_CLOSE_HOUR, 
                                                         minute=config.EXCHANGE_CLOSE_MIN,tzinfo=newYorkTz))
        request_params = StockBarsRequest(symbol_or_symbols=[symbol], 
                                      timeframe=TimeFrame.Minute, 
                                      start=open_time,
                                      end=close_time)
        success = False
        delay = random.randint(1, 10)
        while not success:
            try:
                result = stock_client.get_stock_bars(request_params)
                success = True
            except Exception as e:
                print(str(e), ' - retry for single day stock data: {}'.format(symbol))
                print('Wait for {} seconds'.format(delay))
                systime.sleep(delay)
                delay = delay * 2
                if delay >= 1800:
                    delay = 2400 - random.randint(1, 1600)
        if symbol in result.data:
            entries_list = []
            for entry in result.data[symbol]:
                entry_dict = entry.__dict__
                entry_dict['date'] = current_date
                entries_list.append(entry_dict)
            data = data + entries_list
        current_date = current_date + timedelta(1)
    return data 


def threadedGetMinuteDataForStock(symbol, date_from, date_to, result = [], job_id=1, thread_size=50):
    num_days =  (date_to - date_from).days + 1
    if num_days <= thread_size:
        success = False
        delay = random.randint(3, thread_size)
        while not success:
            try:
                result.extend(getMinuteDataForStock(symbol, date_from, date_to))
                print(colored('Completed Single Stock Thread job {} for {}'.format(job_id, symbol), 'red'))
                success = True
            except Exception as e:
                print(str(e), 'retry Single Stock Thread job {} for {}'.format(job_id, symbol))
                print('Wait for {} seconds'.format(delay))
                systime.sleep(delay)
                delay = delay * 2 + random.randint(0, job_id)
                if delay > 1800:
                    delay = 2400 - random.randint(0, 1200)
            
    else:
        split_point = int(num_days/2) - 1
        date_from1 = date_from
        date_to1 = date_from1 +  timedelta(split_point)
        date_from2 = date_to1 + timedelta(1)
        date_to2 = date_to
        result1 = []
        result2 = []
        t1 = threading.Thread(target=threadedGetMinuteDataForStock, name='t{}'.format(job_id*2), 
                              args=(symbol, date_from1, date_to1, result1, job_id*2))
        t1.start()
        t2 = threading.Thread(target=threadedGetMinuteDataForStock, name='t{}'.format(job_id*2+1), 
                              args=(symbol, date_from2, date_to2, result2, job_id*2+1))
        t2.start()

        t1.join()
        t2.join()
        result.extend(result1)
        result.extend(result2)
        print(colored('Completed job {} of all data for {}'.format(job_id, symbol), 'red'))


def updateSingleStockMinuteEntriesToDB(symbol, entries):
    with psycopg.connect(config.pgConnStr) as conn:
        with conn.cursor() as cur:
            tmp_table = 'tmp_{}'.format(symbol.replace('.','_'))
            stmt = '''CREATE TEMP TABLE {} 
                (LIKE alpaca_minute INCLUDING DEFAULTS)
                ON COMMIT DROP'''.format(tmp_table)
            cur.execute(stmt)

            with cur.copy('''COPY {} (symbol, date, datetime, 
                open, close, high, low, trade_count, vol, vwap)
                FROM STDIN'''.format(tmp_table)) as copy:
                for entry in entries: 
                    entry_tuple = (entry['symbol'], entry['date'], entry['timestamp'], 
                            entry['open'], entry['close'], entry['high'], entry['low'],
                            entry['trade_count'], entry['volume'], entry['vwap'])
                    copy.write_row(entry_tuple)

            stmt = '''INSERT INTO alpaca_minute (symbol, date, datetime, 
                open, close, high, low, trade_count, vol, vwap)
                SELECT symbol, date, datetime, 
                open, close, high, low, trade_count, vol, vwap
                FROM {}
                ON CONFLICT(symbol, datetime) DO NOTHING'''.format(tmp_table)
            cur.execute(stmt)
            conn.commit()    


def threadedGetMinuteDataForMultipleStocks(symbols, date_from, date_to, job_id=1, thread_size=200):
    stock_count = len(symbols)
    if  stock_count <= thread_size:
        for symbol in symbols:
            data = []
            success = False
            delay = random.randint(job_id, job_id*2)
            while not success:
                try:
                    threadedGetMinuteDataForStock(symbol, date_from, date_to, data)
                    success = True
                except Exception as e:
                    print(str(e), 'retry job {} for {}'.format(job_id, symbol))
                    print('Wait for {} seconds'.format(delay))
                    systime.sleep(delay)
                    delay = delay * 2
                    if delay > 1000:
                        delay = 500
            success = False
            delay = random.randint(job_id, job_id*2)
            while not success:
                try:
                    updateSingleStockMinuteEntriesToDB(symbol, data)
                    success = True
                    print(colored('Completed minute update for symbol {}'.format(symbol), 'green'))
                except Exception as e:
                    print(str(e), 'retry DB for minute update job {} for symbol {}'.format(job_id, symbol))
                    print('Wait for {} seconds'.format(delay))
                    systime.sleep(delay) 
                    delay = delay * 2
                    if delay > 1000:
                        delay = 500
    else:
        delay = random.randint(1, int(thread_size/2))
        split_point = int(stock_count/2)
        symbols1 = symbols[:split_point]
        symbols2 = symbols[split_point:]
        t1 = threading.Thread(target=threadedGetMinuteDataForMultipleStocks, name='t{}'.format(job_id*2), 
                              args=(symbols1, date_from, date_to, job_id*2))
        t1.start()
        systime.sleep(delay) 
        t2 = threading.Thread(target=threadedGetMinuteDataForMultipleStocks, name='t{}'.format(job_id*2+1), 
                              args=(symbols2, date_from, date_to, job_id*2+1))
        t2.start()

        t1.join()
        t2.join()
        print(colored('Completed minute update job id {}'.format(job_id), 'green'))


#Related to daily data
def getSelectedSymbolsFor10Years():
    with psycopg.connect(config.pgConnStr) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            stmt = '''CREATE TEMP TABLE last_tranx
                ON COMMIT DROP
                AS
                SELECT symbol, max(datetime) AS latest FROM alpaca_minute
                GROUP BY symbol 
                '''
            cur.execute(stmt)
            stmt = '''WITH SI AS 
                (SELECT symbol,
                    CASE RIGHT(info::json->>'Shs Float', 1) 
                        WHEN 'K' THEN 1000 * LEFT(info::json->>'Shs Float', -1)::DECIMAL
                        WHEN 'M' THEN 1000000 * LEFT(info::json->>'Shs Float', -1)::DECIMAL
                        WHEN 'B' THEN 1000000000 * LEFT(info::json->>'Shs Float', -1)::DECIMAL
                        ELSE (info::json->>'Shs Float')::DECIMAL
                    END AS float
                    FROM stock_info
                    WHERE info::json->>'Shs Float' <> '-'
                    AND last_update = ANY(SELECT MAX(last_update) FROM stock_info)
                )
                SELECT AM.symbol FROM alpaca_minute AS AM, last_tranx AS LT, SI
                    
                    WHERE AM.symbol = LT.symbol AND LT.symbol = SI.symbol AND AM.datetime = LT.latest
                    AND AM.close >= 10.0 AND AM.close <= 30.0
                    AND SI.float <= 60000000'''
            result = cur.execute(stmt).fetchall()
            conn.commit()
            return [row['symbol'] for row in result]
        
def getDayLevelDataForStock(symbol, date_from, date_to):
    data = []
    stock_client = StockHistoricalDataClient(config.api['alpaca']['key'] , config.api['alpaca']['secret'])
    open_time = datetime.combine(date_from, time(hour=config.EXCHANGE_OPEN_HOUR, 
                                                 minute=config.EXCHANGE_OPEN_MIN, tzinfo=newYorkTz))
    close_time = datetime.combine(date_to, time(hour=config.EXCHANGE_CLOSE_HOUR, 
                                                minute=config.EXCHANGE_CLOSE_HOUR, tzinfo=newYorkTz))
    request_params = StockBarsRequest(symbol_or_symbols=[symbol], 
                                  timeframe=TimeFrame.Day, 
                                  start=open_time,
                                  end=close_time)
    result = stock_client.get_stock_bars(request_params)
    if symbol in result.data:
        data = []
        for entry in result.data[symbol]:
            entry_dict = entry.__dict__
            data.append(entry_dict)
    return data

def updateStockDailyEntriesToDB(symbol, entries):
    with psycopg.connect(config.pgConnStr) as conn:
        with conn.cursor() as cur:
            tmp_table = 'tmp_daily_{}'.format(symbol.replace('.','_'))
            stmt = '''CREATE TEMP TABLE {} 
                (LIKE alpaca_daily_selected INCLUDING DEFAULTS)
                ON COMMIT DROP'''.format(tmp_table)
            cur.execute(stmt)

            with cur.copy('''COPY {} (symbol, tranx_date, 
                open, close, high, low, trade_count, vol, vwap)
                FROM STDIN'''.format(tmp_table)) as copy:
                for entry in entries: 
                    entry_tuple = (entry['symbol'], entry['timestamp'], 
                            entry['open'], entry['close'], entry['high'], entry['low'],
                            entry['trade_count'], entry['volume'], entry['vwap'])
                    copy.write_row(entry_tuple)

            stmt = '''INSERT INTO alpaca_daily_selected (symbol, tranx_date, 
                open, close, high, low, trade_count, vol, vwap)
                SELECT symbol, tranx_date,
                open, close, high, low, trade_count, vol, vwap
                FROM {}
                ON CONFLICT(symbol, tranx_date) DO NOTHING'''.format(tmp_table)
            cur.execute(stmt)
            conn.commit()   

def threadedGetDailyDataForMultipleStocks(symbols, date_from, date_to, job_id=1, thread_size=50):
    stock_count = len(symbols)
    if  stock_count <= thread_size:
        for symbol in symbols:
            data = []
            success = False
            delay = random.randint(job_id, job_id*2)
            while not success:
                try:
                    data = getDayLevelDataForStock(symbol, date_from, date_to)
                    success = True
                except Exception as e:
                    print(str(e), 'retry job {} for {}'.format(job_id, symbol))
                    print('Wait for {} seconds'.format(delay))
                    systime.sleep(delay)
                    delay = delay * 2
                    if delay > 60:
                        delay = 60
            success = False
            delay = random.randint(job_id, job_id*2)
            while not success:
                try:
                    updateStockDailyEntriesToDB(symbol, data)
                    success = True
                    print(colored('Completed for symbol {}'.format(symbol), 'green'))
                except Exception as e:
                    print(str(e), 'retry DB for job {}'.format(job_id))
                    print('Wait for {} seconds'.format(delay))
                    systime.sleep(delay) 
                    delay = delay * 2
                    if delay > 60:
                        delay = 60
    else:
        delay = random.randint(1, int(thread_size/4))
        split_point = int(stock_count/2)
        symbols1 = symbols[:split_point]
        symbols2 = symbols[split_point:]
        t1 = threading.Thread(target=threadedGetDailyDataForMultipleStocks, name='t{}'.format(job_id*2), 
                              args=(symbols1, date_from, date_to, job_id*2))
        t1.start()
        systime.sleep(delay) 
        t2 = threading.Thread(target=threadedGetDailyDataForMultipleStocks, name='t{}'.format(job_id*2+1), 
                              args=(symbols2, date_from, date_to, job_id*2+1))
        t2.start()

        t1.join()
        t2.join()
        print(colored('Completed job id {}'.format(job_id), 'green'))


#Related to stock info
def getNewProxy():
    while True:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36',
        }
        params = {
            'api': config.api['pubproxy']['key'],
            'level': 'elite',
        }
        result = requests.get('http://pubproxy.com/api/proxy', params=params,  headers=headers).json()
        if "data" in result:
            proxyString = result['data'][0]['type'] + '://' + result['data'][0]['ipPort']
            proxies = {'http': proxyString}
            return proxies
        else:
            systime.sleep(2)
            continue

def getStockInfoPage(symbol, proxies):
    url = "https://finviz.com/quote.ashx?t={}&ty=c&p=d&b=1".format(symbol)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'
    }
    response = requests.get(url,
                          proxies=proxies,
                          headers=headers)
    response.raise_for_status()
    return response.content

def ParseFieldsFromInfoPage(pageHTML, fields):
    soup = BeautifulSoup(pageHTML, features="html.parser")
    table = soup.find("table", {"class": "snapshot-table2"})
    cells = table.find_all('td')
    infoDict = dict.fromkeys(fields)
    for cell in cells:
        if cell.string in fields:
            infoDict[cell.string] = cell.next_sibling.string
    return infoDict

def ScrapInfoForStock(symbol, fields, proxies):
    htmlResult = getStockInfoPage(symbol, proxies)
    return ParseFieldsFromInfoPage(htmlResult, fields)

def batchScrapInfoForStocks(symbols, fields, proxyRefreshRate=config.api['pubproxy']['refresh_rate']):
    count = 0
    all_info = {}
    failed_symbols = []
    proxies = None
    delay = 0
    for symbol in symbols:
        delay = random.randint(30, 60)
        if count % proxyRefreshRate == 0:
            try:
                proxies = getNewProxy()
                print('New Proxy: {}'.format(proxies))
            except:
                print('Get New Proxy Failed. Use Previous Proxy')
        success = False
        while not success:
            try:
                info = ScrapInfoForStock(symbol, fields, proxies)
                all_info[symbol] = info
                success = True
                print('Got info for symbol {}'.format(symbol))
            except Exception as e:
                if str(e)[:3] == '404':
                    failed_symbols.append(symbol)
                    success = True
                    print('Symbol Error for {}, skipped'.format(symbol))
                else:
                    print('Failed to get info for symbol {}'.format(symbol))
                    if delay >= 900:   
                        delay = random.randint(60, 120)
                        proxies = getNewProxy()
                        print('New Proxy: {}'.format(proxies))
                    else:
                        time.sleep(delay)
                        print('Wait for {}s before retrying for {}'.format(delay, symbol))
                        delay = delay * 2
                
        count = count + 1
        systime.sleep(random.randint(2, 7))
    return all_info, failed_symbols



def ThreadedScrapInfoForStocks(symbols, fields, info={}, failed_list=[], job_id=1,  
                               proxyRefreshRate=config.api['pubproxy']['refresh_rate'], thread_size=180):
    if(len(symbols) <= thread_size):
        new_info, new_failed_list = batchScrapInfoForStocks(symbols, fields, proxyRefreshRate=proxyRefreshRate)
        info.update(new_info)
        failed_list.extend(new_failed_list)
        print(colored('Completed for job id {}'.format(job_id), 'red'))
    else:
        split_point = int(len(symbols)/2)
        sym1 = symbols[:split_point]
        sym2 = symbols[split_point:]
        list1 = []
        list2 = []
        info1 = {}
        info2 = {}
        t1 = threading.Thread(target=ThreadedScrapInfoForStocks, name='t{}'.format(job_id*2), args=(sym1, fields, info1, list1, job_id*2))
        t2 = threading.Thread(target=ThreadedScrapInfoForStocks, name='t{}'.format(job_id*2+1), args=(sym2, fields, info2, list2, job_id*2+1))
        t1.start()
        delay = thread_size
        systime.sleep(delay)
        t2.start()
        t1.join()
        t2.join()
        info.update(info1)
        info.update(info2)
        failed_list.extend(list1)
        failed_list.extend(list2)

def updateStockInfoToDB(entries):
    with psycopg.connect(config.pgConnStr) as conn:
        with conn.cursor() as cur:
            stmt = '''CREATE TEMP TABLE temp_stock_info
                (LIKE stock_info INCLUDING DEFAULTS)
                ON COMMIT DROP'''
            cur.execute(stmt)

            with cur.copy('''COPY temp_stock_info (symbol, info)
                FROM STDIN''') as copy:
                for symbol, info_dict in entries.items(): 
                    entry_tuple = (symbol, json.dumps(info_dict))
                    copy.write_row(entry_tuple)

            stmt = '''INSERT INTO stock_info (symbol, info, last_update)
                SELECT symbol, info, last_update
                FROM temp_stock_info
                ON CONFLICT(symbol, last_update) DO NOTHING'''
            cur.execute(stmt)
            conn.commit()    

def updateIgnoreListToDB(ignore_list):
    with psycopg.connect(config.pgConnStr) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            for symbol in ignore_list:
                stmt = '''INSERT INTO stock_info_ignore_list(symbol) VALUES(%s)
                ON CONFLICT DO NOTHING'''
                data = (symbol,)
                cur.execute(stmt, data)
            conn.commit()

def getStockInfo():
    with psycopg.connect(config.pgConnStr) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            stmt = '''SELECT symbol, info FROM stock_info
                WHERE last_update = ANY(
                    SELECT MAX(last_update) FROM stock_info)
                ORDER BY symbol'''
            result = cur.execute(stmt).fetchall()
            conn.commit()
    return result

def getStockInfoTickerList():
    with psycopg.connect(config.pgConnStr) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            stmt = '''SELECT DISTINCT symbol
                FROM alpaca_minute
                WHERE symbol NOT IN (SELECT symbol FROM stock_info_ignore_list)
                ORDER BY symbol'''
            result = cur.execute(stmt).fetchall()
            conn.commit()
            return [row['symbol'] for row in result]
        
def stockInfoUpdate(fields):
    symbols = getStockInfoTickerList()
    info = {}
    failed_list = []
    ThreadedScrapInfoForStocks(symbols, fields, info, failed_list)
    updateStockInfoToDB(info)
    updateIgnoreListToDB(failed_list)