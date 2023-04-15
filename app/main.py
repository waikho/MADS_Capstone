#All imports
import pytz
from datetime import datetime, date, time, timedelta
from dateutil.relativedelta import relativedelta
from data_source import scraper

#Constants
EXCHANGE_OPEN_HOUR = 9
EXCHANGE_OPEN_MIN = 30
EXCHANGE_CLOSE_HOUR = 16
EXCHANGE_CLOSE_MIN = 0
newYorkTz = pytz.timezone("America/New_York") 

#Intial Variables
selected_exchanges = ['AssetExchange.NASDAQ', 'AssetExchange.NYSE', 'AssetExchange.ARCA']
weekends = [6, 7]

#Main function

#Calculate Date Ranges and Adjust for Time Zone
today = date.today()
now = datetime.now(newYorkTz)
last_close = datetime.combine(date.today(), time(hour=EXCHANGE_CLOSE_HOUR, minute=EXCHANGE_CLOSE_MIN, tzinfo=newYorkTz))

if last_close > now:
    last_close = last_close - timedelta(1)
    today = today - timedelta(1)
yesterday = today - timedelta(1)
ten_years_ago = today + relativedelta(years=-10)
dow = today.isoweekday()

if dow in weekends:
    pass
else:
    symbols = scraper.getAllActiveSymbols(selected_exchanges)
    #Minute Level Data at Daily Frequency
    scraper.threadedGetMinuteDataForMultipleStocks(symbols, today, today)
    