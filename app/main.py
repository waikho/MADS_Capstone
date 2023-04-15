#All imports
import config
import pytz
import helpers
from datetime import datetime, date, time, timedelta
from dateutil.relativedelta import relativedelta
from data_source import scraper


newYorkTz = pytz.timezone("America/New_York") 

#Intial Variables
notification_recipients = ["horris@umich.edu", "choichao@umich.edu", "wkho@umich.edu"]
selected_exchanges = ['AssetExchange.NASDAQ', 'AssetExchange.NYSE', 'AssetExchange.ARCA']
benchmarks = ['COMP', 'SPY']
weekends = [6, 7]

#Main function

#Calculate Date Ranges and Adjust for Time Zone
today = date.today()
now = datetime.now(newYorkTz)
last_close = datetime.combine(date.today(), time(hour=config.EXCHANGE_CLOSE_HOUR, minute=config.EXCHANGE_CLOSE_MIN, tzinfo=newYorkTz))

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
    helpers.emailNotification(notification_recipients, 
                              "KCT Capital - Daily Minute Data Pipeline", 
                              "Daily Minute Data for {} is ready".format(today))
    #Stock Information

    #10 Year Daily Data of Selected Stocks
    selected_symbols = scraper.getSelectedSymbolsFor10Years() + benchmarks
    