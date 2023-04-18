#All imports
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from datetime import date
from datetime import time
from datetime import timedelta
from termcolor import colored
from psycopg.rows import dict_row
from psycopg.types.json import Json
import alpaca.trading
import alpaca
import pytz
import config
import psycopg
import threading
import random
import time as systime
import numpy as np

