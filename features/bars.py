
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math



def get_dollar_bars(time_bars, dollar_threshold, interval): #function credit to Max Bodoia

    """
    source of this function: https://davidzhao12.medium.com/advances-in-financial-machine-learning-for-dummies-part-1-7913aa7226f5

    : param interval: 'd' if using daily data, 'm' if using 1minute data
    """

    # initialize an empty list of dollar bars
    dollar_bars = []

    # initialize the running dollar volume at zero
    running_volume = 0

    # initialize the running high and low with placeholder values
    running_high, running_low = 0, math.inf

    # for each time bar...
    for i in range(len(time_bars)):

        # get the timestamp, open, high, low, close, and volume of the next bar
        next_close, next_high, next_low, next_open, next_timestamp, next_volume = [time_bars[i][k] for k in ['close', 'high', 'low', 'open', 'time', 'vol']]

        # get the midpoint price of the next bar (the average of the open and the close)
        midpoint_price = ((next_open) + (next_close))/2

        # get the approximate dollar volume of the bar using the volume and the midpoint price
        dollar_volume = next_volume * midpoint_price

        # update the running high and low
        running_high, running_low = max(running_high, next_high), min(running_low, next_low)

        # if the next bar's dollar volume would take us over the threshold...
        if dollar_volume + running_volume >= dollar_threshold:

            # set the timestamp for the dollar bar as the timestamp at which the bar closed (i.e. one minute after the timestamp of the last minutely bar included in the dollar bar)
            bar_timestamp = next_timestamp + timedelta(days=1 if interval == 'd' else 0, minutes=1 if interval =='m' else 0)
            
            # add a new dollar bar to the list of dollar bars with the timestamp, running high/low, and next close
            dollar_bars += [{'timestamp': bar_timestamp, 'open': next_open, 'high': running_high, 'low': running_low, 'close': next_close}]

            # reset the running volume to zero
            running_volume = 0

            # reset the running high and low to placeholder values
            running_high, running_low = 0, math.inf

        # otherwise, increment the running volume
        else:
            running_volume += dollar_volume

    # return the list of dollar bars
    return dollar_bars


