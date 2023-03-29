from sklearn.linear_model import LinearRegression
import numpy as np


def get_one_slope(window_size, this_y): 
    """
    Linear regression to get the t-value and slope based on a defined window size
    """   

    # # define window of data
    this_x = np.arange(1,window_size+1).reshape((-1,1))
    trimmed_this_y = (this_y[0:window_size])



    # fit linear regression model
    this_reg = LinearRegression().fit(this_x, trimmed_this_y)

    # get slope and intercept
    this_slope = this_reg.coef_[0]
    this_intercept = this_reg.intercept_

    # get the residuals of each data point from the regression line
    this_residuals = trimmed_this_y - this_reg.predict(this_x)

    # calculate the std error of the slope estimate
    this_mse = np.sum(this_residuals**2)/(len(this_x)-2) #mean sq error of the residuals
    this_se_slope = np.sqrt(this_mse/np.sum((this_x-np.mean(this_x))**2)) #std error of the slope estimate

    #calculate t-value
    this_t = this_slope/0.00000000001 if this_se_slope == 0 else this_se_slope 
    this_t_abs = np.abs(this_t)

    return this_t_abs, this_slope


def get_one_best_slope(this_y, window_size_max):
    """
    most significant trend observed in the past for a single time step, 
    out of multiple possible look-forward periods from window_size_min to window_size_max
    
    """
    # window size cannot be less than 3 coz demon will become <0 when calculating mean sq errors
    #window_size_min = 3 
    #window_size_max = 3 if window_size_max < 3 else window_size_max
    d = {}

    #for window_size in range(window_size_min, window_size_max+1):
    for i in range(3, window_size_max):
        d[i] = get_one_slope(window_size=i, this_y=this_y)

    max_key = max(d, key=lambda k: d[k][0]) #get the key of with the best abs t-value
    one_best_slope = d[max_key][1]
    
    return one_best_slope


def get_trend_scanning_labels(time_series, window_size_max, threshold=0, side='both'):
    """
    get trend scanning labels on entire time series

    :param time_series: numpy list of prices
    :param window_size_max: window size to use to calculate most signifcant slope
    :param threshold: threshold to define as no trend, value between [0,1]
    :param opp_sign_ct: # of opposite sign require before determining as trend change
    :param side: define which trend to identify 'up', 'down', or 'both'
    :return dict of slope and threshold labels for each t except last window_size_max-1 t
    """
    d = {'slope':[], 'label':[], 
        #  'isEvent':[]
        }
    arr = 0

    # rolling window on time series
    for i in range(window_size_max, len(time_series)+1):
    #for i in range(len(time_series)-window_size_max+1):
        # define window of data
        this_y = (time_series[i-window_size_max:i])
        this_one_best_slope = get_one_best_slope(this_y, window_size_max=window_size_max)
        d['slope'].append(this_one_best_slope)
        d['label'].append(1 if this_one_best_slope >= threshold else (-1 if this_one_best_slope < -threshold else 0))
    

    return d



# import getdata as gd
# df = gd.get_yf_data(tickers= "SPY AAPL ALGM DNOW", period='1y', interval='1d')
# df = df[df['Ticker'] == 'ALGM']

# # convert Adj Close to numpy
# time_series = df['Adj Close'].to_numpy()

# # define window size
# window_size_max = 7

# #define threshold 
# threshold = 0.0

# opp_sign_ct=2

# # get trend scanning labels
# label_output = get_trend_scanning_labels(time_series=time_series, window_size_max=window_size_max, threshold=threshold)

# # opp_sign_ct = 2
# # # a = np.array([1,1,-1,-2,3,-4,5])
# # # asign = np.sign(a)
# # # signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
# # # signchange

# # lst = np.array([1,1,-1,-1,3,-4,5])
# # i = 2
# # arr = np.sign(lst[i])

# # np.all(arr == arr[0])
# # lst[i]

lst = [0,1,2]
lst[0:0]