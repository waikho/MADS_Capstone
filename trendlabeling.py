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
    window_size_min = 3 # cannot be less than 3 coz demon will become <0 when calculating mean sq errors
    d = {}

    for window_size in range(window_size_min, window_size_max+1):
        d[window_size] = get_one_slope(window_size=window_size, this_y=this_y)

    max_key = max(d, key=lambda k: d[k][0]) #get the key of with the best abs t-value
    one_best_slope = d[max_key][1]
    
    return one_best_slope


def get_trend_scanning_labels(time_series, window_size_max, threshold=0):
    """
    get trend scanning labels on entire time series

    :return list of slope and threshold labels for each t except last window_size_max-1 t
    """
    d = {'slope':[], 'label':[]}

    # rolling window on time series
    for i in range(len(time_series)-window_size_max+1):
        # define window of data
        this_y = (time_series[i:i+window_size_max])
        this_one_best_slope = get_one_best_slope(this_y, window_size_max=window_size_max)
        d['slope'].append(this_one_best_slope)
        d['label'].append(1 if this_one_best_slope >= threshold else (-1 if this_one_best_slope < -threshold else 0))

    return d

