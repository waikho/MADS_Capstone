#import config
import psycopg
import pandas as pd
import numpy as np
from psycopg.rows import dict_row

import matplotlib
import matplotlib.pyplot as plt
import seaborn

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


#create secrete
def pgDictToConn(secretDict):
    pgStrs = []
    for key in secretDict:
        pgStrs.append('{}={}'.format(key, secretDict[key]))
    return ' '.join(pgStrs)


#get daily prices
def getDailyPrices(symbol_lst, pgConnStr):
    #df = pd.DataFrame()
    dfs = []

    with psycopg.connect(pgConnStr) as conn:
        with conn.cursor() as cur:
            for symbol in symbol_lst:
                stmt = '''SELECT tranx_date, close FROM alpaca_daily_selected WHERE symbol = %s'''
                data = (symbol, )
                cur.execute(stmt, data)
                rows = cur.fetchall()
                ticker_df = pd.DataFrame(rows, columns=['tranx_date',symbol])
                #ticker_df.drop_duplicates(subset=['tranx_date']).set_index('tranx_date')   ###alpaca data may have duplicate prices for the same datetime###
                #df = pd.concat([df, ticker_df[symbol]], axis=1)
                #below are suggested by CGPT
                ticker_df.drop_duplicates(subset=['tranx_date'], inplace=True)
                ticker_df.set_index('tranx_date', inplace=True)
                dfs.append(ticker_df)

                # Combine dataframes using outer join
                combined_df = pd.concat(dfs, axis=1, sort=True)

                # Set index to datetime format
                combined_df.index = pd.to_datetime(combined_df.index)

                # Find earliest and latest dates
                earliest_date = combined_df.index.min()
                latest_date = combined_df.index.max()

                # Reindex to fill missing dates with NaN
                idx = pd.date_range(start=earliest_date, end=latest_date)
                combined_df = combined_df.reindex(idx, method='ffill')
                combined_df = combined_df.reindex(idx, method='bfill')
                
    return combined_df
    #return df
    

#get distinct tickers from daily prices
def getDailyPricesTickersLst(pgConnStr):
    with psycopg.connect(pgConnStr) as conn:
        with conn.cursor() as cur:
            stmt = '''SELECT DISTINCT symbol FROM alpaca_daily_selected'''
            result = cur.execute(stmt).fetchall()
            result = [res[0] for res in result]

    return result


#find clusters of stocks from dailyDF using kmeans and PCA
def findStockClusters(df, n_clusters):
    
    #need to convert all values to numeric values first
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Interpolate missing values using linear interpolation
    df = df.interpolate(method='linear', axis=0).ffill().bfill().fillna(0).drop_duplicates()

    # Fit KMeans clustering model to the dataframe
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df.T)   #to find clusters by columns

    # Get cluster labels for each stock
    labels = kmeans.labels_
    
    # Use PCA to reduce the dataframe to 2 dimensions
    pca = PCA(n_components=2)
    coords = pca.fit_transform(df.transpose())

    # Create a dataframe with the stock symbols, their corresponding cluster label, and their 2D coordinates
    clusters = pd.DataFrame({'symbol': df.columns, 'cluster': labels, 'x': coords[:, 0], 'y': coords[:, 1]})

    return clusters


#to find cointegrated pairs
def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    hr = []
    for i in range(n):
        for j in range(i+1, n):
            df = data[[keys[i], keys[j]]].dropna()
            S1 = df[keys[i]]
            S2 = df[keys[j]]
            result = coint(S1, S2)    #statsmodel built-in cointegration hypothesis test
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue

            #run OLS regression
            ols_model=OLS(S1, S2).fit()
            #get pair's hedge ratio
            hr_pair = ols_model.params[0]
            
            #calculate spread
            spread = np.log(S1) - hr_pair * np.log(S2)
            #spread = S1 - hr_pair * S2   ###changed###

            #adf
            # conduct Augmented Dickey-Fuller test
            adf = adfuller(spread, maxlag = 1)
        
            #if pvalue < 0.05 :    #reject null: there should be cointegration
            #add one condition: adf needs to < -3.435 to confirm stationarity
            if (pvalue < 0.05 and adf[0] < -3.435):    #reject null: there should be cointegration
                pairs.append((keys[i], keys[j]))
                
                #append hr_pair into hr list
                hr.append(hr_pair)
     
    return score_matrix, pvalue_matrix, hr, pairs


#filter out some outlier clusters, generate a dict of good clusters
def findStocksinClusters(clusters, n):
    good_clusters = []
    good_clusters_dict = {}

    for i in range(n):
        if len(clusters[clusters['cluster']==i]) >= 5:
            good_clusters.append(i)

    for cluster in good_clusters:
        good_clusters_dict[cluster] = clusters[clusters['cluster']==cluster]['symbol'].reset_index()['symbol']

    return good_clusters_dict


#plot stock clusters
def plotStockClusters(clusters):
    # Get unique cluster labels
    unique_clusters = np.unique(clusters['cluster'])

    # Create a color map for the clusters
    cmap = plt.get_cmap('viridis', len(unique_clusters))

    # Plot the scatter plot with each cluster colored differently
    fig, ax = plt.subplots(figsize=(8,6))
    for i, cluster in enumerate(unique_clusters):
        x = clusters.loc[clusters['cluster'] == cluster, 'x']
        y = clusters.loc[clusters['cluster'] == cluster, 'y']
        symbol = clusters.loc[clusters['cluster'] == cluster, 'symbol']
        ax.scatter(x, y, label=f"Cluster {cluster}", color=cmap(i))
        #for j, sym in enumerate(symbol):
        #    ax.annotate(sym, (x.iloc[j], y.iloc[j]))

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.legend()
    plt.show()


#plot cointegration heatmap
def cointHeatmap(df):

    # Randomly sample 50 columns
    num_cols_to_sample = min(50, len(df.columns)) # sample up to 50 columns or all columns if there are fewer than 50
    
    if len(df) > 50:
        sampled_df = df.sample(n=num_cols_to_sample, axis=1)
    else:
        sampled_df = df
    
    tickers = sampled_df.columns
    scores, pvalues, hr, pairs = find_cointegrated_pairs(sampled_df)
    
    fig, ax = plt.subplots(figsize=(10,10))
    seaborn.heatmap(pvalues, xticklabels=tickers, yticklabels=tickers, cmap='RdYlGn_r', mask=(pvalues >= 0.05))
    print(pairs)
    return pairs, pvalues, sampled_df


#find pairs, pvalues, sampled_df without plotting heatmap
def find_pairs_pv_df(df):

    # Randomly sample 50 columns
    num_cols_to_sample = min(50, len(df.columns)) # sample up to 50 columns or all columns if there are fewer than 50
    
    if len(df) > 50:
        sampled_df = df.sample(n=num_cols_to_sample, axis=1)
    else:
        sampled_df = df
    
    tickers = sampled_df.columns
    scores, pvalues, hr, pairs = find_cointegrated_pairs(sampled_df)
    
    return pairs, pvalues, sampled_df