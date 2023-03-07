import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from xgboost import XGBRegressors

#https://pypi.org/project/timeseries-cv/

data = yf.download(
            tickers = "SPY AAPL ALGM",  # list of tickers
            period = "1y",         # time period
            interval = "60m",       # trading interval
            ignore_tz = True,      # ignore timezone when aligning data from different exchanges?
            prepost = False,       # download pre/post market hours data?
            group_by='ticker')
df = pd.DataFrame(data).stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)


scaler=RobustScaler() #XGBRegressor()
model=SVC()
param_search = {'kernel':('linear', 'rbf'), 'C':[1, 10]}  #{'max_depth' : [3, 5]}
tss = TimeSeriesSplit(n_splits=10)
scoring = {"Acc": "accuracy", "F1": 'f1'}


def pipe(scalar=scaler, model=model):
    """
    build pipeline
    """

    pipe = Pipeline([('scaler', scalar), ('model', model)])
    
    return pipe


def GridSearch(pipe, param_search, tss, scoring, X_train, y_train):
    """
    perform grid search
    """

    GS = GridSearchCV(
        estimator=pipe, 
        param_grid=param_search,
        scoring=scoring,
        n_jobs=-1,
        cv=tss
            )
    GS_result = GS.fit(X_train, y_train)
    
    return GS_result


def traintestsplit(df, train_start, train_end, test_start, test_end):
    """
    one day split train, validate, test set
    """
    #todo


    return X_train, y_train, X_test, y_test

# for back testing
def walk_forward_process(pipe, param_search, tss, scoring):
    """
    walk forward backtest on results
    """
    date_list = []
    for trading_day in date_list:
        X_train, y_train, X_test, y_test =traintestsplit()
        estimator = GridSearch(pipe, param_search, tss, scoring, X_train, y_train)
        prediction = estimator.predict() # "win_size" of predicted y

    return None

# Define a function to create the LSTM model
def create_model(units=32, optimizer ='adam'):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

#create the pipeline to scale the data and train the LSTM model
pipeNN=Pipeline([('scalar', StandardScaler()), ('lstm', KerasClassifier(build_fn=create_model))])

# Define the hyperparameters to tune with grid search
paramsNN = {
    'lstm__units':[32,64],
    'lstm__optimizer':['adam', 'rmsprop']
}

#perform grid search to find the best params
grid_serach = GridSearchCV(
    pipeNN,
    param_grid=paramsNN,
    cv=3,
    n_jobs=-1
    )
grid_serach.fit(X_train, y_train)

# print best params and accuracy score
print('Best params: ', grid_serach.best_params_)
print('Accuracy: ', grid_serach.best_score_)