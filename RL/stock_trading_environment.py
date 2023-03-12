import os
import gym
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller

from gym import spaces
from enum import Enum

class Positions(Enum):   #Python's enum class
    Short = 0
    Flat = 1
    Long = 2


class StockTradingEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, window_size, trade_period):
        super().__init__()

        self.trade(data)
        
        self.window_size = window_size
        self.trade_period = trade_period

        # Actions: SHORT(0), FLAT(1), LONG(2)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
          low=-np.inf, high=np.inf, shape=(window_size, 1), dtype=np.float16)   #a box as observation space

    def test(self):
        print('testtesttest')
    
    def trade(self, data):
        S1 = data.iloc[:, 0]
        S2 = data.iloc[:, 1]
        _score, self.pvalue, _ = coint(S1, S2)
        
        data['ratios'] = S1/S2
        ma1 = data['ratios'].rolling(window=5, center=False).mean()
        ma2 = data['ratios'].rolling(window=60, center=False).mean()   #hard-coding window of 60
        std = data['ratios'].rolling(window=60, center=False).std()    #hard-coding window of 60
        data['zscore'] = (ma1 - ma2)/std

        data['position1'] = np.where(data['zscore'] > 1.5, -1, np.nan)
        data['position1'] = np.where(data['zscore'] < -1.5, 1, data['position1'])
        data['position1'] = np.where(abs(data['zscore']) < 0.5, 0, data['position1'])

        data['position1'] = data['position1'].ffill().fillna(0)
        data['position2'] = -np.sign(data['position1'])

        data['returns1'] = np.log(S1/S1.shift(1)).fillna(0)
        data['returns2'] = np.log(S2/S2.shift(1)).fillna(0)
        #strategy is doubtful
        data['strategy'] = 0.5*(data['position1'].shift(1) * data['returns1']) + 0.5*(data['position2'].shift(1) * data['returns2'])
        
        self.df = data.iloc[60:]   #hard-coding window of 60?
        self.returns = self.df[['returns1', 'returns2', 'strategy']].dropna().cumsum().apply(np.exp).tail(1)
        #return display(self.returns)   #new
        #print(self.returns)
        print('testtestets')

    def reset(self):
        self._done = False
        self._current_tick = self.window_size
        self._position = Positions.Flat
        self._positionhistory = []
        self._total_reward = 0.0
        self._total_return = 0.0
        self.history = {}
        return self.get_observation()
    
    def step(self, action):
        step_reward = 0
        for i in range(self.trade_period):
            return1 = self.df.loc[self.df.index[self._current_tick], 'returns1']
            return2 = self.df.loc[self.df.index[self._current_tick], 'returns2']
            if self._position == Positions.Short:
                step_reward += -0.5 * return1 + 0.5 * return2   #Positions.Short, then step_reward = short S1 long S2
            elif self._position == Positions.Long:
                step_reward += 0.5 * return1 - 0.5 * return2    #Positions.Long, then step_reward = long S1 short S2
            self._current_tick += 1
            self._positionhistory.append(self._position.value)
        
        self._position = action
        self._done = (self._current_tick + self.trade_period > self.df.shape[0])
        self._total_reward += step_reward
        self._total_return += step_reward
        observation = self.get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_return = self._total_return
        )

        return observation, step_reward, self._done, info
    
    def get_observation(self):
        end_index = self._current_tick
        start_index = end_index - self.window_size
        zscore = self.df.iloc[start_index:end_index, self.df.columns.get_loc('zscore')].values   #zscore = (ma2-ma1)/std
        return np.append(zscore, [self._position.value, self.pvalue])

    def render(self, mode='human', close=False):
        start_index = self.window_size
        end_index = start_index + len(self._positionhistory)
        mask = pd.Series(self._positionhistory, index=self.df.index[start_index:end_index])
        
        self.plot_returns_delta(mask)
        self.plot_prices(mask)
        
    def plot_returns(self, mask):
        plt.figure(figsize=(12,6))
        
        start_index = self.window_size
        end_index = start_index + len(self._positionhistory)
        zscore = self.df.iloc[start_index:end_index, self.df.columns.get_loc('zscore')]
        zscore.plot()
        
        buy = zscore.copy()
        sell = zscore.copy()
        
        buy[mask==Positions.Short.value] = -np.inf
        sell[mask==Positions.Long.value] = -np.inf
        buy[mask==Positions.Flat.value] = -np.inf
        sell[mask==Positions.Flat.value] = -np.inf
        
        buy.plot(color='r', linestyle='None', marker='^')
        sell.plot(color='g', linestyle='None', marker='^')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, zscore.min(), zscore.max()))
        
        plt.legend(['Zscore', 'Buy Signal', 'Sell Signal'])
        plt.show()

    def plot_returns_delta(self, mask):
        plt.figure(figsize=(12,6))
        
        start_index = self.window_size
        end_index = start_index + len(self._positionhistory)
        zscore = self.df.iloc[start_index:end_index, self.df.columns.get_loc('zscore')]      ### NEW ###
        
        #return1 = self.df.iloc[start_index:end_index, self.df.columns.get_loc('return1')]
        #return2 = self.df.iloc[start_index:end_index, self.df.columns.get_loc('return2')]
        return1 = self.df.iloc[start_index:end_index, self.df.columns.get_loc('returns1')]   ### NEW ###
        return2 = self.df.iloc[start_index:end_index, self.df.columns.get_loc('returns2')]   ### NEW ###
        returns_delta = return1 - return2
        returns_delta.plot()
        
        buy = returns_delta.copy()
        sell = returns_delta.copy()
        
        buy[mask==Positions.Short.value] = -np.inf
        sell[mask==Positions.Long.value] = -np.inf
        buy[mask==Positions.Flat.value] = -np.inf
        sell[mask==Positions.Flat.value] = -np.inf
        
        buy.plot(color='r', linestyle='None', marker='^')
        sell.plot(color='g', linestyle='None', marker='^')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, zscore.min(), zscore.max()))
        
        plt.legend(['Returns Delta', 'Buy Signal', 'Sell Signal'])
        plt.show()

    def plot_zscore(self, mask):
        plt.figure(figsize=(12,6))
        
        start_index = self.window_size
        end_index = start_index + len(self._positionhistory)
        zscore = self.df.iloc[start_index:end_index, self.df.columns.get_loc('zscore')]
        zscore.plot()
        
        buy = zscore.copy()
        sell = zscore.copy()
        
        buy[mask==Positions.Short.value] = -np.inf
        sell[mask==Positions.Long.value] = -np.inf
        buy[mask==Positions.Flat.value] = -np.inf
        sell[mask==Positions.Flat.value] = -np.inf
        
        buy.plot(color='r', linestyle='None', marker='^')
        sell.plot(color='g', linestyle='None', marker='^')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, zscore.min(), zscore.max()))
        
        plt.legend(['Zscore', 'Buy Signal', 'Sell Signal'])
        plt.show()
    
    def plot_ratios(self, mask):
        plt.figure(figsize=(12,6))
        
        start_index = self.window_size
        end_index = start_index + len(self._positionhistory)
        ratios = self.df.iloc[start_index:end_index, self.df.columns.get_loc('ratios')]
        ratios.plot()
        
        buy = 0*ratios.copy()
        sell = 0*ratios.copy()
        
        buy[mask==Positions.Long.value] = ratios[mask==Positions.Long.value]
        sell[mask==Positions.Short.value] = ratios[mask==Positions.Short.value]
        
        buy.plot(color='r', linestyle='None', marker='^')
        sell.plot(color='g', linestyle='None', marker='^')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, ratios.min(), ratios.max()))
        
        plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
        plt.show()
    
    def plot_prices(self, mask):
        plt.figure(figsize=(16,6))

        start_index = self.window_size
        end_index = start_index + len(self._positionhistory)
        S1 = self.df.iloc[start_index:end_index, 0]
        S2 = self.df.iloc[start_index:end_index, 1]
        
        S1.plot(color='b')
        S2.plot(color='c')
        buyR = 0*S1.copy()
        sellR = 0*S1.copy()

        # When you buy the ratio, you buy stock S1 and sell S2
        buyR[mask==Positions.Long.value] = S1[mask==Positions.Long.value]
        sellR[mask==Positions.Long.value] = S2[mask==Positions.Long.value]

        # When you sell the ratio, you sell stock S1 and buy S2
        buyR[mask==Positions.Short.value] = S2[mask==Positions.Short.value]
        sellR[mask==Positions.Short.value] = S1[mask==Positions.Short.value]

        buyR.plot(color='r', linestyle='None', marker='^')
        sellR.plot(color='g', linestyle='None', marker='^')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, min(S1.min(), S2.min()), max(S1.max(), S2.max())))

        plt.legend([S1.name, S2.name, 'Buy Signal', 'Sell Signal'])
        plt.show()