#copying stock_trading_environment code here for easier editing
#remember to copy back this to the stock_trading environment.py
import gym
from gym import spaces
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS


class Positions(Enum):   #Python's enum class
    #Short = 0
    #Long = 0
    #Flat = 1
    #Long = 2
    #Short = 2
    Risk_off = 0
    Risk_on = 1


class StockTradingEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, window_size, trade_period, trans_cost):
        super().__init__()

        self.window_size = window_size
        self.trade_period = trade_period
        self.trans_cost = trans_cost
        
        self.trade(data)
        
        # Actions: SHORT(0), FLAT(1), LONG(2)
        #actions: risk_off(0), risk_on(1)
        #self.action_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
          low=-np.inf, high=np.inf, shape=(window_size, 1), dtype=np.float16)   #a box as observation space

    #def test(self):
    #    print('testtesttest')
    
    def trade(self, data):
        S1 = data.iloc[:, 0]
        S2 = data.iloc[:, 1]
        _score, self.pvalue, _ = coint(S1, S2)   #probably don't need to include self.pvalue in info; to delete?
        
        #run OLS regression
        ols_model=OLS(S1, S2).fit()
        #get pair's hedge ratio
        self._hr = ols_model.params[0]
            
        #calculate spread
        data['spread'] = np.log(S1) - self._hr * np.log(S2)
        #data['spread'] = S1 - self._hr * S2

        #calculate rolling mean
        data['roll_mean'] = data['spread'].rolling(window=self.window_size, center=False).mean()

        #calculate rolling std
        data['roll_std'] = data['spread'].rolling(window=self.window_size, center=False).std()

        #calculate zscore
        data['zscore'] = (data['spread'] - data['roll_mean'])/data['roll_std']
        
        #exclude the first window length
        self.df = data.iloc[self.window_size:]
        #self.df = data.iloc[60:]   #hard-coding window of 60?
        #self.returns = self.df[['returns1', 'returns2', 'strategy']].dropna().cumsum().apply(np.exp).tail(1)   #tail - last item
        #print(self.returns)
        #return self.returns   #new
        #print('testtestets')

    def reset(self):
        self._done = False
        self._current_tick = self.window_size
        #self._position = Positions.Flat
        self._position = Positions.Risk_off
        self._positionhistory = []
        self._exit_record = [0.0, 0]
        self._total_reward = 0.0
        self._total_return = 0.0
        #self._return = 0.0   #new
        #self.history = {}
        return self.get_observation()
    
    def step(self, action):
        step_reward = 0.0
        step_return = 0.0
        #print(f'self._position: {self._position}')
        #print(f'action: {action}')

        if self._position == Positions.Risk_off:
            if action == Positions.Risk_off:
                step_reward = 0.0
            else:
                self._enter_record = self.enter_trade()
                if self._current_tick - self._exit_record[1] <= 5:   #penalize frequent trading of less than a week
                    step_reward = - np.abs(self._exit_record[0]) * 0.05
                    #step_reward = 0.0
                else:
                    step_reward = 0.0
                #step_reward = 0.1   #some reward to encourage taking risk
                #step_reward = -1   #transaction cost to discourage trading   ###changed###
                self._position = Positions.Risk_on

        elif self._position == Positions.Risk_on:
            if action == Positions.Risk_on:
                step_reward = 0.0
            else:
                #self._return = self.exit_trade(self._enter_record)
                self._exit_record = self.exit_trade(self._enter_record)   #self._exit_record = [pnl, exit_tick]
                step_return = self._exit_record[0]
                
                #penalize frequent trading of less than a week
                if self._current_tick - self._enter_record[2] <= 5:
                    #step_reward = self._return - np.abs(self._return) * 0.2
                    step_reward = self._exit_record[0] - np.abs(self._exit_record[0]) * 0.05   #pnl - 0.05 * pnl
                else:
                    #step_reward = self._return
                    step_reward = self._exit_record[0]
                #print(f'Trade PnL: {self._return}')
                self._position = Positions.Risk_off

        #print(f'step_reward: {step_reward}')
        self._current_tick += 1
        self._positionhistory.append(self._position.value)

        self._position = action   #update self._position with action returned by nn
        #self._done = (self._current_tick + self.trade_period > self.df.shape[0])
        self._done = (self._current_tick + 1 > self.df.shape[0])   #new; no longer need trade_period
        
        self._total_reward += step_reward
        #self._total_return += self._return
        self._total_return += step_return
        #print(f'self._total_return: {self._total_return}')
        
        observation = self.get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_return = self._total_return
        )

        return observation, step_reward, self._done, info, step_return
        #return observation, step_reward, self._done
    
    
    def get_observation(self):
        end_index = self._current_tick
        start_index = end_index - self.window_size
        zscore = self.df.iloc[start_index:end_index, self.df.columns.get_loc('zscore')].values   #zscore = (spread-roll+mean)/roll_std
        return np.append(zscore, [self._position.value, float(self.pvalue)])   #60 zscores + _position.value + pvalue = 62 items in each obs
        #return np.append(zscore, [self._position.value, self._current_tick])   #60 zscores + _position.value + _current_tick = 62 items in each obs


    def enter_trade(self):
        #+ve and -ve directions are a bit counter-intuitive below
        #the logic is: if you short/sell something, your cash level increases by the price
        #if you long/buy something, your cash level decreases by the price
        #this is just for easier calculations: say you short $100 and later it drops to $90, the pnl is +$100-$90=+$10
        #say you long $90 and later it increases to $110, the pnl is -$90+$110=+$20
        #adjusted by hedge ratio

        if self._hr >= 1:   #S1 more expensive than S2; should short S1 long S2
            self._S1_entry_price = self.df.loc[self.df.index[self._current_tick]][0]               #short/sell
            self._S2_entry_price = -self._hr * self.df.loc[self.df.index[self._current_tick]][1]   #long/buy
        
        else:   #S2 more expensive than S1; should long S1 short S2
            self._S1_entry_price = -self.df.loc[self.df.index[self._current_tick]][0]             #long/buy
            self._S2_entry_price = self._hr * self.df.loc[self.df.index[self._current_tick]][1]   #short/sell

        return [self._S1_entry_price, self._S2_entry_price, self._current_tick]
    

    def exit_trade(self, trade_record):
        
        if self._hr >= 1:   #S1 more expensive than S2; should short S1 long S2 initially
            #unwind: buy S1 sell S2
            pnl = (trade_record[0] - self.df.loc[self.df.index[self._current_tick]][0]) + \
                (trade_record[1] + self._hr * self.df.loc[self.df.index[self._current_tick]][1]) - self.trans_cost
        
        else:   #S2 more expensive than S1; should long S1 short S2 initially
            #unwind: sell S1 buy S2
            pnl = (trade_record[0] + self.df.loc[self.df.index[self._current_tick]][0]) + \
                (trade_record[1] - self._hr * self.df.loc[self.df.index[self._current_tick]][1]) - self.trans_cost

        #should not penalize here    
        #penalize frequent trading of less than a week
        #if self._current_tick - trade_record[2] <= 5:
        #    pnl = pnl*0.9

        return [pnl, self._current_tick]
           
    
    def render(self, mode='human', close=False):
        start_index = self.window_size
        #start_index = len(self._positionhistory) - 244   #to shorten the plot
        end_index = start_index + len(self._positionhistory)
        #mask = pd.Series(self._positionhistory, index=self.df.index[start_index:end_index])
        #mask = pd.DataFrame(self._positionhistory, index=self.df.index[start_index:end_index])    ###changed###
        #short_mask = pd.DataFrame(self._positionhistory[-244:], index=self.df.index[-244:])   #newly created to plot twists
        short_start_index = max(-365, -len(self._positionhistory))   #to shorten the plot, subject to lower of -244 and len(positionhistory)
        short_mask = pd.DataFrame(self._positionhistory[short_start_index:], index=self.df.index[short_start_index:])   #newly created to plot twists
        
        
        #new
        short_mask['shift'] = short_mask[0].shift(1).fillna(0)
        short_mask['diff'] = short_mask[0] - short_mask['shift']
        #if diff = +1, from 0 to 1, trade, risk_on
        #if diff = 0, from 0 to 0, or 1 to 1, no trade
        #if diff = -1, from 1 to 0, trade, risk_off


        #self.plot_returns_delta(mask)
        #self.plot_prices(mask)
        #print(short_mask['diff'].unique())
        self.plot_short_prices(short_mask['diff'])
        

        #mask['shift'] = mask[0].shift(1).fillna(0)
        #mask['diff'] = mask[0] - mask['shift']
        #print(mask['diff'].unique())
        #self.plot_short_prices(mask['diff'])


    def plot_returns(self, mask):
        plt.figure(figsize=(12,6))
        
        #start_index = self.window_size
        start_index = len(self._positionhistory) - 244   #to shorten the plot
        end_index = start_index + len(self._positionhistory)
        zscore = self.df.iloc[start_index:end_index, self.df.columns.get_loc('zscore')]
        zscore.plot()
        
        buy = zscore.copy()
        sell = zscore.copy()
        
        buy[mask==Positions.Risk_off.value] = -np.inf
        sell[mask==Positions.Risk_on.value] = -np.inf
        #buy[mask==Positions.Flat.value] = -np.inf
        #sell[mask==Positions.Flat.value] = -np.inf
        
        buy.plot(color='r', linestyle='None', marker='^')
        sell.plot(color='g', linestyle='None', marker='^')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, zscore.min(), zscore.max()))
        
        plt.legend(['Zscore', 'Buy Signal', 'Sell Signal'])
        plt.show()

    def plot_returns_delta(self, mask):
        plt.figure(figsize=(12,6))
        
        #start_index = self.window_size
        start_index = len(self._positionhistory) - 244
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
        
        buy[mask==Positions.Risk_on.value] = -np.inf
        sell[mask==Positions.Risk_off.value] = -np.inf
        #buy[mask==Positions.Flat.value] = -np.inf
        #sell[mask==Positions.Flat.value] = -np.inf
        
        buy.plot(color='r', linestyle='None', marker='^')
        sell.plot(color='g', linestyle='None', marker='^')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, zscore.min(), zscore.max()))
        
        plt.legend(['Returns Delta', 'Buy Signal', 'Sell Signal'])
        plt.show()

    def plot_zscore(self, mask):
        plt.figure(figsize=(12,6))
        
        #start_index = self.window_size
        start_index = len(self._positionhistory) - 244   #to shorten the plot
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
        #start_index = len(self._positionhistory) - 244   #to shorten the plot
        end_index = start_index + len(self._positionhistory)
        S1 = self.df.iloc[start_index:end_index, 0]
        S2 = self.df.iloc[start_index:end_index, 1]
        #print(S1)
        #print(S2)
        
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

    def plot_short_prices(self, mask):
        plt.figure(figsize=(16,6))

        start_index = self.window_size
        short_start_index = max(-365, -len(self._positionhistory))   #to shorten the plot, subject to lower of -244 and len(positionhistory)
        end_index = start_index + len(self._positionhistory)
        #S1 = self.df.iloc[start_index:end_index, 0]
        #S2 = self.df.iloc[start_index:end_index, 1]
        S1 = self.df.iloc[short_start_index:end_index, 0]
        S2 = self.df.iloc[short_start_index:end_index, 1]
        #print(S1)
        #print(S2)
        
        S1.plot(color='b')
        S2.plot(color='c')
        #buyR = 0*S1.copy()
        #sellR = 0*S1.copy()
        buy_ron = 0*S1.copy()
        sell_ron = 0*S1.copy()
        buy_roff = 0*S1.copy()
        sell_roff = 0*S1.copy()

        #if diff = +1, from 0 to 1, trade, risk_on
        #if diff = 0, from 0 to 0, or 1 to 1, no trade
        #if diff = -1, from 1 to 0, trade, risk_off
        if self._hr >= 1:   #S1 more expensive than S2; should short S1 long S2
            #risk_on
            sell_ron[mask==1] = S1[mask==1]
            buy_ron[mask==1] = S2[mask==1]

            #risk_off
            buy_roff[mask==-1] = S1[mask==-1]
            sell_roff[mask==-1] = S2[mask==-1]


        else:   #S2 more expensive than S1; should long S1 short S2
            #risk_on
            buy_ron[mask==1] = S1[mask==1]
            sell_ron[mask==1] = S2[mask==1]

            #risk_off
            sell_roff[mask==-1] = S1[mask==-1]
            buy_roff[mask==-1] = S2[mask==-1]


        buy_ron.plot(color='g', linestyle='None', marker='^')
        sell_ron.plot(color='r', linestyle='None', marker='v')
        buy_roff.plot(color='g', linestyle='None', marker='x')
        sell_roff.plot(color='r', linestyle='None', marker='x')
        #buyR.plot(color='g', linestyle='None', marker='^')
        #sellR.plot(color='r', linestyle='None', marker='v')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, min(S1.min(), S2.min()), max(S1.max(), S2.max())))

        plt.legend([S1.name, S2.name, 'Buy Risk-on', 'Sell Risk-on', 'Buy Risk-off', 'Sell Risk-off'])
        plt.show()

        