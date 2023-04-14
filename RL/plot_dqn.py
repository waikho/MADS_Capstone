import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import count

import torch

from stock_trading_environment import Positions


#plot entries and exits for all envs
def plotEntryExits(policy_net, target_net, envs, device):
    policy_net.eval()   #changes forward(); disables Dropout, BatchNorm, etc.
    target_net.eval()

    total_rewards = []
    total_returns = []
    with torch.no_grad():
        for env in envs:
            obs = env.reset()
            state = torch.tensor(np.array([obs]), dtype=torch.float, device=device)
            for t in count():   #keep looping until done -> break
                # Select and perform an action
                action = policy_net(state).max(1)[1].view(1, 1)
                obs, _reward, done, info, _return = env.step(Positions(action.item()))
                state = torch.tensor(np.array([obs]), dtype=torch.float, device=device)
                
                if done:
                    env.render()   #only rendering the last year
                    print(f"Total reward for this pair: {info['total_reward']}")
                    print(f"Total return for this pair: {info['total_return']}")
                    total_rewards.append(info['total_reward'])
                    total_returns.append(info['total_return'])
                    break

    #avg_returns = np.mean(returns)
    print(f'Average overall rewards: {np.mean(total_rewards)}')
    print(f'Average overall returns: {np.mean(total_returns)}')

    plt.ioff()
    plt.show()


#plot episodic results
def plot_episodes(data_dict):
    # Convert dictionary to a Pandas DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='columns').astype(float)

    #Plot all reward/return across all the paths and episodes
    sns.lineplot(data=df, dashes=False, legend=False)
    
    # Set the plot title and axis labels
    plt.suptitle("Each step's returns across all episodes")
    plt.xlabel('Steps in each episode')
    plt.show()

    #Calculate the (means) cumsum for each column
    #means = df.mean()
    cumsum = df.cumsum()
    
    #get the index of the column with the highest cumulative sum
    highest_cumulative_sum = cumsum.iloc[-1,:].idxmax()

    #print("Column with highest cumulative sum:", highest_cumulative_sum)

    #Plot all cumulative sums
    sns.lineplot(data=cumsum, legend=False, dashes=False)#, palette=['pink'], dashes=False, legend=False)
    
    #Set the plot title and axis labels
    plt.suptitle('Cumulative returns across all episodes')
    plt.xlabel('Steps in each episode')
    plt.ylabel('Cumulative return')
    plt.show()


    #Plot the highest cumulative sum only
    sns.lineplot(data=cumsum[highest_cumulative_sum], legend=False)#, palette=['pink'], dashes=False, legend=False)

    # Set the plot title and axis labels
    plt.suptitle('Cumulative returns in the highest-return episode')
    plt.xlabel('Steps in the episode')
    plt.ylabel('Cumulative return')

    #Display the plot
    plt.show()
    print(f'Highest cumulative return in epoch {highest_cumulative_sum}, return: {cumsum.iloc[-1][highest_cumulative_sum]}')


#plot specific episode range
def plot_n_episodes(data_dict, start_ep, end_ep):
    # Convert dictionary to a Pandas DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='columns').astype(float)

    # Calculate the (means) cumsum for each column
    #means = df.mean()
    cumsum = df.iloc[:, start_ep:end_ep].cumsum()

    #Plot all cumulative sums
    sns.lineplot(data=cumsum, legend=False, dashes=False)#, palette=['pink'], dashes=False, legend=False)
    
    # Set the plot title and axis labels
    plt.suptitle('Cumulative returns across selected episodes')
    plt.xlabel('Steps in each episode')
    plt.ylabel('Cumulative return')
    plt.show()

    # get the index of the column with the highest cumulative sum
    highest_cumulative_sum = cumsum.iloc[-1,:].idxmax()

    #print("Column with highest cumulative sum:", highest_cumulative_sum)
    print(f'Highest cumulative return in epoch {highest_cumulative_sum}, return: {cumsum.iloc[-1][highest_cumulative_sum]}.')

    return cumsum