from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import gym
import math
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


#memory class
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

#DQN class
class DQN(nn.Module):   #PyTorch's Module class

    def __init__(self, input_size, n_actions, device):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, n_actions),
            #nn.Softmax(dim=1)
            nn.Sigmoid()
        )
        self.device = device

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.model(x.to(self.device))
    

    #select_action function
    def select_action(state, EPS_START, EPS_END, EPS_DECAY, steps_done, policy_net, n_actions, device):
        #global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)   #decreasing from 1 to 0.36xxx
        steps_done += 1
        if sample > eps_threshold:   #epsilon-greedy: at first, more random less nn; gradually more nn less random
            with torch.no_grad():   #disable tracking of grad in autograd; reduce memory usage and speed up computations; no backprop
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #print(policy_net(state)).max(1)
                return policy_net(state).max(1)[1].view(1, 1)   #exploit
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)   #explore, because random
        

    #optimize_model function
    def optimize_model(memory, BATCH_SIZE, device, policy_net, target_net, GAMMA, optimizer):
        if len(memory) < BATCH_SIZE:   #min memory = 128
            return
        transitions = memory.sample(BATCH_SIZE)
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        #each batch has 4 elements: 62*128, 1*128, 62*128, 1*128: memory.push(state, action, next_state, reward)
        batch = Transition(*zip(*transitions))   #batch is a namedtuple; tuple (state, action, next_state, reward) from memory
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        #len(non_final_mask) = 128
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,   #lambda s: s is not None will return a tuple of bools over bench.next_state
                                            batch.next_state)), device=device, dtype=torch.bool)   #next_state is an observation
        #each non_final_next_states has len = 62; one batch of non_final_next_states = 128; so 62 * 128 = 7936 elements in a batch
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)     #128*62 = 7936 elements
        action_batch = torch.cat(batch.action)   #128
        reward_batch = torch.cat(batch.reward)   #128

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        #Q(s_t, a) is what the policy_net says based on state_batch
        state_action_values = policy_net(state_batch).gather(1, action_batch)   #gather from policy_net(state_batch) (axis, index)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)   #initialize next_state_values
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()   #what target_net says
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch   
        #Bellman equation: expected state_action values (from target network) * GAMMA + immediate reward should equal to
        #state_action values returned by policy network; difference = loss

        # Compute Huber loss
        #criterion = nn.SmoothL1Loss()
        #criterion = nn.MSELoss()
        criterion = nn.BCELoss()
        #criterion = nn.L1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()   #set optimizer's grad to zero
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)   #clamp_ is inplace version of clamp: clamp grad.data into -1 and 1 range
        optimizer.step()   #update parameters of optimizer


