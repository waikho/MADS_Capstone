#Code credit: https://github.com/jeffreyyu0602/ZheShang/tree/0cebb2ec90a5921d06a4d117e9202cca1ae511fc
#With modifications

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple, deque
import math
import random


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
            nn.LeakyReLU(),        #try LeakyReLU
            nn.Dropout(0.5),       #increased from 0.25
            nn.Linear(256, 256),
            nn.LeakyReLU(),        #try LeakyReLu
            nn.Dropout(0.5),       #increased from 0.25
            nn.Linear(256, 256),   # new layer; try 256 -> 256
            nn.LeakyReLU(),        #try LeakyReLU
            nn.Dropout(0.5),      # new dropout layer; increased from 0.25
            nn.Linear(256, n_actions),   #try 256 neurons
            nn.Sigmoid()
        )
        
        self.device = device

    def forward(self, x):
        return self.model(x.to(self.device))
    

    #select_action function
    def select_action(self, state, EPS_START, EPS_END, EPS_DECAY, steps_done, policy_net, n_actions):
        #global steps_done
        sample = random.random()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:   #epsilon-greedy: at first, more random less nn; gradually more nn less random
            with torch.no_grad():   #disable tracking of grad in autograd; reduce memory usage and speed up computations; no backprop
                #t.max(1) = largest column value of each row.
                #second column on max result = index of where max element was found
                #pick action with the larger expected reward.
                nn_count = 1
                return policy_net(state).max(1)[1].view(1, 1), nn_count   #exploit
        else:
            nn_count = 0
            return torch.tensor([[random.randrange(n_actions)]], device=self.device, dtype=torch.long), nn_count   #explore, because random
        

    #optimize_model function
    def optimize_model(self, memory, BATCH_SIZE, policy_net, target_net, GAMMA, optimizer):
        if len(memory) < BATCH_SIZE:   #min memory = 128
            return
        transitions = memory.sample(BATCH_SIZE)
        
        #Transpose the batch
        #each batch has 4 elements: 62*128, 1*128, 62*128, 1*128: memory.push(state, action, next_state, reward)
        batch = Transition(*zip(*transitions))   #batch is a namedtuple; tuple (state, action, next_state, reward) from memory
        #mask of non-final states and concatenate the batch elements
        #len(non_final_mask) = 128
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,   #lambda s: s is not None will return a tuple of bools over bench.next_state
                                            batch.next_state)), device=self.device, dtype=torch.bool)   #next_state is an observation
        #each non_final_next_states has len = 62; one batch of non_final_next_states = 128; so 62 * 128 = 7936 elements in a batch
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)     #128*62 = 7936 elements
        action_batch = torch.cat(batch.action)   #128
        reward_batch = torch.cat(batch.reward)   #128

        #Q(s_t, a) is what the policy_net says based on state_batch
        state_action_values = policy_net(state_batch).gather(1, action_batch)   #gather from policy_net(state_batch) (axis, index)

        #Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)   #initialize next_state_values
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()   #what target_net says, based on next states only
        #Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch   
        #Bellman equation: next state_action values (from target network) * GAMMA + immediate reward should equal to
        #state_action values returned by policy network; difference = loss

        #BCELoss best for binary classification
        criterion = nn.BCELoss(reduction='mean')
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        #Optimize the model
        optimizer.zero_grad()   #set optimizer's grad to zero
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)   #clamp_ is inplace version of clamp: clamp grad.data into -1 and 1 range
        optimizer.step()   #update parameters of optimizer


