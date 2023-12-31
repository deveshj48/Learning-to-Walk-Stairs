#!/usr/bin/env python
# coding: utf-8

import math
import random
import logging
import wandb

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt
from matplotlib import animation
from environment import * # get custom environment

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


# <h2>Auxilliary Functions</h2>

# In[3]:


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


# In[4]:


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return actions


# In[5]:


def plot(steps, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('steps %s. reward: %s' % (steps, rewards[-1]))
    plt.plot(rewards)
    plt.show()


# <h1>Network Definitions</h1>

# In[6]:


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample()
        action = torch.tanh(mean+ std*z.to(device))
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        # print(state)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(device)
        action = torch.tanh(mean + std*z)
        
        action  = action.cpu()#.detach().cpu().numpy()
        # action  = action.detach().cpu().numpy()
        # print(action)
        return action[0]

# <h1> Update Function </h1>

# In[7]:


def update(batch_size, replay_buffer, gamma=0.99,soft_tau=1e-2,):
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value    = value_net(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

    
    
# Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())


    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()    
# Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action),soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - log_prob
    value_loss = value_criterion(predicted_value, target_value_func.detach())

    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
# Training Policy Function
    policy_loss = (log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    stats["policy_loss"] = policy_loss
    
    
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


# In[8]:    


hyperparams = {
        "env": 'CustomAnt-v0',
        "batch_size": 128,
        "buffer_size": int(1e6),
        "lr":3e-4,
        "hidden_size":256,
        "gamma":0.99,
        "tau":1e-2,
        "seed":42,
        "max_steps":1000000,
        "record_every":50
}


# <h2> Initializations </h2>

# In[9]:


# env = NormalizedActions(gym.make("Pendulum-v1"))
env = NormalizedActions(gym.make(hyperparams["env"], render_mode="rgb_array"))
env = gym.wrappers.RecordVideo(env, 'video', episode_trigger = lambda x: x % hyperparams["record_every"] == 0)

action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]

value_net        = ValueNetwork(state_dim, hyperparams["hidden_size"]).to(device)
target_value_net = ValueNetwork(state_dim, hyperparams["hidden_size"]).to(device)

soft_q_net1 = SoftQNetwork(state_dim, action_dim, hyperparams["hidden_size"]).to(device)
soft_q_net2 = SoftQNetwork(state_dim, action_dim, hyperparams["hidden_size"]).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hyperparams["hidden_size"]).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    

value_criterion  = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

value_optimizer  = optim.Adam(value_net.parameters(), lr=hyperparams["lr"])
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=hyperparams["lr"])
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=hyperparams["lr"])
policy_optimizer = optim.Adam(policy_net.parameters(), lr=hyperparams["lr"])

replay_buffer = ReplayBuffer(hyperparams["buffer_size"])


# # Training Loop

# In[11]:

total_steps = 0
rewards_arr = []
ave_rewards_arr = []
episode_rewards = [0.0]

wandb.init(
            project="SAC Ant",
            entity="2312213",
        )
stats = {}

state = env.reset()[0]
episode_reward = 0
while (total_steps < hyperparams["max_steps"]):

    
    if total_steps >1000:
        action = policy_net.get_action(state).detach()
        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
    else:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
    
    replay_buffer.push(state, action, reward, next_state, done)
    
    episode_rewards[-1] += reward
    state = next_state
    total_steps += 1
    
    if len(replay_buffer) > hyperparams["batch_size"]:
        update(hyperparams["batch_size"], replay_buffer)

    
    if done:

        state = env.reset()[0]
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1) # average of last 100 episodes
        stats["return"] = mean_100ep_reward
        wandb.log(stats)


        rewards_arr.append(episode_rewards[-1])
        ave_rewards_arr.append(mean_100ep_reward)

        episode_rewards.append(0.0)
        logging.basicConfig(level=logging.INFO)
        logging.info("********************************************************")
        logging.info("steps: {}".format(total_steps))
        logging.info("episodes: {}".format(len(episode_rewards)))
        logging.info("mean 100 episode reward: {}".format(mean_100ep_reward))
        logging.info("********************************************************")

