#necessary packages
import gym
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from replaymemory import ReplayMemory, Transition
from cartpole_helpers import *
from dqncartpole import DQN
from itertools import count
from PIL import Image

#get the v1 environment
env = gym.make('CartPole-v1').unwrapped

#we will mimic the live training plots from pytorches documentation
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

#trained on my mac without GPU
device ='cpu'

#list to keep track of the duration of each played episode for plotting
episode_durations = []
#reset environment
env.reset()
#set figure size
plt.figure(figsize=(10,6))
#set hypyter paramters
batch_size = 128
gamma = 0.999
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
target_update = 10

#from the torch documentation
#take an example screen to calculate size paramters for layers of DQN network
temp_screen = get_screen(env)
screen_batch, screen_channels, screen_height, screen_width = temp_screen.shape

# Get number of actions from gym action space
action_count = env.action_space.n

#models
PModel = DQN(screen_height, screen_width, action_count).to(device)
TModel = DQN(screen_height, screen_width, action_count).to(device)
TModel.load_state_dict(PModel.state_dict())
TModel.eval()

#optimizer
optimizer = optim.Adam(PModel.parameters())
#replay memory to hold transitions
memory = ReplayMemory(10000)
#number of steps completed
steps_done = 0
#number of traning episodes
num_episodes = 500

#begin training
for episode_num in range(num_episodes):
    #reset the environment
    env.reset()
    #torch documentation recommends to compute the difference between the last screen and current screen
    last_screen = get_screen(env)
    current_screen = get_screen(env)
    #compute state as the difference
    state = current_screen - last_screen
    #iterate until terminated
    counter = 0
    while True:
        #retrieve an action
        action = select_action(env, state, steps_done, eps_start, eps_end, eps_decay, PModel)
        #increment step counter
        steps_done+=1
        #retrive the reward and terminal state information
        ignore1, reward, done, ignore2 = env.step(action.item())
        #push reward to tensor
        reward = torch.tensor([reward], device=device)
        #previous screen as current screen
        last_screen = current_screen
        #retreive the new screen
        current_screen = get_screen(env)
        next_state = current_screen - last_screen if not done else None
        #store transition into replay memory
        memory.push(state, action, next_state, reward)
        #rollover to next state
        state = next_state
        #optimize the model through q-learning
        optimize_model(memory, PModel, TModel, batch_size, gamma, optimizer)
        #terminate episode if done
        if done:
            #plot to episode durations for live plotting
            episode_durations.append(counter)
            #plot
            plot_durations(episode_durations)
            #break episode loop
            break
        #continue otherwise
        else:
            counter+=1
    #update the target network every 'target_update' steps 
    if episode_num % target_update == 0:
        TModel.load_state_dict(PModel.state_dict())
