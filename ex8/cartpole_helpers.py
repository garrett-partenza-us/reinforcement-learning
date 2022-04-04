#necessary libraries
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
from dqncartpole import DQN
from itertools import count
from PIL import Image
device='cpu'

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

#function to retrieve carts location within the current frame
def get_cart_location(w):
    world_w = 2 * env.x_threshold
    scale = w / world_w
    loc = int(env.state[0] * scale + w / 2.0)
    return loc

def get_screen(env):
    #get screen frame from gym environment
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    #convert to tensor
    screen = torch.tensor(np.ascontiguousarray(screen, dtype=np.float32) / 255)
    #convert to batch, channels, height, width dimensions
    transform = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])
    return transform(screen).unsqueeze(0)

#get the current epsilon value
def decay_eps(steps_done, eps_start, eps_end, eps_decay):
    return eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)

def select_action(env, state, steps_done, eps_start, eps_end, eps_decay, PModel):
    #generate randome number between 0 and 1 for epsilon greedy probability 
    random_num = random.random()
    #decay epsilon based on step count
    eps_threshold = decay_eps(steps_done, eps_start, eps_end, eps_decay)
    #generate an action
    if random_num > eps_threshold:
        return PModel(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)

#function from torch documentation for live plotting
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Live Loss Watcher')
    plt.xlabel('Episode Number')
    plt.ylabel('Time Spent Alive')
    plt.plot(durations_t.numpy())
    #plot the rolling 10-episode average, pytorch uses 100 but is too long and not informative 
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())
    plt.pause(0.0001)
    #not sure what this is but the program breaks without it, also found in pytorch docs
    #https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        

#optimization step
def optimize_model(memory, PModel, TModel, batch_size, gamma, optimizer):
    #only run optimization if there is enough to sample from replay memory
    if len(memory) < batch_size:
        return
    #sample transitions from replay memory
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    #get a tensor list of boolean masks which indicate whether the next state was a terminal state
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    #get a tensor of states where the next state is not terminal
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    #get s, a, s_new variables from batch
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    #forward pass states to get policy q values
    state_action_values = PModel(state_batch).gather(1, action_batch)
    #forward pass next states to get target q values for expectation
    torch.zeros(batch_size, device=device)[non_final_mask] = TModel(non_final_next_states).max(1)[0].detach()
    #correct with the reward and discount factor
    expected_state_action_values = (torch.zeros(batch_size, device=device) * gamma) + reward_batch
    #use hubor loss
    loss = nn.HuberLoss()(state_action_values, expected_state_action_values.unsqueeze(1))
    #zero out the gradients
    optimizer.zero_grad()
    #backward pass
    loss.backward()
    #step optimizer
    optimizer.step()
    #the torch tutorial includes another weird step the clips paramter values between -1 and 1
    #not sure why they do this but I dont see a drastic change in performance
    
    