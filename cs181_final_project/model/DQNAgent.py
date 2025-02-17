import gymnasium as gym
import ale_py as ale
from atariari.benchmark.wrapper import AtariARIWrapper

import os
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
from PIL import Image

import numpy as np
import random

from .Agent import Agent
from . import utils


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = utils.deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(utils.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent(Agent):
    
    def __init__(self, **kwargs):
        super(DQNAgent, self).__init__()
        self.BATCH_SIZE = kwargs.get('batch_size', 64)
        self.GAMMA = kwargs.get('gamma', 0.99)
        self.EPS_START = kwargs.get('eps_start', 0.9)
        self.EPS_END = kwargs.get('eps_end', 0.1)
        self.EPS_DECAY = kwargs.get('eps_decay', 1000)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.action_dim = 3
        self.state_dim = kwargs.get('state_dim')
        self.epsilons = 0.0
        self.update_counter = 0
        self.interaction_counter = 0
        
        self.memory = ReplayMemory(kwargs.get('memory_capacity', 10000))
        
        self.policy_net = kwargs.get('policy_net').to(self.device)
        self.target_net = kwargs.get('target_net').to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = kwargs.get('optimizer')

    def update(self, **kwargs):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = utils.Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, model, path, **kwargs):
        path = os.path.join(path, '{}/checkpoint_{}.pth'.format(model, self.interaction_counter))
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        print(f"### step {self.interaction_counter} ### saved to {path}")
        torch.save(self.policy_net.state_dict(), path)
     
    def load(self, path, **kwargs):
        # self.target_net.load_state_dict(torch.load(path+'/target_checkpoint_{}.pth'.format(self.interaction_counter)))
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()
    
    def select_action(self, state, **kwargs):
        self.interaction_counter += 1
        self.epsilon = self.EPS_END + np.maximum( (self.EPS_START-self.EPS_END) * (1 - self.interaction_counter/self.EPS_DECAY), 0)
        if kwargs.get('train'):
            if np.random.rand() < self.epsilon:
                return torch.tensor([[np.random.choice(self.action_dim)]], device=self.device, dtype=torch.int16)
            else:
                with torch.no_grad():
                    return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

    def store_transition(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

class DQN_info(nn.Module):
    
    def __init__(self, n_observations, n_actions):
        super(DQN_info, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent_info(DQNAgent):
    
    def __init__(self, **kwargs):
        state_dim = 10
        policy_net = DQN_info(state_dim, 3)
        target_net = DQN_info(state_dim, 3)
        lr = kwargs.get('lr', 1e-4)
        self.env = kwargs.get('env')
        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        super(DQNAgent_info, self).__init__(policy_net=policy_net, target_net=target_net, optimizer=optimizer, state_dim=state_dim, **kwargs)

    def reset(self):
        observation, info = self.env.reset()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

class DQN_image(nn.Module):
    
    def __init__(self, h, w, outputs):
        super(DQN_image, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
        linear_input_size = convw * convh * 64
        
        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.fc(x.view(x.size(0), -1)))
        return self.head(x)

class DQNAgent_image(DQNAgent):
    
    def __init__(self, **kwargs):
        state_dim = (84, 84)
        policy_net = DQN_image(state_dim[0], state_dim[1], 3)
        target_net = DQN_image(state_dim[0], state_dim[1], 3)
        self.env = kwargs.get('env')
        lr = kwargs.get('lr', 1e-4)
        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        super(DQNAgent_image, self).__init__(policy_net=policy_net, target_net=target_net, optimizer=optimizer, state_dim=state_dim, **kwargs)
        self.agent_history_length = 4

    def reset(self):
        observation, info = self.env.reset()
        frame = self.image_proc(observation).to(self.device)
        self.state = frame.repeat(1,self.agent_history_length,1,1)
        return self.state, info

    def image_proc(self, image):
        processed_image = utils.image_proc(image)
        return processed_image

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        frame = self.image_proc(observation).to(self.device)
        next_state = torch.cat((self.state[:, 1:, :, :], frame.unsqueeze(0)), axis=1)
        self.state = next_state
        return next_state, reward, terminated, truncated, info