import gymnasium as gym
import ale_py
import numpy as np
import random
from atariari.benchmark.wrapper import AtariARIWrapper

import math
import random
import wandb
from collections import namedtuple, deque
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Q-Learning Training Script")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes to train")
    parser.add_argument("--max_timesteps", type=int, default=1000, help="Maximum timesteps per episode")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt", help="Directory to save/load checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--play", action="store_true", default=False, help="Play the policy")
    return parser.parse_args()

args = parse_args()

ckpt_path = '/home/jimmyhan/Desktop/Course/CS181-project/ckpt/DQN_add_self_ckpt_250.pth'

gym.register_envs(ale_py)
if args.play:
    env = AtariARIWrapper(gym.make('Freeway-ramNoFrameskip-v4', mode=0))
else:
    env = AtariARIWrapper(gym.make('Freeway-ramNoFrameskip-v4', mode=0))

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
num_episodes = 501
play_policy = False

import os
if args.play:
    os.environ["WANDB_MODE"] = "offline"
else:
    os.environ["WANDB_MODE"] = "online"

wandb.init(project="freeway_ai", config={
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "gamma": GAMMA,
    "num_episodes": num_episodes
})

######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classes:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

NUM_CAR_OBS = 10
num_buckets = 20
value_range = 160
bucket_size = value_range // num_buckets

def map_to_bucket(value):
    return min(int(value // bucket_size), num_buckets - 1)

def state_to_q_index(state):

    if "player_y" not in state:
        print("true")
        indices = torch.zeros(NUM_CAR_OBS + 1, device=device, dtype=torch.int16)
        return indices
    
    player_y = state['player_y']
    enemy_cars_y = np.linspace(0, 180, num=12, dtype=np.int32)  
    enemy_cars_x = [state[f'enemy_car_x_{i}'] for i in range(10)]
    enemy_cars_x = np.pad(enemy_cars_x, (1, 1), 'constant', constant_values=(0))
    distances = np.abs(np.int32(enemy_cars_y) - np.int32(player_y))
    closest_three_indices = np.sort(np.argsort(distances)[:NUM_CAR_OBS])   

    closest_cars_x = [enemy_cars_x[i] for i in closest_three_indices]
    indices = [map_to_bucket(val) for val in closest_cars_x]
    indices.append(player_y)
    return indices

n_actions = 3
max_timesteps = 10000
observation, info = env.reset()
binned_obs = state_to_q_index(info)
n_observations = len(binned_obs)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if not args.play:
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.int16)
    else:
        with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # import ipdb; ipdb.set_trace()
    state_action_values = policy_net(state_batch).gather(1, action_batch.long())

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    wandb.log({"loss": loss.item()})

if args.play:
    print(f"Loading checkpoint from {ckpt_path}")
    policy_net.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("Checkpoint loaded successfully!")

for i_episode in range(num_episodes):
    print("i_episode", i_episode)
    observation, info = env.reset()
    binned_obs = state_to_q_index(info)
    # import ipdb; ipdb.set_trace()
    state = torch.tensor(binned_obs, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0
    reach_target = 0
    y_pos = 0
    for t in range(max_timesteps):
        action = select_action(state)
        # observation, reward, terminated, truncated, _ = env.step(action.item())
        observation, reward, terminated, truncated, info = env.step(action)
        # print("reward_returned", reward)
        binned_obs = state_to_q_index(info['labels'])
        # import ipdb; ipdb.set_trace()
        new_y_pos = info['labels']['player_y']
        # binned_obs.append(new_y_pos)
        reward_y = int(new_y_pos) - int(y_pos) 
        # print("reward_y", reward_y)
        reach_target += reward
        if((action == 1 or action == 0)):
            if reward_y < 0:
                reward_y *= 10
                # reward_y *= 20
            else:
                # reward_y = (reward_y*2)/180 
                reward_y += (new_y_pos*10)/180 
                # reward_y += (new_y_pos*15)/180 
                # print("reward_y", reward_y)
        if(reward == 1):
            reward_y = 0
        y_pos = new_y_pos

        # print("info['labels']['player_y']", info['labels']['player_y'])
        reward = reward_y + reward * 50 - 0.01
        # print("reward", reward)
        episode_reward += reward
        reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(binned_obs, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        if not args.play:
            optimize_model()
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
        if done:
            break
    wandb.log({"episode_reward": episode_reward})
    if args.play:
        print("reward", reach_target) 
    if i_episode % 50 == 0:
        torch.save(policy_net.state_dict(), f"./ckpt/DQN_add_self_ckpt_{i_episode}.pth")

wandb.finish()
print('Complete')