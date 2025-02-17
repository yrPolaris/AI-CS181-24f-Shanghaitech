import gymnasium as gym
import ale_py
import numpy as np
import random
from atariari.benchmark.wrapper import AtariARIWrapper
import os


import math
import random
import wandb
from collections import namedtuple, deque
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import argparse

# def parse_args():
#     parser = argparse.ArgumentParser(description="Deep Q-Learning Training Script")
#     parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes to train")
#     parser.add_argument("--max_timesteps", type=int, default=1000, help="Maximum timesteps per episode")
#     parser.add_argument("--ckpt_dir", type=str, default="./ckpt", help="Directory to save/load checkpoints")
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
#     parser.add_argument("--play", action="store_true", default=False, help="Play the policy")
#     return parser.parse_args()

# args = parse_args()

ckpt_path = '/home/jimmyhan/Desktop/Course/CS181-project/ckpt/DQN_add_self_ckpt_50.pth'

# if args.play:
#     env = AtariARIWrapper(gym.make('Freeway-ramNoFrameskip-v4', mode=0))
# else:
#     env = AtariARIWrapper(gym.make('Freeway-ramNoFrameskip-v4', mode=0))

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# BATCH_SIZE = 64
# GAMMA = 0.99
# EPS_START = 0.9
# EPS_END = 0.1
# EPS_DECAY = 1000
# TAU = 0.005
# LR = 1e-4
# num_episodes = 501
# play_policy = False

import os
# if args.play:
#     os.environ["WANDB_MODE"] = "offline"
# else:
os.environ["WANDB_MODE"] = "online"

wandb.init(project="freeway_ai", config={})

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

parser = argparse.ArgumentParser()
parser.add_argument("--train_ep", default=800, type=int)
parser.add_argument("--mem_capacity", default=30000, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--lr", default=0.00025, type=float)
parser.add_argument("--gamma", default=0.999, type=float)
parser.add_argument("--epsilon_start", default=1.0, type=float)
parser.add_argument("--epsilon_final", default=0.1, type=float)
parser.add_argument("--epsilon_decay", default=1000000, type=float)
parser.add_argument("--target_step", default=10000, type=int)
parser.add_argument("--eval_per_ep", default=10, type=int)
parser.add_argument("--save_per_ep", default=50, type=int)
parser.add_argument("--save_dir", default="./model")
parser.add_argument("--log_file", default="./log.txt")
parser.add_argument("--load_model", default=None)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


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

class CNN(nn.Module):
    def __init__(self, h, w, outputs):
        super(CNN, self).__init__()
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

class DQN(object):
    def __init__(self):
        self.BATCH_SIZE = args.batch_size
        self.GAMMA = args.gamma
        self.EPS_START = args.epsilon_start
        self.EPS_END = args.epsilon_final
        self.EPS_DECAY = args.epsilon_decay
        self.LEARN_RATE = args.lr

        self.action_dim = 3
        self.state_dim = (84,84)
        self.epsilon = 0.0
        self.update_count = 0
        
        self.policy_net = CNN(84, 84, self.action_dim).to(device)
        self.target_net = CNN(84, 84, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.LEARN_RATE)
        
        self.memory = ReplayMemory(args.mem_capacity)
        self.interaction_steps = 0

    def select_action(self, state):
        self.interaction_steps += 1
        self.epsilon = self.EPS_END + np.maximum( (self.EPS_START-self.EPS_END) * (1 - self.interaction_steps/self.EPS_DECAY), 0)
        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

    def evaluate_action(self, state, rand=0.1):
        if random.random() < rand:
            return torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
        with torch.no_grad():
            return self.target_net(state).max(1)[1].view(1, 1)


    def store(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def update(self):
        if len(self.memory) < self.BATCH_SIZE:
            print("[Warning] Memory data less than batch sizes!")
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        final_mask = torch.cat(batch.done)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.BATCH_SIZE,1, device=device)
        next_state_values[final_mask.bitwise_not()] = self.target_net(non_final_next_states).max(1, True)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
    
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % args.target_step == 0:
            self.update_target_net()

        
    def update_target_net(self):
        with torch.no_grad():
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # Partially update weight
            #for q, q_targ in zip(self.policy_net.parameters(), self.target_net.parameters()):
            #    q_targ.data.mul_(0.05)
            #    q_targ.data.add_(0.95 * q.data)
            #self.target_net.eval()

    def save_model(self, path="."):
        torch.save(self.target_net.state_dict(), path+'/q_target_checkpoint_{}.pth'.format(self.interaction_steps))
        torch.save(self.policy_net.state_dict(), path+'/q_policy_checkpoint_{}.pth'.format(self.interaction_steps))

    def restore_model(self, path):
        self.target_net.load_state_dict(torch.load(path, map_location=device))
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.eval()
        print("[Info] Restore model from '%s' !"%path)

frame_proc = T.Compose([T.ToPILImage(),
                        T.Grayscale(), \
                        T.Resize((84,84), interpolation=Image.BILINEAR), \
                        T.ToTensor()])

class Atari(object):
    def __init__(self, env_name="FreewayNoFrameskip-v4", agent_history_length=4):
        gym.register_envs(ale_py)
        self.env = gym.make(env_name)
        self.state = None
        self.agent_history_length = agent_history_length

    def reset(self):
        observation = self.env.reset()
        frame = self.image_proc(observation[0]).to(device)
        self.state = frame.repeat(1,self.agent_history_length,1,1)
        return self.state

    def image_proc(self, image):
        # Apply frame processing
        processed_image = frame_proc(image)
        # Save the processed image to a file
        # processed_image_pil = T.ToPILImage()(processed_image)  # Convert tensor to PIL image
        # processed_image_pil.save("processed_image.png")  # Save the image
        # import ipdb; ipdb.set_trace()
        return processed_image

    def step(self, action):
        observation, reward, done, info, _ = self.env.step(action)
        # import ipdb; ipdb.set_trace()
        # if done:
        #     processed_image_pil = T.ToPILImage()(observation)  # Convert tensor to PIL image
        #     processed_image_pil.save("processed_image.png")  # Save the image
        #     import ipdb; ipdb.set_trace()
        frame = self.image_proc(observation).to(device)
        next_state = torch.cat( (self.state[:, 1:, :, :], frame.unsqueeze(0)), axis=1 )
        self.state = next_state
        return next_state, reward, done, info

    def get_render(self):
        observation = self.env.render(mode='human')
        return observation

    def render(self):
        self.env.render()


def train():
    num_episodes = args.train_ep
    save_model_per_ep = args.save_per_ep
    log_fd = open(args.log_file,'w')

    agent = DQN()
    env = Atari()

    if args.load_model:
      agent.restore_model(args.load_model)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    global_steps = 0
    for i_episode in range(num_episodes):
        episode_reward = 0
        state = env.reset()

        for _ in range(10000):
            # env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            # print("done", done)
            
            if done:
                next_state = None

            agent.memory.push(  state, \
                                action, \
                                next_state, \
                                torch.tensor([[reward]], device=device), \
                                torch.tensor([done], device=device, dtype=torch.bool))
            state = next_state
            episode_reward += reward
            global_steps += 1 

            if global_steps > 50000:
                agent.update()

            if done:
                train_info_str = "Episode: %6d, interaction_steps: %6d, reward: %2d, epsilon: %f"%(i_episode, agent.interaction_steps, episode_reward, agent.epsilon)
                print(train_info_str)
                log_fd.write(train_info_str)
                break
        
        wandb.log({"episode_reward": episode_reward})
        
        if i_episode % save_model_per_ep == 0:
            agent.save_model(args.save_dir)
            print("[Info] Save model at '%s' !"%args.save_dir)

        if i_episode % args.eval_per_ep == 0:
            test_env = Atari()
            average_reward = 0
            test_ep = 5

            for t_ep in range(test_ep):
                state = test_env.reset()
                episode_reward = 0
                for _ in range(10000):
                    action = agent.evaluate_action(state)
                    state, reward, done, _ = test_env.step(action.item())
                    # print("done_test", done)
                    episode_reward += reward
                average_reward += episode_reward
            average_reward /= test_ep
            
            eval_info_str = "Evaluation: True, Episode: %6d, Interaction_steps: %6d, evaluate reward: %f"%(i_episode, agent.interaction_steps, average_reward)
            print(eval_info_str)
            log_fd.write(eval_info_str)
    
    log_fd.close()

args = parser.parse_args(args=[])

train()