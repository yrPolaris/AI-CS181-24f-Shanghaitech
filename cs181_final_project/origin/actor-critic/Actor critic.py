import gymnasium as gym
import ale_py
import numpy as np
import random
from atariari.benchmark.wrapper import AtariARIWrapper
import torch
import pygame
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

gym.register_envs(ale_py)
# env = AtariARIWrapper(gym.make('Freeway-ramNoFrameskip-v4', render_mode='human', mode=0))
env = AtariARIWrapper(gym.make('Freeway-ramNoFrameskip-v4', mode=0))


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 128):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim = 128):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ActorCritic:
    def __init__(self, states, actions, device):
        self.device = device
        self.action = actions
        self.actor = PolicyNet(states, actions).to(device)
        self.critic = ValueNet(states, actions).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-3)
        self.gamma = 0.99

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)  
        actions = actions.to(torch.int64)
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  
        critic_loss.backward()  
        self.actor_optimizer.step()  
        self.critic_optimizer.step() 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

episodes = 200
batch_size = 32
buffer_size = 10000
min_buffer_size = 64

# Initialize Replay Buffer and DQN agent
state_dim = 128
action_dim = 3
agent = ActorCritic(state_dim, action_dim, device)
# agent = DQN(state_dim, action_dim, device)
# Training loop with tqdm progress bar
reward_history = []

rewards_per_episode = []
with tqdm(total=episodes, desc="Training Progress", unit="episode") as pbar:
    for episode in range(episodes):
        state = env.reset()[0]

        episode_reward = 0
        terminated = False
        truncated = False

        transition_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }

        # Run the episode
        while not terminated and not truncated:
            action = agent.take_action(state)  # Select action

            next_state, reward, terminated, truncated, _ = env.step(action)

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(terminated or truncated)

            state = next_state
            episode_reward += reward


        agent.update(transition_dict)

        rewards_per_episode.append(episode_reward)


        if (episode + 1) % 10 == 0:
            pbar.set_postfix({'Last 10 Avg Reward': f'{sum(rewards_per_episode[-10:]) / 10:.2f}'})
        pbar.update(1)

# Save trained model
torch.save(agent.actor, "actor_critic_mountain.pth")
# Plot rewards
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Actor-Critic Training Performance')
plt.show()

env.close()




# for t in range(1000):
#     action = epsilon_greedy(state, 0)  # 测试时不使用epsilon-greedy，而是选择最优动作
#     next_observation, reward, terminated, truncated, info = env.step(action)
#     state = info['labels']
#     total_reward += reward

#     if terminated or truncated:
#         break

# print(f"Test Total Reward: {total_reward}")
# env.close()
