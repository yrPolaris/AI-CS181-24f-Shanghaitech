import os

import numpy as np

from .Agent import Agent
from . import utils

class QLearningAgent(Agent):
    
    def __init__(self, **kwargs):
        self.ALPHA = kwargs.get('alpha', 0.1)
        self.GAMMA = kwargs.get('gamma', 0.99)
        self.EPS_START = kwargs.get('eps_start', 0.9)
        self.EPS_END = kwargs.get('eps_end', 0.1)
        self.EPS_DECAY = kwargs.get('eps_decay', 1000)
        
        self.n_action = 3
        self.num_buckets = 10
        self.num_obs = kwargs.get('num_obs', 2)
        
        self.q_table = np.zeros((self.num_buckets,self.num_buckets,self.num_buckets,self.n_action))
        
    def state_to_q_index(self, state):

        if "player_y" not in state:
            # print(state)
            return [0, 0, 0]
        
        player_y = state['player_y']
        enemy_cars_y = np.linspace(20, 176, num=10, dtype=np.int32)  # 均匀分布的纵坐标

        enemy_cars_x = [state[f'enemy_car_x_{i}'] for i in range(10)]
        

        distances = np.abs(np.int32(enemy_cars_y) - np.int32(player_y))

        closest_three_indices = np.argmin(distances)
        
        enemy_cars_x = np.pad(enemy_cars_x, (1, 1), 'constant', constant_values=(0))
        
        closest_cars_x = [enemy_cars_x[i] for i in range(closest_three_indices, closest_three_indices + 3)]

        # 将敌车的横坐标映射到桶
        indices = [utils.map_to_bucket(val) for val in closest_cars_x]

        return indices
    
    def update(self, reward, next_state, state, action, **kwargs):
        next_q_index = self.state_to_q_index(state)
        q_index = self.state_to_q_index(state)
        index_mask = np.array(0 if self.num_obs != 3 else 1, 0 if self.num_obs == 1 else 1, 1)
        next_q_index = np.array(next_q_index[i]*index_mask[i] for i in range(3))
        q_index = np.array(q_index[i]*index_mask[i] for i in range(3))
        self.q_table[q_index[0], q_index[1], q_index[2], action] += \
            self.ALPHA * (reward + self.GAMMA * np.max(self.q_table[next_q_index[0], next_q_index[1], next_q_index[2]])) \
                -self.q_table[q_index[0], q_index[1], q_index[2], action]
        return self.q_table
    
    def save(self, path, **kwargs):
        np.save(os.path.join(path, 'q_table.npy'). self.q_table)
        
    def load(self, path, **kwargs):
        self.q_table = np.load(path)