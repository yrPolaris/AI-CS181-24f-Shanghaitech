import numpy as np
import random

import torchvision.transforms as T
from PIL import Image

from collections import namedtuple, deque
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

NUM_CAR_OBS = 10
num_buckets = 20
value_range = 160
bucket_size = value_range // num_buckets

def map_to_bucket(value):
    return min(int(value // bucket_size), num_buckets - 1)

def state_to_q_index(state):

    if "player_y" not in state:
        indices = torch.zeros(NUM_CAR_OBS, device='cuda', dtype=torch.int16)
        return indices
    
    player_y = state['player_y']
    enemy_cars_y = np.linspace(0, 180, num=12, dtype=np.int32)  
    enemy_cars_x = [state[f'enemy_car_x_{i}'] for i in range(10)]
    enemy_cars_x = np.pad(enemy_cars_x, (1, 1), 'constant', constant_values=(0))
    distances = np.abs(np.int32(enemy_cars_y) - np.int32(player_y))
    closest_three_indices = np.sort(np.argsort(distances)[:NUM_CAR_OBS])   

    closest_cars_x = [enemy_cars_x[i] for i in closest_three_indices]
    indices = [map_to_bucket(val) for val in closest_cars_x]
    return indices

def image_proc(image):
    frame_proc = T.Compose([T.ToPILImage(), T.Grayscale(), \
                            T.Resize((84, 84), interpolation=Image.Resampling.BILINEAR), \
                            T.ToTensor()])
    return frame_proc(image)