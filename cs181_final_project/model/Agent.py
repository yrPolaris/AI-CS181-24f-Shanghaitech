import gymnasium as gym
import ale_py as ale

import os
import wandb

import numpy as np
import random

from . import utils

class Agent(object):
    
    def __init__(self):
        pass

    def update(self, **kwargs):
        raise NotImplementedError
    
    def save(self, path, **kwargs):
        raise NotImplementedError
    
    def load(self, path, **kwargs):
        raise NotImplementedError
    
    def select_action(self, **kwargs):
        raise NotImplementedError