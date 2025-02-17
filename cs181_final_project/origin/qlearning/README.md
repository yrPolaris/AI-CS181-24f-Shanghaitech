# Project Setup and Instructions

## Installation

To get started with the project, follow the instructions provided in the official documentation:  
[Getting Started with ALE](https://ale.farama.org/getting-started/)

## About the Environment

The environment has been preprocessed to provide the positions of the chicken and each car directly. For more details on how to interact with the environment, refer to the `keyboard.py` file for basic usage.

### Controls:
- Press **W** to move forward.
- Press **S** to move backward.

## Q-learning Overview

### Q-learning with i Lines

The script `Q-learning_with_i_line.py` is designed to handle different numbers of lines in the state space. The state space contains at most 3 car lines: the front, current, and back lines.

 The "bucket trick" is used to reduce the state space size. Each line is discretized into 10 or 20 blocks that represent the positions of cars.

### Q-learning with 2 Lines

The file `Q-learning_with_2_line.py` has been fine-tuned and is the only version that has been optimized. The Q-table learned from this script is saved in the file `q_table.npy`.

## Testing the Policy

During training, the agent uses an epsilon-greedy strategy to select actions. 

During testing, the action selection is purely based on the learned policy.

To test the policy, run the `test.py` script.
