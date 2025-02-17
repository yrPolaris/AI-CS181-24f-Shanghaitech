import gymnasium as gym
import ale_py
import numpy as np
import random
from atariari.benchmark.wrapper import AtariARIWrapper

gym.register_envs(ale_py)
env = AtariARIWrapper(gym.make('Freeway-ramNoFrameskip-v4', render_mode='human', mode=0))
# env = AtariARIWrapper(gym.make('Freeway-ramNoFrameskip-v4', mode=0))

alpha = 0.1  
gamma = 0.99  
epsilon = 0.1  
episodes = 10 
max_timesteps = 10000  
value_range = 160
num_buckets = 10
bucket_size = value_range // num_buckets
state_size = num_buckets ** 3
n_actions = 3 

Q_table = np.zeros((num_buckets, num_buckets, num_buckets, n_actions))

def map_to_bucket(value):
    return min(int(value // bucket_size), num_buckets - 1)

# 定义一个函数来将state映射到Q表的索引
def state_to_q_index(state):

    if "player_y" not in state:
        # print(state)
        return [0, 0, 0]
    
    player_y = state['player_y']
    enemy_cars_y = np.linspace(0, 180, num=12, dtype=np.int32)  # 均匀分布的纵坐标
    enemy_cars_x = [state[f'enemy_car_x_{i}'] for i in range(10)]
    enemy_cars_x = np.pad(enemy_cars_x, (1, 1), 'constant', constant_values=(0))
    distances = np.abs(np.int32(enemy_cars_y) - np.int32(player_y))
    closest_three_indices = np.sort(np.argsort(distances)[:3])   

    closest_cars_x = [enemy_cars_x[i] for i in closest_three_indices]

    # 将敌车的横坐标映射到桶
    indices = [map_to_bucket(val) for val in closest_cars_x]

    return indices

# 示例：更新Q表
def update_Q_table(reward, next_state, state, Q_table, action):
    next_q_index = state_to_q_index(state)
    q_index = state_to_q_index(state)
    # print(Q_table[q_index[0], q_index[1], q_index[2],0],Q_table[q_index[0], q_index[1], q_index[2],1],Q_table[q_index[0], q_index[1], q_index[2],2])
    # Q表更新公式
    # Q_table[q_index[0], q_index[1], q_index[2], action] += alpha * (reward + gamma * np.max(Q_table[next_q_index[0], next_q_index[1], next_q_index[2]]) - Q_table[q_index[0], q_index[1], q_index[2], action])
    Q_table[0, 0, q_index[2], action] += alpha * (reward + gamma * np.max(Q_table[0, 0, next_q_index[2]]) - Q_table[0, 0, q_index[2], action])
    print(q_index[2],Q_table[0, 0, q_index[2], 0],Q_table[0, 0, q_index[2], 1],Q_table[0, 0, q_index[2], 2])
    return Q_table

def epsilon_greedy(state, epsilon):
    q_index = state_to_q_index(state)
    # best_action = np.argmax(Q_table[q_index[0], q_index[1], q_index[2]])
    best_action = np.argmax(Q_table[0, 0, q_index[2]])

    # print(q_index)
    # print(Q_table[q_index[0], q_index[1], q_index[2],0], Q_table[q_index[0], q_index[1], q_index[2],1], Q_table[q_index[0], q_index[1], q_index[2],2])
    # print(best_action)
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, n_actions - 1)  # 随机选择一个动作
    else:
        return best_action  # 选择最优动作

for episode in range(episodes):
    observation, info = env.reset()
    state = info
    
    episode_reward = 0
    pre_info_y = 0
    for t in range(max_timesteps):
        action = epsilon_greedy(state, epsilon)  # 使用epsilon-greedy策略
        next_observation, reward, terminated, truncated, info = env.step(action) 
        print("terminated", terminated)
        next_state = info['labels']  # 记录下一状态
        reward_y = int(info['labels']['player_y']) - int(pre_info_y)  # 基于y轴的位置变化计算奖励

        if((action == 1 or action == 0) and reward_y < 0):
            reward_y *= 10
        if(reward == 1):
            reward_y = 0
        pre_info_y = info['labels']['player_y']
        
        # 更新Q表，奖励乘以100并调整
        update_Q_table(reward_y + reward * 100 - 0.01, next_state, state, Q_table, action)
        
        state = next_state  # 转到下一个状态
        episode_reward += reward

        if terminated or truncated:
            break

    print(f"Episode {episode + 1}/{episodes}, Total Reward: {episode_reward}")

# 保存训练得到的Q表
np.save('q_table.npy', Q_table)
print("Q-table saved to 'q_table.npy'")

# 训练完成后，测试训练好的Q-learning模型
observation, info = env.reset()
state = info['labels']
total_reward = 0

for t in range(1000):
    action = epsilon_greedy(state, 0)  # 测试时不使用epsilon-greedy，而是选择最优动作
    next_observation, reward, terminated, truncated, info = env.step(action)
    state = info['labels']
    total_reward += reward

    if terminated or truncated:
        break

print(f"Test Total Reward: {total_reward}")
env.close()
