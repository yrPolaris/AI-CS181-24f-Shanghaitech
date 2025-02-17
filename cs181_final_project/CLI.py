import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Freeway Agent CLI')
    parser.add_argument('--env', type=str, default='FreewayNoFrameskip-v4', help='Environment name')
    parser.add_argument('--model', type=str, default='DQN_image', help='Model name')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Load model from path')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--num_episodes_test', type=int, default=5, help='Number of episodes for testing')
    parser.add_argument('--max_timesteps', type=int, default=10000, help='Max timesteps per episode')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_start', type=float, default=0.9, help='Starting epsilon')
    parser.add_argument('--eps_end', type=float, default=0.1, help='Ending epsilon')
    parser.add_argument('--eps_decay', type=int, default=1000, help='Epsilon decay')
    parser.add_argument('--memory_capacity', type=int, default=10000, help='Replay memory capacity')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--play', action='store_true', help='Play mode')
    parser.add_argument('--render', action='store_true', help='Render mode')
    parser.add_argument('--save_path', type=str, default='./ckpt', help='Save path')
    parser.add_argument('--save_interval', type=int, default=1, help='Save interval')
    return parser.parse_args()
