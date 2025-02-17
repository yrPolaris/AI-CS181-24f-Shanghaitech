## 📁 Project Structure

```tree
src/
├── main.py 		# Main training and testing script
├── CLI.py 		# CLI interface
├── model/
│ ├── DQNAgent.py 	# DQN agent implementation
│ └── utils.py 		# Utility functions
├── checkpoint/ 	# Saved Model checkpoints
│ ├── DQN_info/ 	# Info-based DQN
│ └── DQN_image/ 	# Image-based DQN
├─── atariari/ 		# Package for AtariARIWrapper
└─── origin/            # origin version of our code with other algorithms
```

## ⚙️ Configuration Options

The following options are available for configuring the Freeway Agent via the command line:

- `--env`: Environment name (default: `FreewayNoFrameskip-v4`)
- `--model`: Model name (default: `DQN_image`)
- `--ckpt_path`: Load model from path (default: `None`)
- `--num_episodes`: Number of episodes for training (default: `100`)
- `--num_episodes_test`: Number of episodes for testing (default: `5`)
- `--max_timesteps`: Max timesteps per episode (default: `10000`)
- `--batch_size`: Batch size for training (default: `64`)
- `--gamma`: Discount factor (default: `0.99`)
- `--eps_start`: Starting epsilon for exploration (default: `0.9`)
- `--eps_end`: Ending epsilon value (default: `0.1`)
- `--eps_decay`: Epsilon decay rate (default: `1000`)
- `--memory_capacity`: Replay memory capacity (default: `10000`)
- `--lr`: Learning rate (default: `1e-4`)
- `--device`: Device to use for computation (default: `cuda` if available, otherwise `cpu`)
- `--play`: Play mode
- `--render`: Render mode
- `--save_path`: Path to save models (default: `./ckpt`)
- `--save_interval`: Interval for saving models (default: 50)

## 🎯 Usage

### Training

```bash
# Train info-based DQN
python src/main.py --model DQN_info

# Train image-based DQN
python src/main.py --model DQN_image
```

### Testing

```bash
# Test info-based DQN
python src/main.py --model DQN_info --play --ckpt_path checkpoint/DQN_info/best.pth

# Test image-based DQN
python src/main.py --model DQN_image --play --ckpt_path checkpoint/DQN_image/best.pth
#Add --render to see the visualization
```

## 📊 Experiment Tracking

The project uses Weights & Biases (wandb) for experiment tracking. Training metrics and model performance are automatically logged and can be viewed in the wandb dashboard.
