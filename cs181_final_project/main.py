from CLI import parse_args
from model.DQNAgent import DQNAgent_info, DQNAgent_image
from model.utils import *
import gymnasium as gym
from atariari.benchmark.wrapper import AtariARIWrapper
import wandb
import os


def train_and_play(args, env, agent):
    num_episodes = args.num_episodes if not args.play else args.num_episodes_test
    test_total_reward = 0

    for i_episode in range(num_episodes):
        episode_reward = 0
        state, info = agent.reset()
        if args.model == "DQN_info":
            obs = state_to_q_index(info)
            state = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            y_pos = 0

        for t in range(args.max_timesteps):
            if args.play:
                action = agent.select_action(state, train=False)
            else:
                action = agent.select_action(state, train=True)
            
            observation, reward, terminated, truncated, info = agent.step(action)
            reward_image = reward
            episode_reward += reward
            test_total_reward += reward
            if args.model == "DQN_info":    
                new_y_pos = info['labels']['player_y']
                reward_y = int(new_y_pos) - int(y_pos) 
                binned_obs = state_to_q_index(info['labels'])

                if((action == 1 or action == 0)):
                    if reward_y < 0:
                        reward_y *= 10
                    else:
                        reward_y += (new_y_pos*10)/180 
                if(reward == 1):
                    reward_y = 0
                y_pos = new_y_pos

                reward_info = reward_y + reward * 50 - 0.01
                reward = torch.tensor([reward_info], device=agent.device)

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(binned_obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            
            elif args.model == "DQN_image":
                next_state = observation
                reward = torch.tensor([reward_image], device=agent.device)
           
            done = terminated or truncated 

            if not args.play:
                agent.store_transition(state, action, next_state, reward, done)
                agent.update()
            
            state = next_state
 
            if done:
                break
                
        if not args.play:
            wandb.log({"episode_reward": episode_reward})
            if i_episode % args.save_interval == 0:
                agent.save(args.model, args.save_path)
    
        print(f"[episode {i_episode}] reward: {episode_reward}")

    if args.play:
        print(f"Test finished, rollout {args.num_episodes_test} episodes, total reward: {test_total_reward}, average reward: {test_total_reward / args.num_episodes_test}")

    if not args.play:
        wandb.finish()
    print('Complete')

def main():
    args = parse_args()

    # import os
    # os.environ["WANDB_MODE"] = "offline"
    if not args.play:
        wandb.init(project="freeway_ai", config={
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "num_episodes": args.num_episodes,
            "eps_start": args.eps_start,
            "eps_end": args.eps_end,
            "eps_decay": args.eps_decay,
            "memory_capacity": args.memory_capacity,
            "model": args.model,
            "env": args.env
        })
    
    if args.render:
        env = gym.make(args.env, render_mode="human")
    else:
        env = gym.make(args.env)
    env = AtariARIWrapper(env)

    if args.model == "DQN_info":
        agent = DQNAgent_info(
            batch_size=args.batch_size,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
            device=args.device,
            lr=args.lr,
            env=env
        )
    elif args.model == "DQN_image":
        agent = DQNAgent_image(
            batch_size=args.batch_size,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
            device=args.device,
            lr=args.lr,
            env=env
        )

    
    if args.play:
        agent.load(args.ckpt_path)
        
    train_and_play(args, env, agent)

if __name__ == "__main__":
    main()
