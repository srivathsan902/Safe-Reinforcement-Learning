import safety_gymnasium
from ddpgAgent import DDPGAgent
import torch
import numpy as np
import tkinter as tk


model_dir = 'artifacts/2024/06/24/Run_1'
# 100, 500, 700, 1000, 1100, 1500, 1700, 2000, 2100, 2500, 2700, 3000, 3100, 3200, 3300, 3400, 
model_nums = [1000]
env_id = 'SafetyPointCircle1-v0'
env = safety_gymnasium.make(env_id, render_mode='human')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_action = float(env.action_space.low[0])

agent = DDPGAgent(state_dim, action_dim, max_action, min_action)
env.reset()

for model_num in model_nums:
    print(f'********{model_num}********')
    # Load the trained model
    agent.actor.load_state_dict(torch.load(f'{model_dir}/{model_num}/actor_{model_num}.pth'))
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    agent.critic.load_state_dict(torch.load(f'{model_dir}/{model_num}/critic_{model_num}.pth'))
    agent.critic_target.load_state_dict(agent.critic.state_dict())

    mean_reward = []
    mean_cost = []

    for episode in range(5):
        state, info = env.reset()
        agent.ou_noise.reset()
        episode_reward = 0
        episode_cost = 0

        for t in range(500):
            action = agent.select_action(np.array(state), noise_enabled=False)
            next_state, reward, cost, done, truncated, _ = env.step(action)
    
            episode_reward += reward
            episode_cost += cost

            state = next_state
            if done or truncated:
                break
        mean_reward.append(episode_reward)
        mean_cost.append(episode_cost)
        print(f'Episode reward: {episode_reward}, Episode cost: {episode_cost}')

    mean_reward = np.mean(np.array(mean_reward))
    mean_cost = np.mean(np.array(mean_cost))

    print(f'Mean reward: {mean_reward}, Mean cost: {mean_cost}')

env.close()
