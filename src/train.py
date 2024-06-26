from tqdm import tqdm
import numpy as np
import os
import copy
from safetyPolicy import select_safe_action

SAVE_EVERY = 100

def train(env, agent, dir_name, num_episodes = 1000, batch_size = 64, start_episode = 0, plot = False, simulation_env = None):

    episode_rewards = []
    episode_costs = []

    for episode in tqdm(range(start_episode, start_episode + num_episodes), desc='Training'):
        state, info = env.reset()
        agent.ou_noise.reset()
        episode_reward = 0
        episode_cost = 0
        for t in range(250):
            print('Episode:', episode, 'Step:', t, end= "\r")
            action = agent.select_action(np.array(state))
            '''
            If original action was itself safe, then it will be returned,
            else a safe action will be returned.
            '''
            action, safe = select_safe_action(env, action)

            if not safe:
                print('Could not find safe action')
                break
            
            next_state, reward, cost, done, truncated, _ = env.step(action)
            if cost > 0:
                break
            agent.replay_buffer.add(state, action, reward, next_state, cost, done)

            state = next_state
            episode_reward += reward
            episode_cost += cost

            if agent.replay_buffer.size > batch_size:
                agent.train(batch_size)

            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)

        # if plot:
        #     yield episode, episode_reward, episode_cost

        if (episode + 1) % SAVE_EVERY == 0:
            agent.save_agent(dir_name, episode + 1)
    
    env.close()
    
    if not plot:
        return episode_rewards, episode_costs

def train_with_plot(env, agent, dir_name, num_episodes = 1000, batch_size = 64, start_episode = 0, simulation_env = None):

    episode_rewards = []
    episode_costs = []

    for episode in tqdm(range(start_episode, start_episode + num_episodes), desc='Training'):
        state, info = env.reset()
        agent.ou_noise.reset()
        episode_reward = 0
        episode_cost = 0
        for t in range(250):
            # print('Episode:', episode, 'Step:', t)
            action = agent.select_action(np.array(state))

            '''
            If original action was itself safe, then it will be returned,
            else a safe action will be returned.
            '''
            action, safe = select_safe_action(env, action)

            if not safe:
                print('Could not find safe action')
                break
            
            next_state, reward, cost, done, truncated, _ = env.step(action)
            if cost > 0:
                print(cost)
            agent.replay_buffer.add(state, action, reward, next_state, cost, done)

            state = next_state
            episode_reward += reward
            episode_cost += cost

            if agent.replay_buffer.size > batch_size:
                agent.train(batch_size)

            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)

        yield episode, episode_reward, episode_cost

        if (episode + 1) % SAVE_EVERY == 0:
            agent.save_agent(dir_name, episode + 1)
    
    env.close()
