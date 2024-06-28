from tqdm import tqdm
import numpy as np
import os
from safetyPolicy import select_safe_action


def train(env, agent, dir_name, params, start_episode = 0):
    
    num_episodes = params['train'].get('num_episodes', 1000)
    batch_size = params['train'].get('batch_size', 64)
    max_steps_per_episode = params['train'].get('max_steps_per_episode', 250)

    SAVE_EVERY = params['train'].get('save_every', 100)

    episode_rewards = []
    episode_costs = []

    for episode in tqdm(range(start_episode, start_episode + num_episodes), desc='Training'):
        state, info = env.reset()
        agent.ou_noise.reset()
        episode_reward = 0
        episode_cost = 0
        num_safe_actions = 0
        for t in range(max_steps_per_episode):
            print('Episode:', episode, 'Step:', t, end= "\r")
            action = agent.select_action(np.array(state))
            # '''
            # If original action was itself safe, then it will be returned,
            # else a safe action will be returned.
            # '''
            # action, safe = select_safe_action(env, action)

            # if not safe:
            #     print('Could not find safe action')
            #     break
            
            next_state, reward, cost, done, truncated, _ = env.step(action)
            # if cost > 100:
            #     done = True
            #     truncated = True
            if cost == 0:
                num_safe_actions += 1
            agent.replay_buffer.add(state, action, reward, next_state, cost, done)

            state = next_state
            episode_reward += reward
            episode_cost += cost

            if agent.replay_buffer.size > batch_size:
                agent.train(batch_size)

            if done or truncated:
                break

        # episode_rewards.append(episode_reward)
        # episode_costs.append(episode_cost)
        os.system('cls' if os.name == 'nt' else 'clear')

        # if plot:
        #     yield episode, episode_reward, episode_cost

        if (episode + 1) % SAVE_EVERY == 0:
            agent.save(os.path.join(dir_name , f'{episode + 1}'))
        percent_safe_actions = num_safe_actions / max_steps_per_episode * 100
        yield episode, episode_reward, episode_cost, percent_safe_actions
    
    env.close()
    
    # if not plot:
    #     return episode_rewards, episode_costs

def train_with_plot(env, agent, dir_name, params, start_episode = 0):

    num_episodes = params['train'].get('num_episodes', 1000)
    batch_size = params['train'].get('batch_size', 64)
    max_steps_per_episode = params['train'].get('max_steps_per_episode', 250)

    SAVE_EVERY = params['train'].get('save_every', 100)

    episode_rewards = []
    episode_costs = []

    for episode in tqdm(range(start_episode, start_episode + num_episodes), desc='Training'):
        state, info = env.reset()
        agent.ou_noise.reset()
        episode_reward = 0
        episode_cost = 0
        num_safe_actions = 0
        for t in range(max_steps_per_episode):
            # print('Episode:', episode, 'Step:', t)
            action = agent.select_action(np.array(state))

            # '''
            # If original action was itself safe, then it will be returned,
            # else a safe action will be returned.
            # '''
            # action, safe = select_safe_action(env, action)

            # if not safe:
            #     print('Could not find safe action')
            #     break
            
            next_state, reward, cost, done, truncated, _ = env.step(action)
            # if cost > 0:
            #     print(cost)
            if cost == 0:
                num_safe_actions += 1
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

        if (episode + 1) % SAVE_EVERY == 0:
            agent.save(os.path.join(dir_name , f'{episode + 1}'))
        percent_safe_actions = percent_safe_actions = num_safe_actions / max_steps_per_episode * 100
        yield episode, episode_reward, episode_cost, percent_safe_actions
    
    env.close()
