import os
import sys
import time
import yaml
import torch
import wandb
import shutil
import argparse
import numpy as np
import safety_gymnasium
from dotenv import load_dotenv
from PyQt5.QtCore import QTimer
from ddpgAgent import DDPGAgent
from plotting import setup_plotting
from train import train, train_with_plot

load_dotenv()

artifacts_folder = 'artifacts'

def main(dir_name, params):

    env_id = params['main'].get('env_id', 'SafetyPointCircle1-v0')
    render_mode = params['main'].get('render_mode', None)
    if render_mode == 'None':
        env = safety_gymnasium.make(env_id)
    else:
        env = safety_gymnasium.make(env_id, render_mode = render_mode)

    plot = params['main'].get('plot', False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    hidden_size_1 = params['main']['agent'].get('hidden_size_1', 64)
    hidden_size_2 = params['main'].get('hidden_size_2', 128)
    Prioritized_buffer = params['main'].get('Prioritized_buffer', True)

    agent = DDPGAgent(state_dim, action_dim, max_action, min_action, hidden_size_1 = hidden_size_1, hidden_size_2 = hidden_size_2, Prioritized_buffer = Prioritized_buffer)

    episode_rewards = []
    episode_costs = []

    # Load the latest models if they exist
    start_episode = 0
    if os.path.exists(dir_name) and len(os.listdir(dir_name)) > 0:
        latest_episode = max(int(ep) for ep in os.listdir(dir_name) if ep.isdigit())
        agent.load_agent(dir_name, latest_episode)
        start_episode = latest_episode
    # Load existing reward and cost logs if they exist
    if os.path.exists(os.path.join(dir_name,'episode_rewards.npy')):
        episode_rewards = np.load(os.path.join(dir_name,'episode_rewards.npy')).tolist()
    else:
        episode_rewards = [0]*start_episode

    if os.path.exists(os.path.join(dir_name,'episode_costs.npy')):
        episode_costs = np.load(os.path.join(dir_name,'episode_costs.npy')).tolist()
    else:
        episode_costs = [0]*start_episode

    if os.path.exists(os.path.join(dir_name,'episode_percent_safe_actions.npy')):
        episode_percent_safe_actions = np.load(os.path.join(dir_name,'episode_percent_safe_actions.npy')).tolist()
    else:
        episode_percent_safe_actions = [0]*start_episode

    wandb_enabled = params['base']['wandb_enabled']

    if wandb_enabled:
        try:
            wandb_api_key = os.getenv('WANDB_API_KEY')
            wandb.login(key=wandb_api_key)

        except Exception as e:
            print(f"Error occurred while logging into wandb: {e}")
            sys.exit(1)

        wandb.init(project='testing', name = f'{env_id}', config = params)
        
    if wandb_enabled:

        for i in range(len(episode_rewards)):

            wandb.log({
                'Reward': episode_rewards[i],
                'Cost': episode_rewards[i]
            })

    if plot:
        app, window_reward, window_cost = setup_plotting()
        for episode, reward in enumerate(episode_rewards):
            window_reward.add_data(episode, reward)
        for episode, cost in enumerate(episode_costs):
            window_cost.add_data(episode, cost)
        training_gen = train_with_plot(env, agent, dir_name, params, start_episode)
        
        def update():

            try:
                episode, episode_reward, episode_cost, percent_safe_actions = next(training_gen)
                episode_rewards.append(episode_reward)
                episode_costs.append(episode_cost)
                window_reward.add_data(episode, episode_reward)
                window_cost.add_data(episode, episode_cost)

                if wandb_enabled:
                    wandb.log({
                        'Reward': episode_reward,
                        'Cost': episode_cost,
                        '% Safe Actions': percent_safe_actions
                    })
                
            except StopIteration:

                if wandb_enabled:
                    wandb.finish()

                np.save(os.path.join(dir_name,'episode_rewards.npy'), np.array(episode_rewards))
                np.save(os.path.join(dir_name,'episode_costs.npy'), np.array(episode_costs))

                QTimer.singleShot(0, app.quit)

        
        timer = QTimer()
        timer.timeout.connect(update)
        timer.start(100)

        sys.exit(app.exec_())

    else:

        training_gen = train(env, agent, dir_name, params, start_episode)

        try:
            # new_episode_rewards, new_episode_costs = train(env, agent, dir_name, params, start_episode)
            for episode, reward, cost, percent_safe_actions in training_gen:
                episode_rewards.append(reward)
                episode_costs.append(cost)
                episode_percent_safe_actions.append(percent_safe_actions)
                
                if wandb_enabled:
                    wandb.log({
                        'Reward': reward,
                        'Cost': cost,
                        '% Safe Actions': percent_safe_actions,
                    })

        except ValueError as e:
            print(f"Error occurred during training: {e}")

        if wandb_enabled:
            wandb.finish()

        np.save(os.path.join(dir_name,'episode_rewards.npy'), np.array(episode_rewards))
        np.save(os.path.join(dir_name,'episode_costs.npy'), np.array(episode_costs))

if __name__ == '__main__':

    with open('src/params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    '''
    Create the dir name based on the current time: dd_mm_yyyy_hh_mm_ss
    artifacts/yyyy/mm/dd/hh_mm should be the structure
    '''
    dir_name = os.path.join(artifacts_folder,
                        time.strftime('%Y'),
                        time.strftime('%m'),
                        time.strftime('%d'))
    
    run = 1
    while os.path.exists(os.path.join(dir_name, f'Run_{run}')):
        run += 1
    dir_name = os.path.join(dir_name, f'Run_{run}')

    try:
        if params['main'].get('update', False):
            update_from = params['main'].get('update_from', False)

            if update_from:
                if os.path.exists(update_from):
                    # Create new directory if it doesn't exist
                    os.makedirs(dir_name, exist_ok=True)

                    # Copy contents from old_dir_name to new dir_name
                    for item in os.listdir(update_from):
                        old_item_path = os.path.join(update_from, item)
                        new_item_path = os.path.join(dir_name, item)
                        if os.path.isdir(old_item_path):
                            shutil.copytree(old_item_path, new_item_path)
                        else:
                            shutil.copy2(old_item_path, new_item_path)

                    print(f"Contents from '{update_from}' copied to '{dir_name}'.")
                else:
                    raise FileNotFoundError(f"Directory '{update_from}' does not exist.")

            else:
                os.makedirs(dir_name, exist_ok=True)

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)
    print('Reached here')
    main(dir_name, params)
    


