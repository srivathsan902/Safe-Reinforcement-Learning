import argparse
import os
import shutil
import sys
import time
import torch
import numpy as np
import safety_gymnasium
from ddpgAgent import DDPGAgent
from plotting import setup_plotting
from PyQt5.QtCore import QTimer
from train import train, train_with_plot

artifacts_folder = 'artifacts'

def main(dir_name, plot=False):
    env_id = 'SafetyPointCircle1-v0'
    env = safety_gymnasium.make(env_id)
    # env = safety_gymnasium.make(env_id, render_mode='human')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    agent = DDPGAgent(state_dim, action_dim, max_action, min_action, hidden_size_1= 64, hidden_size_2 = 128, Prioritized_buffer=True)

    num_episodes = 1000
    batch_size = 64

    episode_rewards = []
    episode_costs = []

    # Load the latest models if they exist
    start_episode = 0
    if os.path.exists(dir_name) and len(os.listdir(dir_name)) > 0:
        latest_episode = max(int(ep) for ep in os.listdir(dir_name) if ep.isdigit())
        agent.load_agent(dir_name, latest_episode)
        start_episode = latest_episode
    # Load existing reward and cost logs if they exist
    if os.path.exists(os.path.join(dir_name,'episode_rewards.npy')) and os.path.exists(os.path.join(dir_name,'episode_costs.npy')):
        episode_rewards = np.load(os.path.join(dir_name,'episode_rewards.npy')).tolist()
        episode_costs = np.load(os.path.join(dir_name,'episode_costs.npy')).tolist()


    if plot:
        app, window_reward, window_cost = setup_plotting()
        for episode, reward in enumerate(episode_rewards):
            window_reward.add_data(episode, reward)
        for episode, cost in enumerate(episode_costs):
            window_cost.add_data(episode, cost)
        training_gen = train_with_plot(env, agent, dir_name, num_episodes, batch_size, start_episode)
        
        def update():

            try:
                episode, episode_reward, episode_cost = next(training_gen)
                episode_rewards.append(episode_reward)
                episode_costs.append(episode_cost)
                window_reward.add_data(episode, episode_reward)
                window_cost.add_data(episode, episode_cost)
                
            except StopIteration:
                np.save(os.path.join(dir_name,'episode_rewards.npy'), np.array(episode_rewards))
                np.save(os.path.join(dir_name,'episode_costs.npy'), np.array(episode_costs))

                QTimer.singleShot(0, app.quit)

        
        timer = QTimer()
        timer.timeout.connect(update)
        timer.start(100)

        sys.exit(app.exec_())

    else:

        new_episode_rewards = []
        new_episode_costs = []

        try:
            new_episode_rewards, new_episode_costs = train(env, agent, dir_name, num_episodes, batch_size, start_episode, plot=False)
        except ValueError as e:
            print(f"Error occurred during training: {e}")
    
        episode_rewards.extend(new_episode_rewards)
        episode_costs.extend(new_episode_costs)

        np.save(os.path.join(dir_name,'episode_rewards.npy'), np.array(episode_rewards))
        np.save(os.path.join(dir_name,'episode_costs.npy'), np.array(episode_costs))


if __name__ == '__main__':
    '''
    Create the dir name based on the current time: dd_mm_yyyy_hh_mm_ss
    artifacts/yyyy/mm/dd/hh_mm should be the structure
    '''

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Script to perform tasks.')

    # Arguments for directory update and plotting
    parser.add_argument('--update_from', metavar='RUN_NUMBER',
                        help='Update from previous run directory specified by RUN_NAME (e.g., Run_1)')
    parser.add_argument('--plot', action='store_true', help='Enable plotting')

    args = parser.parse_args()

    
    dir_name = os.path.join(artifacts_folder,
                        time.strftime('%Y'),
                        time.strftime('%m'),
                        time.strftime('%d'))
    
    run = 1
    while os.path.exists(os.path.join(dir_name, f'Run_{run}')):
        run += 1
    dir_name = os.path.join(dir_name, f'Run_{run}')

    try:
        # Check if --update_from argument is provided
        if args.update_from:
            run_name_parts = args.update_from.split('_')
            if len(run_name_parts) != 2 or not run_name_parts[1].isdigit():
                raise ValueError("Invalid format for RUN_NAME. Should be like Run_1, Run_2, etc.")
            old_run = int(run_name_parts[1])
            old_dir_name = os.path.join(os.path.dirname(dir_name), f'Run_{old_run}')

            if os.path.exists(old_dir_name):
                # Create new directory if it doesn't exist
                os.makedirs(dir_name, exist_ok=True)

                # Copy contents from old_dir_name to new dir_name
                for item in os.listdir(old_dir_name):
                    old_item_path = os.path.join(old_dir_name, item)
                    new_item_path = os.path.join(dir_name, item)
                    if os.path.isdir(old_item_path):
                        shutil.copytree(old_item_path, new_item_path)
                    else:
                        shutil.copy2(old_item_path, new_item_path)

                print(f"Contents from '{old_dir_name}' copied to '{dir_name}'.")
            else:
                raise FileNotFoundError(f"Directory '{old_dir_name}' does not exist.")

        else:
            os.makedirs(dir_name, exist_ok=True)
        
        plot = args.plot

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

    main(dir_name, args.plot)
    


