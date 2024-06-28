import os
import sys
import yaml
import wandb
import numpy as np
import safety_gymnasium
from dotenv import load_dotenv
from ddpgAgent import DDPGAgent

load_dotenv()

with open('src/params.yaml', 'r') as f:
    params = yaml.safe_load(f)  

wandb_enabled = params['base'].get('wandb_enabled', False)
models_dir = params['test'].get('models_dir', False)
model_nums = params['test'].get('model_nums', [])
env_id = params['main'].get('env_id', 'SafetyPointCircle1-v0')
render_mode = params['test'].get('render_mode', None)
num_runs = params['test'].get('num_runs', 5)
max_steps_per_episode = min(params['test'].get('max_steps_per_episode', 500), 500)

if render_mode == 'None':
    env = safety_gymnasium.make(env_id)
else:
    env = safety_gymnasium.make(env_id, render_mode = render_mode)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high
min_action = env.action_space.low

agent = DDPGAgent(state_dim, action_dim, max_action, min_action, hidden_size_1 = 64, hidden_size_2 = 128, Prioritized_buffer=True)
env.reset()

if wandb_enabled:
    try:
        wandb_api_key = os.getenv('WANDB_API_KEY')
        wandb.login(key=wandb_api_key)
    except Exception as e:
        print(f"Error occurred while logging into wandb: {e}")
        sys.exit(1)

for model_num in model_nums:
    if wandb_enabled:
        wandb.init(project='testing', name=f'{model_num}', config = params)

    print(f'Model {model_num} testing:')
    # Load the trained model
    try:
        agent.load(os.path.join(models_dir, f'{model_num}'))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    rewards_across_runs = []
    costs_across_runs = []
    num_safe_actions_across_runs = []

    for run in range(num_runs):
        state, info = env.reset()
        agent.ou_noise.reset()

        rewards_across_steps = []
        costs_across_steps = []
        num_safe_actions_across_steps = []

        for step in range(max_steps_per_episode):
            action = agent.select_action(np.array(state), noise_enabled=False)
        
            next_state, reward, cost, done, truncated, _ = env.step(action)
    
            rewards_across_steps.append(reward)
            costs_across_steps.append(cost)

            if cost > 0:
                num_safe_actions_across_steps.append(0)
            else:
                num_safe_actions_across_steps.append(1)

            state = next_state
            if done or truncated:
                break

        rewards_across_runs.append(rewards_across_steps)
        costs_across_runs.append(costs_across_steps)
        num_safe_actions_across_runs.append(num_safe_actions_across_steps)

        # print(f'Run {run+1} Reward: {sum(rewards_across_runs[-1])}, Run {run+1} Cost: {sum(costs_across_runs[-1])}, Run {run+1} No. of Safe actions: {sum(num_safe_actions_across_runs[-1])}')
    
    
    mean_rewards_across_runs = np.mean(np.array(rewards_across_runs), axis=0)
    mean_costs_across_runs = np.mean(np.array(costs_across_runs), axis=0)
    mean_num_safe_actions_across_runs = np.mean(np.array(num_safe_actions_across_runs), axis=0)
    
    percent_safe_actions_till_step = np.cumsum(mean_num_safe_actions_across_runs) / np.arange(1, len(mean_num_safe_actions_across_runs) + 1)

    if wandb_enabled:
        reward_till_step = 0
        cost_till_step = 0

        for i in range(len(mean_rewards_across_runs)):
                reward_till_step += mean_rewards_across_runs[i]
                cost_till_step += mean_costs_across_runs[i]
                wandb.log({
                    'Reward': reward_till_step,
                    'Cost': cost_till_step,
                    '% Safe Actions': percent_safe_actions_till_step[i]*100,
                })
        
        if wandb_enabled:
            wandb.finish()
    

    mean_reward_run = np.sum(np.array(mean_rewards_across_runs))
    mean_cost_run = np.sum(np.array(mean_costs_across_runs)[-1])
    mean_safe_actions_run = np.sum(np.array(mean_num_safe_actions_across_runs))

    print(f'\nMean reward: {mean_reward_run}, Mean cost: {mean_cost_run}, Mean Num safe actions: {int(mean_safe_actions_run)}/{max_steps_per_episode}\n')


env.close()
