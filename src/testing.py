import safety_gymnasium
from ddpgAgent import DDPGAgent
import numpy as np

model_dir = 'artifacts/2024/06/27/Run_1'

model_nums = [500]
env_id = 'SafetyPointCircle1-v0'
# env = safety_gymnasium.make(env_id)
env = safety_gymnasium.make(env_id, render_mode='human')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high
min_action = env.action_space.low

agent = DDPGAgent(state_dim, action_dim, max_action, min_action, hidden_size_1 = 64, hidden_size_2 = 128, Prioritized_buffer=True)
env.reset()

for model_num in model_nums:
    print(f'********{model_num}********')
    # Load the trained model
    agent.load_agent(model_dir + f'/{model_num}', model_num)
    
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
