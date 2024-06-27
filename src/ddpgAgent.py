from actorCritic import Actor, Critic
from replayBuffer import ReplayBuffer, PERBuffer
from OUNoise import OUNoise
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, min_action, hidden_size_1 = 64, hidden_size_2 = 128, Prioritized_buffer = True):
        
        self.max_action = np.array(max_action)
        self.min_action = np.array(min_action)
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        self.actor = Actor(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.safety_critic = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.safety_critic_target = Critic(state_dim, action_dim, hidden_size_1, hidden_size_2).to(device)
        self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())
        self.safety_critic_optimizer = optim.Adam(self.safety_critic.parameters(), lr=1e-3)

        self.Prioritized_buffer = Prioritized_buffer

        if self.Prioritized_buffer:
            self.replay_buffer = PERBuffer(max_size=1_000_000)
        else:
            self.replay_buffer = ReplayBuffer(max_size=1_000_000)

        self.gamma = 0.99
        self.tau = 0.005

        self.ou_noise = OUNoise(action_dim)

        self.cnt = 0
        self.UPDATE_INTERVAL = 20

    def select_action(self, state, noise_enabled=True):
        self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise_enabled:
            action += self.ou_noise.noise()
        # Scale action to [min_action, max_action]

        action = self.min_action + (action + 1.0) * 0.5 * (self.max_action - self.min_action)
        return action.clip(self.min_action, self.max_action)

    def train(self, batch_size):
        if self.Prioritized_buffer:
            transitions, indices, weights = self.replay_buffer.sample(batch_size)

            # Unzip transitions into separate variables
            states, actions, rewards, next_states, costs, dones = zip(*transitions)
        else:
            states, actions, rewards, next_states, costs, dones = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.array(states)).to(device)
        action = torch.FloatTensor(np.array(actions)).to(device)
        next_state = torch.FloatTensor(np.array(next_states)).to(device)

        reward = torch.FloatTensor(np.array(rewards)).to(device).view(-1, 1)
        done = torch.FloatTensor(np.array(dones)).to(device).view(-1, 1)
        cost = torch.FloatTensor(np.array(costs)).to(device).view(-1, 1)

        # Compute the target Q value for reward critic
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1 - done) * self.gamma * target_Q

        target_Q = target_Q.view(-1)

        # Compute the target Q value for safety critic
        target_S = self.safety_critic_target(next_state, self.actor_target(next_state))
        target_S = cost + (1 - done) * self.gamma * target_S

        target_S = target_S.view(-1)

        # Get current Q estimate for reward critic
        current_Q = self.critic(state, action).view(-1)

        # Get current Q estimate for safety critic
        current_S = self.safety_critic(state, action).view(-1)

        # Compute TD errors for prioritization
        reward_errors = torch.abs(current_Q - target_Q).cpu().data.numpy()
        safety_errors = torch.abs(current_S - target_S).cpu().data.numpy()

        errors = reward_errors + safety_errors

        if self.Prioritized_buffer:
            # Update priorities in PER buffer
            self.replay_buffer.update_priorities(indices, errors)

            # Compute critic loss with importance sampling weights
            critic_loss = (torch.FloatTensor(weights).to(device) * nn.MSELoss()(current_Q, target_Q)).mean()
            safety_critic_loss = (torch.FloatTensor(weights).to(device) * nn.MSELoss()(current_S, target_S)).mean()
        else:
            critic_loss = nn.MSELoss()(current_Q, target_Q).mean()
            safety_critic_loss = nn.MSELoss()(current_S, target_S).mean()

        # Optimize the reward critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize the safety critic
        self.safety_critic_optimizer.zero_grad()
        safety_critic_loss.backward()
        self.safety_critic_optimizer.step()

        # Compute actor loss considering both reward and safety critics
        actor_loss = -self.critic(state, self.actor(state)).mean() + self.safety_critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models every UPDATE_INTERVAL steps to stabilize training

        self.cnt = (self.cnt + 1) % self.UPDATE_INTERVAL

        if self.cnt == 0:
            self.update_target_network(self.critic, self.critic_target)
            self.update_target_network(self.safety_critic, self.safety_critic_target)
            self.update_target_network(self.actor, self.actor_target)


    def update_target_network(self, network, target_network):
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save_agent(self, artifacts_path, episode):
        dir_path = os.path.join(artifacts_path, f'{episode}')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        paths = {
            'actor': os.path.join(dir_path, f'actor_{episode}.pth'),
            'critic': os.path.join(dir_path, f'critic_{episode}.pth'),
            'actor_target': os.path.join(dir_path, f'actor_target_{episode}.pth'),
            'critic_target': os.path.join(dir_path, f'critic_target_{episode}.pth'),
            'safety_critic': os.path.join(dir_path, f'safety_critic_{episode}.pth'),
            'safety_critic_target': os.path.join(dir_path, f'safety_critic_target_{episode}.pth'),
            'replay_buffer': os.path.join(dir_path, f'replay_buffer_{episode}.pickle')
        }

        for element, path in paths.items():
            getattr(self, element).save(path)


    def load_agent(self, dir_path, episode):
        
        paths = {
            'actor': os.path.join(dir_path, f'actor_{episode}.pth'),
            'critic': os.path.join(dir_path, f'critic_{episode}.pth'),
            'actor_target': os.path.join(dir_path, f'actor_target_{episode}.pth'),
            'critic_target': os.path.join(dir_path, f'critic_target_{episode}.pth'),
            'safety_critic': os.path.join(dir_path, f'safety_critic_{episode}.pth'),
            'safety_critic_target': os.path.join(dir_path, f'safety_critic_target_{episode}.pth'),
            'replay_buffer': os.path.join(dir_path, f'replay_buffer_{episode}.pickle')
        }

        for element, path in paths.items():
            if os.path.exists(path):
                getattr(self, element).load(path)

        
        