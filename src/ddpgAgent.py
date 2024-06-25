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
    def __init__(self, state_dim, action_dim, max_action, min_action, Prioritized_buffer = False):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.Prioritized_buffer = Prioritized_buffer

        if self.Prioritized_buffer:
            self.replay_buffer = PERBuffer(max_size=1_000_00)
        else:
            self.replay_buffer = ReplayBuffer(max_size=1_000_00)

        self.gamma = 0.99
        self.tau = 0.005
        self.max_action = max_action
        self.min_action = min_action

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
        reward = torch.FloatTensor(np.array(rewards)).to(device)
        next_state = torch.FloatTensor(np.array(next_states)).to(device)
        done = torch.FloatTensor(np.array(dones)).to(device)
        cost = torch.FloatTensor(np.array(costs)).to(device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        reward = reward.view(-1, 1)
        done = done.view(-1, 1)
        
        target_Q = reward + (1 - done) * self.gamma * target_Q

        # Compute CBF loss
        cbf_loss = torch.max(torch.zeros_like(cost), cost).mean()  # Ensure cost is non-negative

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute TD errors for prioritization
        errors = torch.abs(current_Q - target_Q).cpu().data.numpy()
        
        if self.Prioritized_buffer:
            # Update priorities in PER buffer
            self.replay_buffer.update_priorities(indices, errors)

            # Compute critic loss with importance sampling weights
            critic_loss = (torch.FloatTensor(weights).to(device) * nn.MSELoss()(current_Q, target_Q)).mean()
        else:
            critic_loss = nn.MSELoss()(current_Q, target_Q).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        (critic_loss + cbf_loss).backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models every UPDATE_INTERVAL steps to stabilize training

        self.cnt = (self.cnt + 1)%self.UPDATE_INTERVAL

        if self.cnt == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_agent(self, artifacts_path, episode):
        dir_path = os.path.join(artifacts_path, f'{episode}')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        actor_path = os.path.join(dir_path, f'actor_{episode}.pth')
        critic_path = os.path.join(dir_path, f'critic_{episode}.pth')
        actor_target_path = os.path.join(dir_path, f'actor_target_{episode}.pth')
        critic_target_path = os.path.join(dir_path, f'critic_target_{episode}.pth')
        buffer_path = os.path.join(dir_path, f'replay_buffer_{episode}.pickle')

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.actor_target.state_dict(), actor_target_path)
        torch.save(self.critic_target.state_dict(), critic_target_path)
        self.replay_buffer.save(buffer_path)

    def load_agent(self, dir_path, episode):
        actor_path = os.path.join(dir_path, f'actor_{episode}.pth')
        critic_path = os.path.join(dir_path, f'critic_{episode}.pth')
        actor_target_path = os.path.join(dir_path, f'actor_target_{episode}.pth')
        critic_target_path = os.path.join(dir_path, f'critic_target_{episode}.pth')
        buffer_path = os.path.join(dir_path, f'replay_buffer_{episode}.pickle')

        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path))
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))
        if os.path.exists(actor_target_path):
            self.actor.load_state_dict(torch.load(actor_target_path))
        if os.path.exists(critic_target_path):
            self.critic.load_state_dict(torch.load(critic_target_path))
        if os.path.exists(buffer_path):
            self.replay_buffer.load(buffer_path)
        
        