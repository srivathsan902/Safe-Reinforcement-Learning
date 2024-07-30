import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def save(self, path):
        '''
        Save the model, parameters, and optimizer state
        '''
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_size_1': self.hidden_size_1,
            'hidden_size_2': self.hidden_size_2
        }, path)

    def load(self, path):
        '''
        Load the model, parameters, and optimizer state
        '''
        checkpoint = torch.load(path)
        self.hidden_size_1 = checkpoint.get('hidden_size_1', 64)
        self.hidden_size_2 = checkpoint.get('hidden_size_2', 128)
        self.load_state_dict(checkpoint.get('model_state_dict', checkpoint))


class Actor(BaseModel):
    def __init__(self, state_dim, action_dim, hidden_size_1, hidden_size_2):
        super(Actor, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.layer1 = nn.Linear(state_dim, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, action_dim)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x

class Critic(BaseModel):
    def __init__(self, state_dim, action_dim, hidden_size_1, hidden_size_2):
        super(Critic, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
