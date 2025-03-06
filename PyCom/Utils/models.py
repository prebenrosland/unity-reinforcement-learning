import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        output = F.tanh(self.fc3(x))
        return output

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128 + action_dim, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, actions):
        x = F.relu(self.fc1(state))
        x = self.bn(x)
        x = torch.cat((x, actions), dim=1)
        x = F.relu(self.fc2(x))
        output = F.tanh(self.fc3(x))
        return output