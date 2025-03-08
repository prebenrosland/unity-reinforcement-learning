import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, layer1=128, layer2=128, layer3=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, layer1)
        self.bn = nn.BatchNorm1d(layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, action_dim)
        #self.init_weights()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        output = F.tanh(self.fc3(x))
        return output
    
    def init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.uniform_(self.fc3.weight, -3e-3, 3e-3)
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, layer1=128, layer2=128, layer3=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, layer1)
        self.bn = nn.BatchNorm1d(layer1)
        self.fc2 = nn.Linear(layer1 + action_dim, layer2)
        self.fc3 = nn.Linear(layer2, 1)
        #self.init_weights()

    def forward(self, state, actions):
        x = F.relu(self.fc1(state))
        x = self.bn(x)
        x = torch.cat((x, actions), dim=1)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output
    
    def init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.uniform_(self.fc3.weight, -3e-3, 3e-3)