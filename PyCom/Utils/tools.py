import numpy as np
import random
from collections import deque
import torch


class ReplayBuffer():
    def __init__(self, size=100000):
        self.size = size
        self.reset()

    def length(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = deque(maxlen=self.size)

    def append(self, observation):
        self.buffer.append(observation)

    def draw_sample(self, sample_size):
        sample = random.choices(self.buffer, k=sample_size)
        states, actions, rewards, next_states, dones = zip(*sample)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state