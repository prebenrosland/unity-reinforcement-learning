import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, action_size)
        

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.2, epsilon_decay=0.995, batch_size=1024):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Q-network and target Q-network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Experience replay
        self.memory = deque(maxlen=50000)

        self.steps = 0
        
        # Initialize target network weights
        self.update_target_network()


    def update_target_network(self):
    # Update target network periodically (e.g., every 10 episodes)
        if self.steps % 200 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


    def remember(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.float32)

        self.memory.append((state, action, reward, next_state, done))

    
    def epsilon_greedy(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.uniform(-1, 1, self.action_size)  # Random continuous action
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Ensure batch dimension
            q_values = self.q_network(state_tensor)

            action = q_values.squeeze(0).detach().numpy()  # Convert tensor to NumPy array
        
        return action  # Continuous values


    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.stack(actions)  # Continuous actions
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Get Q-values for current states
        q_values = self.q_network(states)
        
        # Get the Q-values for the next states from the target network
        next_q_values = self.target_network(next_states)
        
        # Calculate the target Q-values (using max Q-value for next state)
        next_q_value = next_q_values.max(dim=1)[0]
        target_q_values = rewards + (self.gamma * next_q_value * ~dones)

        # Compute the loss (no need to gather actions for continuous action space)
        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))  # Match shape for loss computation

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon after each replay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



    def update(self, state, reward, next_state, done):
        self.steps += 1
        if len(self.memory) > self.batch_size * 10:
            self.replay()  # Only update when enough experiences are stored


def preprocess_observations(obs):
    # Example: Flatten the observations (if they are multi-dimensional like images)
    return np.reshape(obs, (-1,))