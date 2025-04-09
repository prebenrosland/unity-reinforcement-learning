import random
import torch
from torch import optim
from collections import deque

from .models import *
from .tools import *


class DDPGAgent():
    def __init__(self, critic, actor, state_size, action_size, target_network_update_rate, discount_factor, buffer_size, batch_size, num_batches, alpha=0):
        
        self.critic = critic(state_size, action_size)
        self.critic_target = critic(state_size, action_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=2e-4)

        self.actor = actor(state_size, action_size)
        self.actor_target = actor(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=2e-4)

        self.state_size = state_size
        self.action_size = action_size
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.alpha = alpha
        self.target_network_update_rate = target_network_update_rate
        self.gamma = discount_factor

        self.noise = OUNoise(action_size)

        self._target_update(target_network_update_rate=1)
        

    def step(self, states, actions, rewards, next_states, done):
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, done):
            self.replay_buffer.append([s, a, r, ns, d])
        
        for _ in range(self.num_batches):
            states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = self.replay_buffer.draw_sample(self.batch_size)

            effort = torch.sqrt((actions_batch**2).sum(dim=1))
            effort  = torch.unsqueeze(effort, dim=1)
            
            rewards_batch -= effort * self.alpha

            if self.replay_buffer.length() > self.batch_size:
                self.update_actor(states_batch)
                self.update_critic(states_batch, actions_batch, rewards_batch, next_states_batch, done_batch)
                self._target_update(self.target_network_update_rate)


    def update_actor(self, states):
        actor_actions = self.actor.forward(states)
        critic_values = self.critic.forward(states, actor_actions)

        loss = -critic_values.mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        

    def update_critic(self, states, actions, rewards, next_states, done):
        actor_actions = self.actor_target.forward(next_states)
        best_q = self.critic_target.forward(next_states, actor_actions)
        target_q = rewards + best_q * self.gamma * (1 - done)

        q_value = self.critic.forward(states, actions)

        loss = F.mse_loss(q_value, target_q)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

    def act(self, states, epsilon=1):
        states = np.array(states)
        states = torch.from_numpy(states).float()

        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)

        self.actor.eval()
        with torch.no_grad():
            actions = self.actor.forward(states).cpu().data.numpy()
        self.actor.train()

        actions += self.noise.sample() * epsilon
        actions = np.clip(actions, -1, 1)
        return actions
    
    def reset(self):
        self.noise.reset()

    def _target_update(self, target_network_update_rate=None):
        self._update_target_network(self.critic_target, self.critic, target_network_update_rate)
        self._update_target_network(self.actor_target, self.actor, target_network_update_rate)

    def _update_target_network(self, target_network, local_network, target_network_update_rate):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_((1.0 - target_network_update_rate) * target_param.data + target_network_update_rate * local_param.data)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.q_network = QNetwork(state_size, 300, 400, action_size)
        self.target_network = QNetwork(state_size, 300, 400, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.memory = []
        self.batch_size = 256
        self.update_factor = 100

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        # Calculating q-value and target-q to update network
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones.float()) * self.gamma * next_q

        loss = torch.nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decrease exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Hard updating target network
        if self.epsilon % self.update_factor == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())