import random
import torch
from torch import optim
from collections import deque

from .models import *
from .tools import *


class DDPGAgent():
    def __init__(self, critic, actor, state_size, action_size, tau, gamma, buffer_size, batch_size, num_batches, alpha=0):
        
        self.critic = critic(state_size, action_size)
        self.critic_target = critic(state_size, action_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0005)

        self.actor = actor(state_size, action_size)
        self.actor_target = actor(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0005)

        self.state_size = state_size
        self.action_size = action_size
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma

        self.noise = OUNoise(action_size)

        self._soft_target_update(tau=1)
        

    def step(self, states, actions, rewards, next_states, done, epsilon=1):
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, done):
            self.replay_buffer.append([s, a, r, ns, d])
        
        for _ in range(self.num_batches):
            states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = self.replay_buffer.draw_sample(self.batch_size)

            effort = torch.sqrt((actions_batch**2).sum(dim=1))
            effort  = torch.unsqueeze(effort, dim=1)
            
            rewards_batch -= effort * self.alpha

            if self.replay_buffer.length() > self.buffer_size:
                self.update_actor(states_batch)
                self.update_critic(states_batch, actions_batch, rewards_batch, next_states_batch, done_batch)


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

    def _soft_target_update(self, tau=None):
        self._update_target_network(self.critic_target, self.critic, tau)
        self._update_target_network(self.actor_target, self.actor, tau)

    def _update_target_network(self, target_network, local_network, tau):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * local_param.data)