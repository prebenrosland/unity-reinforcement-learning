import time
import logging
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import torch

logging.basicConfig(level=logging.INFO)

# Configuration
channel = EngineConfigurationChannel()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_size)
        )
        self.target_network = torch.nn.Sequential(
            torch.nn.Linear(state_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_size)
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.memory = []
        self.batch_size = 256  # Smaller batch size for quicker updates

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

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones.float()) * self.gamma * next_q

        loss = torch.nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        if self.epsilon % 10 == 0:  # Update target network periodically
            self.target_network.load_state_dict(self.q_network.state_dict())


try:
    logging.info("Starting Unity Environment...")
    env = UnityEnvironment(file_name=None, side_channels=[channel])
    logging.info("Unity Environment connected successfully!")
    
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    agent = DQNAgent(
        state_size=spec.observation_specs[0].shape[0],
        action_size=3  # Discrete actions: left, neutral, right
    )

    ACTION_MAP = {
        0: [-1.0, 0.0, 1.0],  # Steer left, no throttle, brake
        1: [0.0, 1.0, 0.0],    # Center steering, throttle, no brake
        2: [1.0, 0.0, 0.0]     # Steer right, no throttle, no brake
    }

    for episode in range(10000):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        episode_reward = 0
        
        while True:
            current_state = decision_steps.obs[0][0]
            action_idx = agent.get_action(current_state)
            
            action_tuple = ActionTuple()
            action_tuple.add_continuous(np.array([ACTION_MAP[action_idx]], dtype=np.float32))
            env.set_actions(behavior_name, action_tuple)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            reward = 0.0
            next_state = current_state
            done = False
            
            if len(terminal_steps) > 0:
                reward = terminal_steps.reward[0]
                next_state = terminal_steps.obs[0][0]
                done = True
            elif len(decision_steps) > 0:
                reward = decision_steps.reward[0]
                next_state = decision_steps.obs[0][0]

            agent.remember(current_state, action_idx, reward, next_state, done)
            agent.replay()
            
            episode_reward += reward
            if done:
                break

        logging.info(f"Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

except Exception as e:
    logging.error(f"Error: {str(e)}")
finally:
    env.close()
