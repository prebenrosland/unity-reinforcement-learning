import time
import logging
import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

from Utils.models import *
from Utils.agents import *

logging.basicConfig(level=logging.INFO)

# Configuration
channel = EngineConfigurationChannel()

try:
    logging.info("Starting Unity Environment...")
    env = UnityEnvironment(file_name=None, side_channels=[channel])
    logging.info("Unity Environment connected successfully!")
    
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    ACTION_MAP = {
        0: [1.0, -1.0, 0.0],  # steer left + throttle
        1: [1.0, 0.0, 0.0],   # throttle
        2: [1.0, 1.0, 0.0],   # steer right + throttle
        3: [0.0, 0.0, 1.0],   # brake
        4: [0.5, 0.0, 0.0],   # slow throttle
        5: [0.0, -1.0, 0.0],  # steer left + no throttle
        6: [0.0, 1.0, 0.0],   # steer right + no throttle
        7: [1.0, -0.5, 0.0],  # steer light left + throttle
        8: [1.0, 0.5, 0.0],   # steer light right + throttle
        9: [0.0, -0.5, 0.0],  # steer light left
        10:[0.0, 0.5, 0.0],   # steer light right
    }

    # Initializing an agent with state size and action size(corresponding to ACTION_MAP)
    agent = DQNAgent(
        state_size=spec.observation_specs[0].shape[0],
        action_size=len(ACTION_MAP)
    )

    # Load the model weights from saved weights
    agent.q_network.load_state_dict(torch.load("Weights/dqn/checkpoint_dqn.pth"))
    agent.q_network.eval()

    for episode in range(100):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        episode_reward = 0
        
        while True:
            current_state = decision_steps.obs[0][0]
            action_idx = agent.get_action(current_state)
            
            # Create and send a continuous action to Unity based on the discrete actions created in ACTION_MAP
            action_tuple = ActionTuple()
            action_tuple.add_continuous(np.array([ACTION_MAP[action_idx]], dtype=np.float32))
            env.set_actions(behavior_name, action_tuple)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            reward = 0
            next_state = current_state
            done = False
            
            if len(terminal_steps) > 0:
                reward = terminal_steps.reward[0]
                next_state = terminal_steps.obs[0][0]
                done = True
            elif len(decision_steps) > 0:
                reward = decision_steps.reward[0]
                next_state = decision_steps.obs[0][0]

            episode_reward += reward
            if done:
                break

        logging.info(f"----- Episode : {episode} Reward : {episode_reward:.2f}\t-----")

except Exception as e:
    logging.error(f"Error: {str(e)}")
finally:
    env.close()
