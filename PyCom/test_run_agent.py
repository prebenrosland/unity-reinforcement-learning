import logging
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from agents import *

from Utils.agents import *
from Utils.models import *
from Utils.tools import *


batch_size = 128
n_episodes = 100
n_batches_train = 1
exp_replay_buffer_size = int(2e5)
epsilon_decay = 0.9925
epsilon = 1
epsilon_final = 0.1
tau = 0.001
gamma = 0.99


logging.basicConfig(level=logging.DEBUG)

# Configuration
channel = EngineConfigurationChannel()

try:
    logging.info("Starting Unity Environment...")

    # Connect to ML-Agents Unity Environment
    env = UnityEnvironment(file_name=None, side_channels=[channel])
    
    logging.info("Unity Environment connected successfully!")

    # Reset the environment, no need to store return value
    env.reset()

    # Check behaviors (Agent brains)
    behavior_names = list(env.behavior_specs.keys())
    if not behavior_names:
        logging.warning("No behaviors found!")
        exit()

    behavior_name = behavior_names[0]
    logging.info(f"Using Behavior: {behavior_name}")
    behavior_spec = env.behavior_specs[behavior_name]

    # Get action dimensions from agent
    continuous_action_size = behavior_spec.action_spec.continuous_size
    discrete_action_size = behavior_spec.action_spec.discrete_size
    logging.info(f"Continuous Action Size: {continuous_action_size}")
    logging.info(f"Discrete Action Size: {discrete_action_size}")

    # Defining amount of action and state sizes
    state_size = behavior_spec.observation_specs[0].shape[0]
    action_size = continuous_action_size

    list_rewards = []

    # Initializing the agent network
    agent = DDPGAgent(critic=Critic, actor=Actor, state_size=state_size, action_size=action_size, 
                    tau=tau, gamma=gamma, batch_size=batch_size, buffer_size=exp_replay_buffer_size,
                    num_batches=n_batches_train)
    
    agent.actor.load_state_dict(torch.load('Weights/checkpoint_actor.pth'))
    agent.actor_target.load_state_dict(torch.load('Weights/checkpoint_target_actor.pth'))
    agent.critic.load_state_dict(torch.load('Weights/checkpoint_critic.pth'))
    agent.critic_target.load_state_dict(torch.load('Weights/checkpoint_target_critic.pth'))
    
    epsilons = []
    score_list = []

    # Loop over episodes
    for episode in range(n_episodes):
        logging.info(f"Starting episode {episode+1}/{n_episodes}")

        # Reset environment
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        state = decision_steps.obs[0]  # Initial state
        score = 0
        step = 0
        done = [False]  # Track episode termination

        while not any(done):
            step += 1
            action = agent.act(state, epsilon=epsilon)

            action_tuple = ActionTuple(continuous=action)

            env.set_actions(behavior_name, action_tuple)
            env.step()

            new_decision_steps, new_terminal_steps = env.get_steps(behavior_name)

            if len(new_terminal_steps) > 0:
                next_state = new_terminal_steps.obs[0]
                reward = new_terminal_steps.reward
                done = [True]
            else:
                next_state = new_decision_steps.obs[0]
                reward = new_decision_steps.reward
                done = [False]

            state = next_state
            score += np.sum(reward)

        score_list.append(score)
        logging.info(f"Episode {episode+1} finished with score: {score}")

except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    env.close()