import logging
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from agents import *

from Utils.agents import *
from Utils.models import *
from Utils.tools import *

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

    # Hyperparameters
    batch_size = 128  # Size of the batch to train the neural networks
    n_episodes = 10000 # Number of episodes to run when training the agent
    n_batches_train = 1 # Number of times to train for each time step
    exp_replay_buffer_size = int(2e5) # Experience replay buffer size
    epsilon_decay = 0.9925 # Decay of the exploration constant
    epsilon = 1 # Initial value of the exploration constant
    epsilon_final = 0.1 # Final value of the exploration constant
    plot_every_n = 10 # Period to update the rewards chart
    save_every_n = 100 # Period to save the model if an improvement has been found
    tau = 0.001 # Parameter that controls how fast the local networks update the target networks
    gamma = 0.99 # Discount factor

    # Initializing the agent
    agent = DDPGAgent(Critic, Actor, state_size=state_size, action_size=action_size, 
                    tau=tau, gamma=gamma, batch_size=batch_size, buffer_size=exp_replay_buffer_size,
                    num_batches=n_batches_train)
    scores = [] 
    epsilons = []
    max_score = 0
    score_list = []

    # Loop over episodes
    for episode in range(n_episodes):
        epsilons.append(epsilon)
        epsilon = epsilon_decay * epsilon + (1 - epsilon_decay) * epsilon_final

        # Reset environment
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        state = decision_steps.obs[0]  # Initial state
        score = 0
        done = [False]  # Track episode termination


        while not any(done):
            # Agent takes action
            action = agent.act(state, epsilon=epsilon)

            # Create ActionTuple
            action_tuple = ActionTuple(continuous=action)

            # Set actions and step
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # Get updated steps
            new_decision_steps, new_terminal_steps = env.get_steps(behavior_name)

            # Check termination
            if len(new_terminal_steps) > 0:
                next_state = new_terminal_steps.obs[0]
                reward = new_terminal_steps.reward
                done = [True]
            else:
                next_state = new_decision_steps.obs[0]
                reward = new_decision_steps.reward
                done = [False]

            # Update agent
            agent.step(state, action, reward, next_state, done)

            # Update state and score
            state = next_state
            score += np.sum(reward)

        score_list.append(score)

        if episode % 10 == 0:
            logging.info(score_list)
            logging.info(f'\n----- Average score : {np.mean(score_list)}, ----- Standard deviation : {np.std(score_list)} -----\n')
            score_list = []

        

except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    env.close()